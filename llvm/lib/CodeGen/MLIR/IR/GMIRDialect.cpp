//===- GMIRDialect.cpp - GMIR dialect implementation ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Registers the `gmir` dialect and its `!gmir.llt` type.
//
//===----------------------------------------------------------------------===//

#include "GMIRDialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/AtomicOrdering.h"

using namespace mlir;
using namespace llvm::gmir;

#include "IR/GMIRDialect.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "IR/GMIRDialectTypes.cpp.inc"

#define GET_OP_CLASSES
#include "IR/GMIRDialectOps.cpp.inc"

// ConstantOp::fold -- see GMIRDialect.td's `hasFolder = 1` comment on
// gmir.constant: a ConstantLike op must return its own value attribute
// from fold() (mirrors e.g. mlir::arith::ConstantOp::fold), which is what
// callers like the greedy pattern rewrite driver's constant-CSE step rely
// on via matchPattern(op, m_Constant()).
OpFoldResult ConstantOp::fold(FoldAdaptor adaptor) { return getValueAttr(); }

// BranchOpInterface methods for GMIR_BrOp/GMIR_CondBrOp -- ODS only
// declares these (DeclareOpInterfaceMethods), it doesn't define them.
// Mirrors mlir::cf::BranchOp/CondBranchOp exactly
// (mlir/lib/Dialect/ControlFlow/IR/ControlFlowOps.cpp:290-297,573-584).
// Note: llvm::Attribute (IR function/param attributes) and mlir::Attribute
// are distinct types that collide under this file's `using namespace mlir;`
// plus the ambient `using namespace llvm;` pulled in via the CodeGen PCH --
// explicitly qualified below for the same reason M1's Value/Type/DenseMap
// ambiguities were.
mlir::SuccessorOperands BrOp::getSuccessorOperands(unsigned index) {
  assert(index == 0 && "invalid successor index");
  return mlir::SuccessorOperands(getDestOperandsMutable());
}

Block *BrOp::getSuccessorForOperands(ArrayRef<mlir::Attribute>) {
  return getDest();
}

mlir::SuccessorOperands CondBrOp::getSuccessorOperands(unsigned index) {
  assert(index < 2 && "invalid successor index");
  return mlir::SuccessorOperands(index == 0 ? getTrueDestOperandsMutable()
                                             : getFalseDestOperandsMutable());
}

Block *CondBrOp::getSuccessorForOperands(ArrayRef<mlir::Attribute> operands) {
  if (mlir::IntegerAttr condAttr =
          dyn_cast_or_null<mlir::IntegerAttr>(operands.front()))
    return condAttr.getValue().isOne() ? getTrueDest() : getFalseDest();
  return nullptr;
}

// MemoryEffectsOpInterface methods for GMIR_LoadOp/GMIR_StoreOp -- ODS only
// declares these (DeclareOpInterfaceMethods), same reason as BranchOpInterface
// above. Mirrors mlir::LLVM::LoadOp/StoreOp::getEffects exactly
// (mlir/lib/Dialect/LLVMIR/IR/LLVMDialect.cpp:839-853,918-932): a plain
// Read/Write on the pointer operand, plus a conservative extra Read+Write
// pair when the access is volatile or "stronger than unordered" atomic,
// since such accesses can have target-specific effects beyond the single
// pointed-to location.
// mlir::MemoryEffects/mlir::SideEffects collide with llvm::MemoryEffects
// (IR-level function/call memory-effect attributes) under this file's
// blanket `using namespace mlir;` + the ambient `using namespace llvm;` --
// same class of ambiguity as Value/Type/Attribute/DataLayout above,
// explicitly qualified for the same reason.
// "Stronger than unordered": NotAtomic and Unordered both get no extra
// effects, matching mlir::LLVM::LoadOp/StoreOp exactly -- the raw `!= 0`
// check this used to be also flagged Unordered, contradicting this very
// comment. Reuses llvm::isStrongerThanUnordered (AtomicOrdering.h) rather
// than a hand-rolled `>` comparison: AtomicOrdering deliberately deletes
// operator> (comparing orderings directly is a common source of bugs,
// e.g. Acquire vs Release aren't ordered relative to each other even
// though both are "stronger than Unordered").
static bool isOrderingStrongerThanUnordered(mlir::IntegerAttr orderingAttr) {
  return llvm::isStrongerThanUnordered(
      static_cast<llvm::AtomicOrdering>(orderingAttr.getInt()));
}

void LoadOp::getEffects(SmallVectorImpl<mlir::SideEffects::EffectInstance<
                            mlir::MemoryEffects::Effect>> &effects) {
  effects.emplace_back(mlir::MemoryEffects::Read::get(), &getPtrMutable());
  if (getIsVolatile() || isOrderingStrongerThanUnordered(getOrderingAttr())) {
    effects.emplace_back(mlir::MemoryEffects::Write::get());
    effects.emplace_back(mlir::MemoryEffects::Read::get());
  }
}

void StoreOp::getEffects(SmallVectorImpl<mlir::SideEffects::EffectInstance<
                             mlir::MemoryEffects::Effect>> &effects) {
  effects.emplace_back(mlir::MemoryEffects::Write::get(), &getPtrMutable());
  if (getIsVolatile() || isOrderingStrongerThanUnordered(getOrderingAttr())) {
    effects.emplace_back(mlir::MemoryEffects::Write::get());
    effects.emplace_back(mlir::MemoryEffects::Read::get());
  }
}

// Hand-written verifiers for GMIR_UnmergeOp/GMIR_MergeOp (`hasVerifier = 1`
// in GMIRDialect.td -- ODS only declares these, it doesn't define them,
// same as the BranchOpInterface/MemoryEffectsOpInterface methods above).
// No stock ODS trait expresses "all *results* share one type"
// (SameOperandsAndResultType doesn't fit either op: gmir.unmerge's single
// operand legitimately differs in type from its results, and gmir.merge's
// single result legitimately differs from its operands), so each is a
// plain hand-rolled check, matching this file's existing style of direct,
// unabstracted per-op logic rather than a shared two-op helper. Besides the
// "all same type" check, also require the piece count times that shared
// width to add up to the wide side's width (mirrors buildUnmerge/
// buildMergeValues's own precondition: "the entire register (and no more)
// must be covered by the input registers") -- with
// `useDefaultTypePrinterParser` now on, gmir IR round-trips through text, so a
// width mismatch here is no longer something only correctly-constructed C++
// callers can produce.
LogicalResult UnmergeOp::verify() {
  if (getDsts().empty())
    return emitOpError("expected at least one result");
  auto Ty = cast<gmir::LLTType>(getDsts().front().getType());
  for (mlir::Value Dst : getDsts().drop_front())
    if (Dst.getType() != Ty)
      return emitOpError("all results must have the same type");
  auto SrcTy = cast<gmir::LLTType>(getSrc().getType());
  if (Ty.getScalarSizeInBits() == 0 || SrcTy.getScalarSizeInBits() == 0)
    return emitOpError("expected non-pointer operand and result types");
  if (Ty.getNumElements() != 0)
    return emitOpError("results must be scalar");
  // Two distinct shapes, both real G_UNMERGE_VALUES uses (see the op's
  // doc comment): a vector source splits into its per-element scalar
  // lanes (count-based -- each dst is one element, not a bit-width
  // fraction), while a scalar source splits into narrower bit-width
  // chunks (the original, sum-based check).
  if (SrcTy.getNumElements() != 0) {
    if (SrcTy.getScalarSizeInBits() != Ty.getScalarSizeInBits())
      return emitOpError("results must match the operand's element width");
    if (SrcTy.getNumElements() != getDsts().size())
      return emitOpError("result count must match the operand's element count");
    return success();
  }
  if (SrcTy.getScalarSizeInBits() !=
      Ty.getScalarSizeInBits() * getDsts().size())
    return emitOpError("result bit widths must sum to the operand's width");
  return success();
}

LogicalResult MergeOp::verify() {
  if (getSrcs().empty())
    return emitOpError("expected at least one operand");
  auto Ty = cast<gmir::LLTType>(getSrcs().front().getType());
  for (mlir::Value Src : getSrcs().drop_front())
    if (Src.getType() != Ty)
      return emitOpError("all operands must have the same type");
  auto DstTy = cast<gmir::LLTType>(getDst().getType());
  if (Ty.getScalarSizeInBits() == 0 || DstTy.getScalarSizeInBits() == 0)
    return emitOpError("expected non-pointer operand and result types");
  if (DstTy.getScalarSizeInBits() !=
      Ty.getScalarSizeInBits() * getSrcs().size())
    return emitOpError("operand bit widths must sum to the result's width");
  return success();
}

// Hand-written verifier for GMIR_BuildVectorOp (`hasVerifier = 1` in
// GMIRDialect.td -- see its doc comment for why this is a distinct op
// from gmir.merge rather than a shared one): checks the count- and
// element-width-based relationship a vector build depends on, the
// vector-lane analogue of MergeOp::verify()'s bit-width-sum check above.
LogicalResult BuildVectorOp::verify() {
  if (getSrcs().empty())
    return emitOpError("expected at least one operand");
  auto Ty = cast<gmir::LLTType>(getSrcs().front().getType());
  for (mlir::Value Src : getSrcs().drop_front())
    if (Src.getType() != Ty)
      return emitOpError("all operands must have the same type");
  auto DstTy = cast<gmir::LLTType>(getDst().getType());
  if (Ty.getScalarSizeInBits() == 0 || DstTy.getScalarSizeInBits() == 0)
    return emitOpError("expected non-pointer operand and result types");
  if (Ty.getNumElements() != 0)
    return emitOpError("operands must be scalar");
  if (DstTy.getNumElements() != getSrcs().size())
    return emitOpError("result element count must match the operand count");
  if (DstTy.getScalarSizeInBits() != Ty.getScalarSizeInBits())
    return emitOpError("result element width must match the operands' width");
  return success();
}

// Hand-written verifiers for GMIR_AnyExtOp/GMIR_TruncOp (`hasVerifier = 1`
// in GMIRDialect.td, same reason as UnmergeOp/MergeOp above): each checks
// that the width relationship the op's whole purpose depends on actually
// holds -- no stock ODS trait expresses "result strictly wider/narrower
// than the operand".
LogicalResult AnyExtOp::verify() {
  auto SrcTy = cast<gmir::LLTType>(getSrc().getType());
  auto ResTy = cast<gmir::LLTType>(getResult().getType());
  if (SrcTy.getScalarSizeInBits() == 0 || ResTy.getScalarSizeInBits() == 0)
    return emitOpError("expected non-pointer operand and result types");
  if (ResTy.getScalarSizeInBits() <= SrcTy.getScalarSizeInBits())
    return emitOpError("result must be wider than the operand");
  return success();
}

LogicalResult TruncOp::verify() {
  auto SrcTy = cast<gmir::LLTType>(getSrc().getType());
  auto ResTy = cast<gmir::LLTType>(getResult().getType());
  if (SrcTy.getScalarSizeInBits() == 0 || ResTy.getScalarSizeInBits() == 0)
    return emitOpError("expected non-pointer operand and result types");
  if (ResTy.getScalarSizeInBits() >= SrcTy.getScalarSizeInBits())
    return emitOpError("result must be narrower than the operand");
  return success();
}

void GMIRDialect::initialize() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "IR/GMIRDialectTypes.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "IR/GMIRDialectOps.cpp.inc"
      >();
}
