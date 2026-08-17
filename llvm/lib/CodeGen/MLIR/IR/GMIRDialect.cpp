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
// unabstracted per-op logic rather than a shared two-op helper.
LogicalResult UnmergeOp::verify() {
  if (getDsts().empty())
    return emitOpError("expected at least one result");
  mlir::Type Ty = getDsts().front().getType();
  for (mlir::Value Dst : getDsts().drop_front())
    if (Dst.getType() != Ty)
      return emitOpError("all results must have the same type");
  return success();
}

LogicalResult MergeOp::verify() {
  if (getSrcs().empty())
    return emitOpError("expected at least one operand");
  mlir::Type Ty = getSrcs().front().getType();
  for (mlir::Value Src : getSrcs().drop_front())
    if (Src.getType() != Ty)
      return emitOpError("all operands must have the same type");
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
