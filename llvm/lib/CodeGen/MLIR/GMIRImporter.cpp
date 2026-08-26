//===- GMIRImporter.cpp - LLVM IR -> gmir importer -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "GMIRImporter.h"
#include "IR/GMIRDialect.h"
#include "mlir/IR/Builders.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"

using namespace llvm;

/// Converts a scalar integer llvm::Type to !gmir.llt. Returns a null type
/// for anything else (vectors, pointers, floats, aggregates), or for an
/// integer wider than 64 bits: gmir.constant always stores its value in a
/// 64-bit attribute (see its doc comment), so a wider !gmir.llt would make
/// the translator's later truncate-back-down step truncate *up*, which
/// APInt::trunc() forbids and asserts on. Rejecting it here, at the single
/// shared type-conversion choke point, also correctly rejects non-constant
/// wide-integer values (e.g. two real i128 arguments added together),
/// which nothing downstream validates either. Note: both `Value` and
/// `Type` name distinct classes in ::llvm and ::mlir, so this file
/// explicitly qualifies every mlir:: type/call instead of also pulling
/// in `using namespace mlir;` alongside `using namespace llvm;` below --
/// this file implements code in ::llvm, not ::mlir, so only the former
/// is the standard-sanctioned exception to LLVM's "don't `using
/// namespace`" rule (see CodingStandards.md); the latter would just be
/// inviting exactly this Value/Type-style collision instead of avoiding
/// it.
static gmir::LLTType convertType(mlir::MLIRContext &Context, llvm::Type *Ty) {
  auto *IntTy = dyn_cast<llvm::IntegerType>(Ty);
  if (!IntTy || IntTy->getBitWidth() > 64)
    return {};
  return gmir::LLTType::get(&Context, IntTy->getBitWidth(),
                            /*numElements=*/0, /*addressSpace=*/0,
                            /*isScalable=*/false);
}

namespace {

/// Walks one llvm::Function's single basic block, building `gmir` ops.
/// Bails out (leaving ValueMap/Builder state to be discarded by the
/// caller) the moment it sees anything outside the currently-supported
/// subset.
class FunctionImporter {
public:
  FunctionImporter(mlir::MLIRContext &Context, mlir::OpBuilder &Builder)
      : Context(Context), Builder(Builder) {}

  /// Returns false the moment an unsupported construct is seen.
  bool importBody(Function &F, mlir::func::FuncOp FuncOp) {
    // Memoized constants (see getOperand) are materialized at a fixed,
    // growing-forward cursor at the very front of the entry block, mirroring
    // IRTranslator's own dedicated EntryBuilder: the entry block dominates
    // every other block, and ops preceding all others in it precede
    // everything within it too, so any later reuse of a memoized constant's
    // mlir::Value is guaranteed to dominate its use. Without this, a
    // ConstantInt shared by two non-dominating blocks (LLVM interns/uniques
    // ConstantInts, so this is routine once multiple blocks exist, not an
    // edge case) could get memoized wherever it was first encountered and
    // then reused from a sibling block that doesn't dominate that point.
    EntryBlock = &FuncOp.getBody().front();
    ConstantInsertPt = EntryBlock->begin();

    unsigned ArgIdx = 0;
    for (Argument &Arg : F.args())
      ValueMap[&Arg] = FuncOp.getArgument(ArgIdx++);

    for (Instruction &I : F.getEntryBlock()) {
      if (auto *Ret = dyn_cast<ReturnInst>(&I)) {
        return importReturn(*Ret);
      }
      if (auto *BinOp = dyn_cast<BinaryOperator>(&I)) {
        if (!importBinaryOp(*BinOp))
          return false;
        continue;
      }
      // Anything else (calls, loads/stores, branches, casts, ...) is out
      // of scope for now -- fall back rather than mistranslate.
      return false;
    }
    // Fell off the end without a terminator: malformed IR, shouldn't
    // happen, but bail gracefully rather than assert.
    return false;
  }

private:
  bool importBinaryOp(BinaryOperator &BinOp) {
    gmir::LLTType ResTy = convertType(Context, BinOp.getType());
    if (!ResTy)
      return false;
    mlir::Value LHS, RHS;
    if (!getOperand(BinOp.getOperand(0), LHS) ||
        !getOperand(BinOp.getOperand(1), RHS))
      return false;

    mlir::Location Loc = Builder.getUnknownLoc();
    mlir::Operation *Op;
    switch (BinOp.getOpcode()) {
    case Instruction::Add:
      Op = gmir::AddOp::create(Builder, Loc, ResTy, LHS, RHS);
      break;
    case Instruction::Sub:
      Op = gmir::SubOp::create(Builder, Loc, ResTy, LHS, RHS);
      break;
    case Instruction::Mul:
      Op = gmir::MulOp::create(Builder, Loc, ResTy, LHS, RHS);
      break;
    case Instruction::SDiv:
      Op = gmir::SDivOp::create(Builder, Loc, ResTy, LHS, RHS);
      break;
    case Instruction::And:
      Op = gmir::AndOp::create(Builder, Loc, ResTy, LHS, RHS);
      break;
    case Instruction::Or:
      Op = gmir::OrOp::create(Builder, Loc, ResTy, LHS, RHS);
      break;
    case Instruction::Xor:
      Op = gmir::XorOp::create(Builder, Loc, ResTy, LHS, RHS);
      break;
    default:
      // Unsigned div/rem, signed rem, shifts, float ops, etc: deferred.
      return false;
    }
    ValueMap[&BinOp] = Op->getResult(0);
    return true;
  }

  bool importReturn(ReturnInst &Ret) {
    llvm::Value *RetVal = Ret.getReturnValue();
    if (!RetVal) {
      mlir::func::ReturnOp::create(Builder, Builder.getUnknownLoc());
      return true;
    }
    mlir::Value MLIRRetVal;
    if (!getOperand(RetVal, MLIRRetVal))
      return false;
    mlir::func::ReturnOp::create(Builder, Builder.getUnknownLoc(), MLIRRetVal);
    return true;
  }

  /// Resolves an llvm::Value operand to its mlir::Value, materializing a
  /// gmir.constant (memoized in ValueMap) the first time a ConstantInt is
  /// seen. Returns false if the operand can't be represented (e.g. doesn't
  /// fit in 64 bits, or isn't an integer).
  bool getOperand(llvm::Value *V, mlir::Value &Out) {
    auto It = ValueMap.find(V);
    if (It != ValueMap.end()) {
      Out = It->second;
      return true;
    }
    auto *CI = dyn_cast<ConstantInt>(V);
    if (!CI || CI->getValue().getSignificantBits() > 64)
      return false;
    gmir::LLTType Ty = convertType(Context, CI->getType());
    if (!Ty)
      return false;
    // Insert at the entry block's front-growing cursor, not wherever the
    // caller's Builder happens to be pointed -- see importBody's comment on
    // ConstantInsertPt for why: this constant may be memoized and reused
    // from a block that doesn't dominate the current one.
    mlir::OpBuilder::InsertionGuard Guard(Builder);
    Builder.setInsertionPoint(EntryBlock, ConstantInsertPt);
    auto ConstOp =
        gmir::ConstantOp::create(Builder, Builder.getUnknownLoc(), Ty,
                                 Builder.getI64IntegerAttr(CI->getSExtValue()));
    ConstantInsertPt = std::next(mlir::Block::iterator(ConstOp));
    Out = ConstOp.getResult();
    ValueMap[V] = Out;
    return true;
  }

  mlir::MLIRContext &Context;
  mlir::OpBuilder &Builder;
  /// Where the next memoized constant gets inserted -- see importBody's
  /// comment and getOperand's use of these.
  mlir::Block *EntryBlock = nullptr;
  mlir::Block::iterator ConstantInsertPt;
  llvm::DenseMap<llvm::Value *, mlir::Value> ValueMap;
};

} // namespace

mlir::func::FuncOp gmir::importFunction(mlir::ModuleOp Module, Function &F) {
  mlir::MLIRContext &Context = *Module.getContext();

  // Only straight-line code is currently supported: exactly one basic
  // block.
  if (F.size() != 1)
    return {};

  SmallVector<mlir::Type> ArgTypes;
  for (Argument &Arg : F.args()) {
    gmir::LLTType Ty = convertType(Context, Arg.getType());
    if (!Ty)
      return {};
    ArgTypes.push_back(Ty);
  }

  SmallVector<mlir::Type> ResultTypes;
  if (!F.getReturnType()->isVoidTy()) {
    gmir::LLTType Ty = convertType(Context, F.getReturnType());
    if (!Ty)
      return {};
    ResultTypes.push_back(Ty);
  }

  mlir::OpBuilder ModuleBuilder = mlir::OpBuilder::atBlockEnd(Module.getBody());
  auto FuncOp = mlir::func::FuncOp::create(
      ModuleBuilder.getUnknownLoc(), F.getName(),
      mlir::FunctionType::get(&Context, ArgTypes, ResultTypes));
  ModuleBuilder.insert(FuncOp);
  mlir::Block *Entry = FuncOp.addEntryBlock();
  mlir::OpBuilder BodyBuilder = mlir::OpBuilder::atBlockEnd(Entry);

  if (!FunctionImporter(Context, BodyBuilder).importBody(F, FuncOp)) {
    FuncOp.erase();
    return {};
  }
  return FuncOp;
}
