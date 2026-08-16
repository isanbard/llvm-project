//===- GMIRImporter.cpp - LLVM IR -> gmir importer -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "GMIRImporter.h"
#include "IR/GMIRDialect.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "mlir/IR/Builders.h"

using namespace llvm;
using namespace mlir;

namespace {

/// Converts a scalar integer llvm::Type to !gmir.llt. Returns a null type
/// for anything else (vectors, pointers, floats, aggregates) -- all out of
/// scope for M1. Note: both `Value` and `Type` name distinct classes in
/// ::llvm and ::mlir, so this file explicitly qualifies mlir::Value/
/// mlir::Type/mlir::FunctionType throughout to avoid ambiguous lookups
/// under the blanket `using namespace llvm;`/`using namespace mlir;` below.
gmir::LLTType convertType(MLIRContext &Context, llvm::Type *Ty) {
  auto *IntTy = dyn_cast<llvm::IntegerType>(Ty);
  if (!IntTy)
    return {};
  return gmir::LLTType::get(&Context, IntTy->getBitWidth(),
                             /*numElements=*/0, /*addressSpace=*/0,
                             /*isScalable=*/false);
}

/// Walks one llvm::Function's single basic block, building `gmir` ops.
/// Bails out (leaving ValueMap/Builder state to be discarded by the
/// caller) the moment it sees anything outside the M1 subset.
class FunctionImporter {
public:
  FunctionImporter(MLIRContext &Context, OpBuilder &Builder)
      : Context(Context), Builder(Builder) {}

  /// Returns false the moment an unsupported construct is seen.
  bool importBody(Function &F, func::FuncOp FuncOp) {
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
      // of scope for M1 -- fall back rather than mistranslate.
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

    Location Loc = Builder.getUnknownLoc();
    Operation *Op;
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
      func::ReturnOp::create(Builder, Builder.getUnknownLoc());
      return true;
    }
    mlir::Value MLIRRetVal;
    if (!getOperand(RetVal, MLIRRetVal))
      return false;
    func::ReturnOp::create(Builder, Builder.getUnknownLoc(), MLIRRetVal);
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
    auto ConstOp = gmir::ConstantOp::create(
        Builder, Builder.getUnknownLoc(), Ty,
        Builder.getI64IntegerAttr(CI->getSExtValue()));
    Out = ConstOp.getResult();
    ValueMap[V] = Out;
    return true;
  }

  MLIRContext &Context;
  OpBuilder &Builder;
  llvm::DenseMap<llvm::Value *, mlir::Value> ValueMap;
};

} // namespace

func::FuncOp gmir::importFunction(ModuleOp Module, Function &F) {
  MLIRContext &Context = *Module.getContext();

  // M1 only handles straight-line code: exactly one basic block.
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

  OpBuilder ModuleBuilder = OpBuilder::atBlockEnd(Module.getBody());
  auto FuncOp = func::FuncOp::create(
      ModuleBuilder.getUnknownLoc(), F.getName(),
      mlir::FunctionType::get(&Context, ArgTypes, ResultTypes));
  ModuleBuilder.insert(FuncOp);
  Block *Entry = FuncOp.addEntryBlock();
  OpBuilder BodyBuilder = OpBuilder::atBlockEnd(Entry);

  if (!FunctionImporter(Context, BodyBuilder).importBody(F, FuncOp)) {
    FuncOp.erase();
    return {};
  }
  return FuncOp;
}
