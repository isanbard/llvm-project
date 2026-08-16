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
/// scope for M1/M2. Note: both `Value` and `Type` name distinct classes in
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

/// Walks an llvm::Function's basic blocks, building `gmir` ops. Bails out
/// (leaving all state to be discarded by the caller) the moment it sees
/// anything outside the M1/M2 subset.
///
/// Two-pass structure mirrors GlobalISel::IRTranslator (see
/// ~/llvm/mlir_instruction_selection_plan.md, decision #5's refinement):
/// pass 1 creates every mlir::Block (with PHI-derived block arguments) up
/// front so forward branches always resolve and so any PHI's value is
/// already in ValueMap before any predecessor's branch needs to resolve an
/// incoming value through it; pass 2 translates each block's real
/// instructions and terminators.
class FunctionImporter {
public:
  FunctionImporter(MLIRContext &Context) : Context(Context), Builder(&Context) {}

  /// Returns false the moment an unsupported construct is seen.
  bool import(Function &F, func::FuncOp FuncOp) {
    unsigned ArgIdx = 0;
    for (Argument &Arg : F.args())
      ValueMap[&Arg] = FuncOp.getArgument(ArgIdx++);

    // Pass 1: entry block already exists (FuncOp.addEntryBlock()); create
    // one mlir::Block per remaining BasicBlock, with one block argument
    // per PHI (in `BB.phis()` order -- the same order used later, from the
    // predecessor side, to resolve each branch's per-successor operand
    // list, so the two stay aligned without extra bookkeeping).
    BlockMap[&F.getEntryBlock()] = &FuncOp.getBody().front();
    for (BasicBlock &BB : F) {
      if (&BB == &F.getEntryBlock())
        continue;
      SmallVector<mlir::Type> ArgTypes;
      for (PHINode &PN : BB.phis()) {
        gmir::LLTType Ty = convertType(Context, PN.getType());
        if (!Ty)
          return false;
        ArgTypes.push_back(Ty);
      }
      auto *MLIRBB = new Block();
      FuncOp.getBody().push_back(MLIRBB);
      SmallVector<mlir::Location> Locs(ArgTypes.size(), Builder.getUnknownLoc());
      MLIRBB->addArguments(ArgTypes, Locs);
      BlockMap[&BB] = MLIRBB;

      unsigned PhiIdx = 0;
      for (PHINode &PN : BB.phis())
        ValueMap[&PN] = MLIRBB->getArgument(PhiIdx++);
    }

    // Pass 2: translate each block's non-PHI instructions.
    for (BasicBlock &BB : F) {
      Builder.setInsertionPointToEnd(BlockMap[&BB]);
      if (!importBlockBody(BB))
        return false;
    }
    return true;
  }

private:
  bool importBlockBody(BasicBlock &BB) {
    for (Instruction &I : BB) {
      if (isa<PHINode>(I))
        continue; // handled in pass 1
      if (auto *Ret = dyn_cast<ReturnInst>(&I))
        return importReturn(*Ret);
      // This checkout splits LLVM upstream's single BranchInst into
      // UncondBrInst/CondBrInst (two distinct opcodes/classes) rather than
      // one class with isConditional() -- see Instruction.def's
      // HANDLE_TERM_INST(UncondBr/CondBr) entries.
      if (auto *Br = dyn_cast<UncondBrInst>(&I))
        return importUncondBr(*Br);
      if (auto *Br = dyn_cast<CondBrInst>(&I))
        return importCondBr(*Br);
      if (auto *ICmp = dyn_cast<ICmpInst>(&I)) {
        if (!importICmp(*ICmp))
          return false;
        continue;
      }
      if (auto *BinOp = dyn_cast<BinaryOperator>(&I)) {
        if (!importBinaryOp(*BinOp))
          return false;
        continue;
      }
      // Anything else (calls, loads/stores, switches, casts, ...) is out
      // of scope for M1/M2 -- fall back rather than mistranslate.
      return false;
    }
    // Fell off the end without a terminator: malformed IR, shouldn't
    // happen, but bail gracefully rather than assert.
    return false;
  }

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

  bool importICmp(ICmpInst &ICmp) {
    gmir::LLTType ResTy = convertType(Context, ICmp.getType());
    if (!ResTy)
      return false;
    mlir::Value LHS, RHS;
    if (!getOperand(ICmp.getOperand(0), LHS) ||
        !getOperand(ICmp.getOperand(1), RHS))
      return false;
    auto Op = gmir::ICmpOp::create(
        Builder, Builder.getUnknownLoc(), ResTy,
        Builder.getI64IntegerAttr(static_cast<int64_t>(ICmp.getPredicate())),
        LHS, RHS);
    ValueMap[&ICmp] = Op.getResult();
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

  bool importUncondBr(UncondBrInst &Br) {
    SmallVector<mlir::Value> DestOperands;
    if (!getSuccessorOperands(*Br.getParent(), *Br.getSuccessor(0), DestOperands))
      return false;
    gmir::BrOp::create(Builder, Builder.getUnknownLoc(), DestOperands,
                        BlockMap[Br.getSuccessor(0)]);
    return true;
  }

  bool importCondBr(CondBrInst &Br) {
    mlir::Value Cond;
    if (!getOperand(Br.getCondition(), Cond))
      return false;
    SmallVector<mlir::Value> TrueOperands, FalseOperands;
    if (!getSuccessorOperands(*Br.getParent(), *Br.getSuccessor(0), TrueOperands) ||
        !getSuccessorOperands(*Br.getParent(), *Br.getSuccessor(1), FalseOperands))
      return false;
    gmir::CondBrOp::create(Builder, Builder.getUnknownLoc(), Cond, TrueOperands,
                            FalseOperands, BlockMap[Br.getSuccessor(0)],
                            BlockMap[Br.getSuccessor(1)]);
    return true;
  }

  /// For a branch from FromBB to ToBB, resolves the operand list that
  /// feeds ToBB's block arguments -- i.e. each PHI in ToBB's incoming
  /// value for this specific predecessor edge, in the same `BB.phis()`
  /// order the block arguments were created in during pass 1.
  bool getSuccessorOperands(BasicBlock &FromBB, BasicBlock &ToBB,
                             SmallVectorImpl<mlir::Value> &Operands) {
    for (PHINode &PN : ToBB.phis()) {
      mlir::Value V;
      if (!getOperand(PN.getIncomingValueForBlock(&FromBB), V))
        return false;
      Operands.push_back(V);
    }
    return true;
  }

  /// Resolves an llvm::Value operand to its mlir::Value: an already-mapped
  /// value (a formal arg, a PHI's block argument, or a prior instruction's
  /// result), or materializes a gmir.constant (memoized in ValueMap) the
  /// first time a ConstantInt is seen. Returns false if the operand can't
  /// be represented (e.g. doesn't fit in 64 bits, isn't an integer, or is
  /// some other unmapped/unsupported value).
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
  OpBuilder Builder;
  llvm::DenseMap<BasicBlock *, Block *> BlockMap;
  llvm::DenseMap<llvm::Value *, mlir::Value> ValueMap;
};

} // namespace

func::FuncOp gmir::importFunction(ModuleOp Module, Function &F) {
  MLIRContext &Context = *Module.getContext();

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
  FuncOp.addEntryBlock();

  if (!FunctionImporter(Context).import(F, FuncOp)) {
    FuncOp.erase();
    return {};
  }
  return FuncOp;
}
