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
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "mlir/IR/Builders.h"

using namespace llvm;
using namespace mlir;

namespace {

/// Converts a scalar integer or pointer llvm::Type to !gmir.llt. Returns a
/// null type for anything else (vectors, floats, aggregates) -- aggregates
/// are handled by flattening to multiple leaf !gmir.llt values (see
/// computeGMIRLeafTypes), not by this function; vectors/floats remain out
/// of scope. Note: both `Value` and `Type` name distinct classes in ::llvm
/// and ::mlir, so this file explicitly qualifies mlir::Value/mlir::Type/
/// mlir::FunctionType throughout to avoid ambiguous lookups under the
/// blanket `using namespace llvm;`/`using namespace mlir;` below.
gmir::LLTType convertType(MLIRContext &Context, llvm::Type *Ty) {
  if (auto *IntTy = dyn_cast<llvm::IntegerType>(Ty))
    return gmir::LLTType::get(&Context, IntTy->getBitWidth(),
                               /*numElements=*/0, /*addressSpace=*/0,
                               /*isScalable=*/false);
  if (auto *PtrTy = dyn_cast<llvm::PointerType>(Ty))
    return gmir::LLTType::get(&Context, /*scalarSizeInBits=*/0,
                               /*numElements=*/0, PtrTy->getAddressSpace(),
                               /*isScalable=*/false);
  return {};
}

/// Recursively flattens an (possibly aggregate) llvm::Type into its leaf
/// scalar/pointer !gmir.llt types plus each leaf's byte offset within Ty,
/// mirroring llvm::ComputeValueTypes (Analysis.cpp) -- struct fields via
/// DataLayout::getStructLayout(), array elements via `i * EltSize`. Every
/// SSA value in the importer is conceptually backed by a list of leaf
/// mlir::Values (see FunctionImporter::ValueMap), the same way IRTranslator
/// backs every llvm::Value with ArrayRef<Register> rather than Register --
/// LLT (and so !gmir.llt) has no aggregate representation, only
/// scalar/vector/pointer. Void produces zero leaves. Returns false (leaving
/// Types/Offsets in a possibly-partial state) if any leaf isn't a supported
/// scalar/pointer type.
bool computeGMIRLeafTypes(MLIRContext &Context, const llvm::DataLayout &DL,
                           llvm::Type *Ty, SmallVectorImpl<gmir::LLTType> &Types,
                           SmallVectorImpl<uint64_t> &Offsets,
                           uint64_t StartOffset = 0) {
  if (Ty->isVoidTy())
    return true;
  if (auto *StTy = dyn_cast<llvm::StructType>(Ty)) {
    const StructLayout *SL = DL.getStructLayout(StTy);
    for (unsigned I = 0, E = StTy->getNumElements(); I != E; ++I) {
      if (!computeGMIRLeafTypes(Context, DL, StTy->getElementType(I), Types,
                                 Offsets,
                                 StartOffset + SL->getElementOffset(I)))
        return false;
    }
    return true;
  }
  if (auto *ArrTy = dyn_cast<llvm::ArrayType>(Ty)) {
    llvm::Type *ElemTy = ArrTy->getElementType();
    uint64_t ElemSize = DL.getTypeAllocSize(ElemTy);
    for (uint64_t I = 0, E = ArrTy->getNumElements(); I != E; ++I) {
      if (!computeGMIRLeafTypes(Context, DL, ElemTy, Types, Offsets,
                                 StartOffset + I * ElemSize))
        return false;
    }
    return true;
  }
  gmir::LLTType LeafTy = convertType(Context, Ty);
  if (!LeafTy)
    return false;
  Types.push_back(LeafTy);
  Offsets.push_back(StartOffset);
  return true;
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
    DL = &F.getParent()->getDataLayout();
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
      if (auto *AI = dyn_cast<AllocaInst>(&I)) {
        if (!importAlloca(*AI))
          return false;
        continue;
      }
      if (auto *LI = dyn_cast<LoadInst>(&I)) {
        if (!importLoad(*LI))
          return false;
        continue;
      }
      if (auto *SI = dyn_cast<StoreInst>(&I)) {
        if (!importStore(*SI))
          return false;
        continue;
      }
      // Anything else (calls, GEPs, switches, casts, selects, ...) is out
      // of scope for this milestone slice -- fall back rather than
      // mistranslate. Note: `select` and `switch` are reachable even from
      // simple hand-written diamond/chained-if IR, since llc's own IR-level
      // pipeline (CodeGenPrepare/SimplifyCFG-style passes) canonicalizes
      // some diamond phi patterns into `select` and some icmp-chains into
      // `switch` before codegen ever sees the function.
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

  /// Static-only (§1.5 of the design doc): dynamic-sized or non-entry-block
  /// allocas fall back to the legacy selector rather than being modeled.
  bool importAlloca(AllocaInst &AI) {
    if (!AI.isStaticAlloca())
      return false;
    gmir::LLTType ResTy = convertType(Context, AI.getType());
    if (!ResTy)
      return false;
    // Matches IRTranslator::getOrCreateFrameIndex: always allocate at least
    // one byte.
    TypeSize Size = AI.getAllocationSize(*DL).value_or(TypeSize::getZero());
    uint64_t SizeBytes = std::max<uint64_t>(Size.getKnownMinValue(), 1);
    auto Op = gmir::AllocaOp::create(
        Builder, Builder.getUnknownLoc(), ResTy,
        Builder.getI64IntegerAttr(SizeBytes),
        Builder.getI64IntegerAttr(AI.getAlign().value()));
    ValueMap[&AI] = Op.getResult();
    return true;
  }

  /// Scalar/pointer only for now -- an aggregate-typed load bails here
  /// (leaf count != 1); aggregate flattening is added on top of this same
  /// helper in a later milestone slice.
  bool importLoad(LoadInst &LI) {
    SmallVector<gmir::LLTType, 1> LeafTypes;
    SmallVector<uint64_t, 1> Offsets;
    if (!computeGMIRLeafTypes(Context, *DL, LI.getType(), LeafTypes, Offsets) ||
        LeafTypes.size() != 1)
      return false;
    mlir::Value Ptr;
    if (!getOperand(LI.getPointerOperand(), Ptr))
      return false;
    auto Op = gmir::LoadOp::create(
        Builder, Builder.getUnknownLoc(), LeafTypes[0], Ptr,
        Builder.getI64IntegerAttr(LI.getAlign().value()),
        Builder.getI64IntegerAttr(static_cast<int64_t>(LI.getOrdering())),
        Builder.getI64IntegerAttr(static_cast<int64_t>(LI.getSyncScopeID())),
        LI.isVolatile() ? Builder.getUnitAttr() : mlir::UnitAttr());
    ValueMap[&LI] = Op.getResult();
    return true;
  }

  bool importStore(StoreInst &SI) {
    SmallVector<gmir::LLTType, 1> LeafTypes;
    SmallVector<uint64_t, 1> Offsets;
    if (!computeGMIRLeafTypes(Context, *DL, SI.getValueOperand()->getType(),
                               LeafTypes, Offsets) ||
        LeafTypes.size() != 1)
      return false;
    mlir::Value Val, Ptr;
    if (!getOperand(SI.getValueOperand(), Val) ||
        !getOperand(SI.getPointerOperand(), Ptr))
      return false;
    gmir::StoreOp::create(
        Builder, Builder.getUnknownLoc(), Val, Ptr,
        Builder.getI64IntegerAttr(SI.getAlign().value()),
        Builder.getI64IntegerAttr(static_cast<int64_t>(SI.getOrdering())),
        Builder.getI64IntegerAttr(static_cast<int64_t>(SI.getSyncScopeID())),
        SI.isVolatile() ? Builder.getUnitAttr() : mlir::UnitAttr());
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
  const llvm::DataLayout *DL = nullptr;
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
