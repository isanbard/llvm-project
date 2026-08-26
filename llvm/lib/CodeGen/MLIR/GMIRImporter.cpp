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
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include <limits>
#include <optional>

using namespace llvm;

/// Converts a scalar integer or pointer llvm::Type to !gmir.llt. Returns a
/// null type for anything else (vectors, floats, aggregates) -- aggregates
/// are handled by flattening to multiple leaf !gmir.llt values (see
/// computeGMIRLeafTypes), not by this function; vectors/floats remain out
/// of scope. Also rejects an integer wider than 64 bits: gmir.constant
/// always stores its value in a 64-bit attribute (see its doc comment), so
/// a wider !gmir.llt would make the translator's later truncate-back-down
/// step truncate *up*, which APInt::trunc() forbids and asserts on.
/// Rejecting it here, at the single shared type-conversion choke point,
/// also correctly rejects non-constant wide-integer values (e.g. two real
/// i128 arguments added together), which nothing downstream validates
/// either. Note: both `Value` and `Type` name distinct classes in ::llvm
/// and ::mlir, so this file explicitly qualifies mlir::Value/mlir::Type/
/// mlir::FunctionType throughout to avoid ambiguous lookups under the
/// blanket `using namespace llvm;`/`using namespace mlir;` below.
static gmir::LLTType convertType(mlir::MLIRContext &Context, llvm::Type *Ty) {
  if (auto *IntTy = dyn_cast<llvm::IntegerType>(Ty)) {
    if (IntTy->getBitWidth() > 64)
      return {};
    return gmir::LLTType::get(&Context, IntTy->getBitWidth(),
                              /*numElements=*/0, /*addressSpace=*/0,
                              /*isScalable=*/false);
  }
  if (auto *PtrTy = dyn_cast<llvm::PointerType>(Ty))
    return gmir::LLTType::get(&Context, /*scalarSizeInBits=*/0,
                              /*numElements=*/0, PtrTy->getAddressSpace(),
                              /*isScalable=*/false);
  return {};
}

// Narrows a uint64_t to int64_t, returning std::nullopt if V doesn't fit
// (i.e. V > INT64_MAX) rather than silently reinterpreting its bit
// pattern as negative via a bare static_cast. importAlloca's allocation
// size needs this: gmir.alloca's size attribute is stored as an int64_t,
// so a uint64_t byte count at or above 2^63 (which doesn't overflow the
// uint64_t arraySize*elementSize product computed to get there) would
// otherwise silently become a negative size.
static std::optional<int64_t> checkedU64ToI64(uint64_t V) {
  if (V > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()))
    return std::nullopt;
  return static_cast<int64_t>(V);
}

namespace {

/// Recursively flattens an (possibly aggregate) llvm::Type into its leaf
/// scalar/pointer !gmir.llt types plus each leaf's byte offset within Ty:
/// struct fields via DataLayout::getStructLayout(), array elements via
/// `i * EltSize`. Every SSA value in the importer is conceptually backed by
/// a list of leaf mlir::Values (see FunctionImporter::ValueMap), the same
/// way IRTranslator backs every llvm::Value with ArrayRef<Register> rather
/// than Register -- LLT (and so !gmir.llt) has no aggregate representation,
/// only scalar/vector/pointer. Void produces zero leaves. Returns false
/// (leaving Types/Offsets in a possibly-partial state) if any leaf isn't a
/// supported scalar/pointer type.
bool computeGMIRLeafTypes(mlir::MLIRContext &Context,
                          const llvm::DataLayout &DL, llvm::Type *Ty,
                          SmallVectorImpl<gmir::LLTType> &Types,
                          SmallVectorImpl<uint64_t> &Offsets,
                          uint64_t StartOffset = 0) {
  if (Ty->isVoidTy())
    return true;
  if (auto *StTy = dyn_cast<llvm::StructType>(Ty)) {
    const StructLayout *SL = DL.getStructLayout(StTy);
    for (unsigned I = 0, E = StTy->getNumElements(); I != E; ++I) {
      if (!computeGMIRLeafTypes(Context, DL, StTy->getElementType(I), Types,
                                Offsets, StartOffset + SL->getElementOffset(I)))
        return false;
    }
    return true;
  }
  if (auto *ArrTy = dyn_cast<llvm::ArrayType>(Ty)) {
    llvm::Type *ElemTy = ArrTy->getElementType();
    // getTypeAllocSize returns a TypeSize, not a plain integer: for a
    // scalable element type its implicit conversion to uint64_t would
    // call reportFatalInternalError and abort the whole compiler, not
    // fail gracefully like every other unsupported-type path in this
    // function. Bail explicitly instead.
    TypeSize ElemSizeTS = DL.getTypeAllocSize(ElemTy);
    if (ElemSizeTS.isScalable())
      return false;
    uint64_t ElemSize = ElemSizeTS.getFixedValue();
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
/// anything outside the currently-supported subset.
///
/// Two-pass structure: pass 1 creates every mlir::Block (with PHI-derived
/// block arguments) up front so forward branches always resolve and so
/// any PHI's value is already in ValueMap before any predecessor's branch
/// needs to resolve an incoming value through it; pass 2 translates each
/// block's real instructions and terminators.
class FunctionImporter {
public:
  FunctionImporter(mlir::MLIRContext &Context)
      : Context(Context), Builder(&Context) {}

  /// Returns false the moment an unsupported construct is seen.
  bool import(Function &F, mlir::func::FuncOp FuncOp) {
    DL = &F.getParent()->getDataLayout();
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

    // Pass 1: entry block already exists (FuncOp.addEntryBlock()); create
    // one mlir::Block per remaining BasicBlock, with one block argument
    // per PHI (in `BB.phis()` order -- the same order used later, from the
    // predecessor side, to resolve each branch's per-successor operand
    // list, so the two stay aligned without extra bookkeeping).
    BlockMap[&F.getEntryBlock()] = EntryBlock;
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
      auto *MLIRBB = new mlir::Block();
      FuncOp.getBody().push_back(MLIRBB);
      SmallVector<mlir::Location> Locs(ArgTypes.size(),
                                       Builder.getUnknownLoc());
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
      // of scope for now -- fall back rather than mistranslate. Note:
      // `select` and `switch` are reachable even from simple hand-written
      // diamond/chained-if IR, since llc's own IR-level pipeline
      // (CodeGenPrepare/SimplifyCFG-style passes) canonicalizes some
      // diamond phi patterns into `select` and some icmp-chains into
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

  /// Static-only: dynamic-sized or non-entry-block allocas fall back to the
  /// legacy selector rather than being modeled.
  bool importAlloca(AllocaInst &AI) {
    if (!AI.isStaticAlloca())
      return false;
    // isStaticAlloca() only checks the array-size operand is a constant and
    // the alloca is in the entry block -- it says nothing about whether the
    // *allocated type itself* is scalable (e.g. `alloca <vscale x 4 x i32>`
    // has no array-size operand at all, so it passes isStaticAlloca()
    // cleanly). Reject that here explicitly: below, Size.getKnownMinValue()
    // would silently return only the vscale=1 size, understating the real
    // (vscale-scaled) allocation and handing the translator a plain byte
    // count with no way to say "and multiply by vscale" -- a silent
    // under-sized stack object, not a crash.
    if (AI.getAllocatedType()->isScalableTy())
      return false;
    gmir::LLTType ResTy = convertType(Context, AI.getType());
    if (!ResTy)
      return false;
    // getAllocationSize returns nullopt not just for a non-constant array
    // size (already excluded by isStaticAlloca() above), but also when
    // arraySize*elementSize itself overflows uint64_t -- e.g. `alloca i64,
    // i64 4611686018427387905`. Reject that explicitly rather than
    // defaulting to TypeSize::getZero(), which would treat an overflowed,
    // astronomically-large allocation the same as a genuine zero-size one,
    // silently under-sizing the real stack object to 1 byte via the clamp
    // below -- the same silent-under-sizing bug class the scalable-type
    // check above guards against, just via a different trigger.
    std::optional<TypeSize> Size = AI.getAllocationSize(*DL);
    if (!Size)
      return false;
    // Matches IRTranslator::getOrCreateFrameIndex: always allocate at least
    // one byte.
    uint64_t SizeBytes = std::max<uint64_t>(Size->getKnownMinValue(), 1);
    // getI64IntegerAttr takes an int64_t: a SizeBytes above INT64_MAX (e.g.
    // `alloca i8, i64 9223372036854775808`, which doesn't overflow the
    // uint64_t product checked above) would silently reinterpret as
    // negative here -- same overflow class checkedU64ToI64 guards against
    // in importGEP, reused here rather than re-deriving the bound.
    std::optional<int64_t> SignedSizeBytes = checkedU64ToI64(SizeBytes);
    if (!SignedSizeBytes)
      return false;
    auto Op = gmir::AllocaOp::create(
        Builder, Builder.getUnknownLoc(), ResTy,
        Builder.getI64IntegerAttr(*SignedSizeBytes),
        Builder.getI64IntegerAttr(AI.getAlign().value()));
    ValueMap[&AI] = Op.getResult();
    return true;
  }

  /// Scalar/pointer only for now -- an aggregate-typed load bails here (leaf
  /// count != 1); aggregate flattening is added on top of this same helper
  /// later.
  bool importLoad(LoadInst &LI) {
    SmallVector<gmir::LLTType, 1> LeafTypes;
    SmallVector<uint64_t, 1> Offsets;
    if (!computeGMIRLeafTypes(Context, *DL, LI.getType(), LeafTypes, Offsets) ||
        LeafTypes.size() != 1)
      return false;
    mlir::Value Ptr;
    if (!getOperand(LI.getPointerOperand(), Ptr))
      return false;
    // isInvariant/isNonTemporal mirror TargetLoweringBase::
    // getLoadMemOperandFlags's MOInvariant/MONonTemporal derivation exactly
    // -- a plain presence check, no analysis dependency, unlike
    // MODereferenceable (deliberately not modeled here: the real derivation
    // needs isDereferenceableAndAlignedPointer, which needs an
    // AssumptionCache/TargetLibraryInfo the importer doesn't have threaded
    // in, and the cheaper !dereferenceable-metadata-only fallback is narrow
    // enough not to be worth the op-shape churn until something needs it).
    mlir::UnitAttr IsInvariant = LI.hasMetadata(LLVMContext::MD_invariant_load)
                                     ? Builder.getUnitAttr()
                                     : mlir::UnitAttr();
    mlir::UnitAttr IsNonTemporal = LI.hasMetadata(LLVMContext::MD_nontemporal)
                                       ? Builder.getUnitAttr()
                                       : mlir::UnitAttr();
    auto Op = gmir::LoadOp::create(
        Builder, Builder.getUnknownLoc(), LeafTypes[0], Ptr,
        Builder.getI64IntegerAttr(LI.getAlign().value()),
        Builder.getI64IntegerAttr(static_cast<int64_t>(LI.getOrdering())),
        Builder.getI64IntegerAttr(static_cast<int64_t>(LI.getSyncScopeID())),
        LI.isVolatile() ? Builder.getUnitAttr() : mlir::UnitAttr(), IsInvariant,
        IsNonTemporal);
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
    // Mirrors TargetLoweringBase::getStoreMemOperandFlags's MONonTemporal
    // derivation exactly (stores have no invariant or dereferenceable
    // concept, unlike loads).
    mlir::UnitAttr IsNonTemporal = SI.hasMetadata(LLVMContext::MD_nontemporal)
                                       ? Builder.getUnitAttr()
                                       : mlir::UnitAttr();
    gmir::StoreOp::create(
        Builder, Builder.getUnknownLoc(), Val, Ptr,
        Builder.getI64IntegerAttr(SI.getAlign().value()),
        Builder.getI64IntegerAttr(static_cast<int64_t>(SI.getOrdering())),
        Builder.getI64IntegerAttr(static_cast<int64_t>(SI.getSyncScopeID())),
        SI.isVolatile() ? Builder.getUnitAttr() : mlir::UnitAttr(),
        IsNonTemporal);
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

  bool importUncondBr(UncondBrInst &Br) {
    SmallVector<mlir::Value> DestOperands;
    if (!getSuccessorOperands(*Br.getParent(), *Br.getSuccessor(0),
                              DestOperands))
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
    if (!getSuccessorOperands(*Br.getParent(), *Br.getSuccessor(0),
                              TrueOperands) ||
        !getSuccessorOperands(*Br.getParent(), *Br.getSuccessor(1),
                              FalseOperands))
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
    // Insert at the entry block's front-growing cursor, not wherever the
    // caller's Builder happens to be pointed -- see import()'s comment on
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
  mlir::OpBuilder Builder;
  const llvm::DataLayout *DL = nullptr;
  /// Where the next memoized constant gets inserted -- see import()'s
  /// comment and getOperand's use of these.
  mlir::Block *EntryBlock = nullptr;
  mlir::Block::iterator ConstantInsertPt;
  llvm::DenseMap<BasicBlock *, mlir::Block *> BlockMap;
  llvm::DenseMap<llvm::Value *, mlir::Value> ValueMap;
};

} // namespace

mlir::func::FuncOp gmir::importFunction(mlir::ModuleOp Module, Function &F) {
  mlir::MLIRContext &Context = *Module.getContext();

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
  FuncOp.addEntryBlock();

  if (!FunctionImporter(Context).import(F, FuncOp)) {
    FuncOp.erase();
    return {};
  }
  return FuncOp;
}
