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
#include "mlir/IR/Location.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/CodeGen/Analysis.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GetElementPtrTypeIterator.h"
#include "llvm/IR/GlobalAlias.h"
#include "llvm/IR/GlobalIFunc.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Operator.h"
#include "llvm/Support/Alignment.h"

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
  if (auto *IntTy = dyn_cast<llvm::IntegerType>(Ty)) {
    // gmir.constant stores every value sign-extended into a 64-bit I64Attr
    // (see its doc comment); a wider destination !gmir.llt would make
    // MLIRToGMIRTranslator.cpp's later `.trunc(Ty.getScalarSizeInBits())`
    // call truncate *up*, which APInt::trunc() forbids (width <=
    // BitWidth) and asserts on. Reject integer types over 64 bits here,
    // at the single shared type-conversion choke point, rather than only
    // in the ConstantInt materialization path -- this also correctly
    // rejects non-constant wide-integer values (e.g. two real i128
    // arguments added together), which this milestone's pipeline has
    // never validated either.
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

/// Flattens an (possibly aggregate) llvm::Type into its leaf scalar/pointer
/// !gmir.llt types plus each leaf's byte offset within Ty. Every SSA value
/// in the importer is conceptually backed by a list of leaf mlir::Values
/// (see FunctionImporter::ValueMap), the same way IRTranslator backs every
/// llvm::Value with ArrayRef<Register> rather than Register -- LLT (and so
/// !gmir.llt) has no aggregate representation, only scalar/vector/pointer.
/// Void produces zero leaves. Returns false (leaving Types/Offsets in a
/// possibly-partial state) if any leaf isn't a supported scalar/pointer
/// type, or if any leaf's offset is scalable (!gmir.llt/gmir ops have no
/// way to represent a vscale-dependent byte offset).
///
/// The actual struct/array recursion is llvm::ComputeValueTypes
/// (llvm/CodeGen/Analysis.h) -- the same representation-agnostic walker
/// IRTranslator's own computeValueLLTs is a thin wrapper over (see
/// Analysis.cpp: computeValueLLTs just maps ComputeValueTypes's leaf
/// llvm::Types through getLLTForType; this does the same through
/// convertType instead). Reusing it directly, rather than hand-porting the
/// struct/array walk again, means any future correctness fix to it (e.g.
/// around scalable-type handling, which it already supports more
/// completely than an earlier version of this function did -- see the
/// design doc) benefits every caller without a separate re-port.
bool computeGMIRLeafTypes(MLIRContext &Context, const llvm::DataLayout &DL,
                          llvm::Type *Ty, SmallVectorImpl<gmir::LLTType> &Types,
                          SmallVectorImpl<uint64_t> &Offsets) {
  SmallVector<llvm::Type *, 1> LeafIRTypes;
  SmallVector<TypeSize, 1> LeafOffsets;
  llvm::ComputeValueTypes(DL, Ty, LeafIRTypes, &LeafOffsets);
  for (auto [LeafIRTy, LeafOffset] : zip(LeafIRTypes, LeafOffsets)) {
    if (LeafOffset.isScalable())
      return false;
    gmir::LLTType LeafTy = convertType(Context, LeafIRTy);
    if (!LeafTy)
      return false;
    Types.push_back(LeafTy);
    Offsets.push_back(LeafOffset.getFixedValue());
  }
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
    // Memoized constants (see getOperands) are materialized at a fixed,
    // growing-forward cursor at the very front of the entry block, exactly
    // like IRTranslator's own dedicated EntryBuilder -- this guarantees
    // any later reuse of a memoized constant's mlir::Value dominates its
    // use, since the entry block dominates every other block and ops
    // preceding all others in the entry block precede everything within
    // it too. Without this, a ConstantInt shared by two non-dominating
    // blocks (LLVM interns/uniques ConstantInts, so this is routine, not
    // an edge case) would get memoized wherever it was first encountered
    // and reused from a sibling block that doesn't dominate that point.
    EntryBlock = &FuncOp.getBody().front();
    ConstantInsertPt = EntryBlock->begin();
    unsigned ArgIdx = 0;
    for (Argument &Arg : F.args())
      ValueMap[&Arg] = {FuncOp.getArgument(ArgIdx++)};

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
        ValueMap[&PN] = {MLIRBB->getArgument(PhiIdx++)};
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
      if (auto *GEP = dyn_cast<GetElementPtrInst>(&I)) {
        if (!importGEP(*GEP))
          return false;
        continue;
      }
      if (auto *CI = dyn_cast<CallInst>(&I)) {
        if (!importCall(*CI))
          return false;
        continue;
      }
      // Anything else (switches, casts, selects, invokes, ...) is out of
      // scope for this milestone slice -- fall back rather than
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
    if (!getScalarOperand(BinOp.getOperand(0), LHS) ||
        !getScalarOperand(BinOp.getOperand(1), RHS))
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
    ValueMap[&BinOp] = {Op->getResult(0)};
    return true;
  }

  bool importICmp(ICmpInst &ICmp) {
    gmir::LLTType ResTy = convertType(Context, ICmp.getType());
    if (!ResTy)
      return false;
    mlir::Value LHS, RHS;
    if (!getScalarOperand(ICmp.getOperand(0), LHS) ||
        !getScalarOperand(ICmp.getOperand(1), RHS))
      return false;
    auto Op = gmir::ICmpOp::create(
        Builder, Builder.getUnknownLoc(), ResTy,
        Builder.getI64IntegerAttr(static_cast<int64_t>(ICmp.getPredicate())),
        LHS, RHS);
    ValueMap[&ICmp] = {Op.getResult()};
    return true;
  }

  /// Static-only (§1.5 of the design doc): dynamic-sized or non-entry-block
  /// allocas fall back to the legacy selector rather than being modeled.
  bool importAlloca(AllocaInst &AI) {
    if (!AI.isStaticAlloca())
      return false;
    // isStaticAlloca() only checks the array-size operand is a constant
    // and the alloca is in the entry block -- it says nothing about
    // whether the *allocated type itself* is scalable (e.g. `alloca
    // <vscale x 4 x i32>` has no array-size operand at all, so it passes
    // isStaticAlloca() cleanly). Reject that here explicitly: below,
    // Size.getKnownMinValue() would silently return only the vscale=1
    // size, understating the real (vscale-scaled) allocation and handing
    // the translator a plain byte count with no way to say "and multiply
    // by vscale" -- a silent under-sized stack object, not a crash.
    if (AI.getAllocatedType()->isScalableTy())
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
    ValueMap[&AI] = {Op.getResult()};
    return true;
  }

  /// Returns BasePtr unchanged when Offset==0 (skip-if-zero, mirroring
  /// MachineIRBuilder::materializePtrAdd's short-circuit -- the common
  /// case, since most loads/stores are scalar/pointer with a single
  /// zero-offset leaf), else emits gmir.constant(Offset) +
  /// gmir.ptr_add(BasePtr, that constant) with NoUWrap+InBounds set
  /// unconditionally (mirrors MachineIRBuilder::materializeObjectPtrOffset's
  /// fixed flags -- this is sub-object offset math within an
  /// already-valid object, not a user GEP, so it always gets the strong
  /// nuw+inbounds guarantee regardless of anything else).
  mlir::Value materializeLeafPtr(mlir::Value BasePtr, gmir::LLTType PtrTy,
                                 gmir::LLTType OffsetTy, uint64_t Offset) {
    if (Offset == 0)
      return BasePtr;
    auto ConstOp = gmir::ConstantOp::create(
        Builder, Builder.getUnknownLoc(), OffsetTy,
        Builder.getI64IntegerAttr(static_cast<int64_t>(Offset)));
    auto AddOp = gmir::PtrAddOp::create(
        Builder, Builder.getUnknownLoc(), PtrTy, BasePtr, ConstOp.getResult(),
        /*NoUWrap=*/Builder.getUnitAttr(), /*NoUSWrap=*/mlir::UnitAttr(),
        /*InBounds=*/Builder.getUnitAttr());
    return AddOp.getResult();
  }

  /// Flattens an aggregate-typed load into one gmir.load per leaf (each at
  /// its own already-resolved offset pointer, per-leaf alignment via
  /// commonAlignment -- sub-object alignment is generally weaker than the
  /// whole aggregate's, mirroring IRTranslator::translateLoad's
  /// multi-register loop), collecting the results into ValueMap[&LI]. A
  /// plain scalar/pointer load is just the trivial single-leaf,
  /// zero-offset case of the same code path.
  bool importLoad(LoadInst &LI) {
    SmallVector<gmir::LLTType, 1> LeafTypes;
    SmallVector<uint64_t, 1> Offsets;
    if (!computeGMIRLeafTypes(Context, *DL, LI.getType(), LeafTypes, Offsets))
      return false;
    mlir::Value BasePtr;
    if (!getScalarOperand(LI.getPointerOperand(), BasePtr))
      return false;
    llvm::Type *PtrIRTy = LI.getPointerOperand()->getType();
    gmir::LLTType PtrTy = convertType(Context, PtrIRTy);
    gmir::LLTType OffsetTy = convertType(Context, DL->getIndexType(PtrIRTy));
    // Mirrors TargetLoweringBase::getLoadMemOperandFlags's MOInvariant/
    // MONonTemporal derivation exactly -- see gmir.load's doc comment for
    // why MODereferenceable isn't modeled alongside these.
    mlir::UnitAttr IsInvariant = LI.hasMetadata(LLVMContext::MD_invariant_load)
                                     ? Builder.getUnitAttr()
                                     : mlir::UnitAttr();
    mlir::UnitAttr IsNonTemporal = LI.hasMetadata(LLVMContext::MD_nontemporal)
                                       ? Builder.getUnitAttr()
                                       : mlir::UnitAttr();
    SmallVector<mlir::Value, 1> Results;
    for (auto [LeafTy, Offset] : zip(LeafTypes, Offsets)) {
      mlir::Value LeafPtr =
          materializeLeafPtr(BasePtr, PtrTy, OffsetTy, Offset);
      Align LeafAlign = commonAlignment(LI.getAlign(), Offset);
      auto Op = gmir::LoadOp::create(
          Builder, Builder.getUnknownLoc(), LeafTy, LeafPtr,
          Builder.getI64IntegerAttr(LeafAlign.value()),
          Builder.getI64IntegerAttr(static_cast<int64_t>(LI.getOrdering())),
          Builder.getI64IntegerAttr(static_cast<int64_t>(LI.getSyncScopeID())),
          LI.isVolatile() ? Builder.getUnitAttr() : mlir::UnitAttr(),
          IsInvariant, IsNonTemporal);
      Results.push_back(Op.getResult());
    }
    ValueMap[&LI] = std::move(Results);
    return true;
  }

  /// Symmetric to importLoad: flattens an aggregate-typed store's value
  /// operand into one gmir.store per leaf.
  bool importStore(StoreInst &SI) {
    SmallVector<gmir::LLTType, 1> LeafTypes;
    SmallVector<uint64_t, 1> Offsets;
    if (!computeGMIRLeafTypes(Context, *DL, SI.getValueOperand()->getType(),
                              LeafTypes, Offsets))
      return false;
    SmallVector<mlir::Value, 1> ValueLeaves;
    mlir::Value BasePtr;
    if (!getOperands(SI.getValueOperand(), ValueLeaves) ||
        ValueLeaves.size() != LeafTypes.size() ||
        !getScalarOperand(SI.getPointerOperand(), BasePtr))
      return false;
    llvm::Type *PtrIRTy = SI.getPointerOperand()->getType();
    gmir::LLTType PtrTy = convertType(Context, PtrIRTy);
    gmir::LLTType OffsetTy = convertType(Context, DL->getIndexType(PtrIRTy));
    // Mirrors TargetLoweringBase::getStoreMemOperandFlags's MONonTemporal
    // derivation exactly.
    mlir::UnitAttr IsNonTemporal = SI.hasMetadata(LLVMContext::MD_nontemporal)
                                       ? Builder.getUnitAttr()
                                       : mlir::UnitAttr();
    for (auto [ValueLeaf, Offset] : zip(ValueLeaves, Offsets)) {
      mlir::Value LeafPtr =
          materializeLeafPtr(BasePtr, PtrTy, OffsetTy, Offset);
      Align LeafAlign = commonAlignment(SI.getAlign(), Offset);
      gmir::StoreOp::create(
          Builder, Builder.getUnknownLoc(), ValueLeaf, LeafPtr,
          Builder.getI64IntegerAttr(LeafAlign.value()),
          Builder.getI64IntegerAttr(static_cast<int64_t>(SI.getOrdering())),
          Builder.getI64IntegerAttr(static_cast<int64_t>(SI.getSyncScopeID())),
          SI.isVolatile() ? Builder.getUnitAttr() : mlir::UnitAttr(),
          IsNonTemporal);
    }
    return true;
  }

  /// Imports a scalar/pointer-only GetElementPtrInst, porting
  /// IRTranslator::translateGetElementPtr's constant-index-coalescing
  /// algorithm exactly (for -global-isel byte parity on multi-index GEPs,
  /// rather than a naive one-ptr_add-per-index scheme): runs of constant
  /// struct-field/array indices accumulate into a single running offset,
  /// flushed into one gmir.constant+gmir.ptr_add only when a variable
  /// index is hit and (if still nonzero) once more at the end; each
  /// variable index gets its own gmir.mul (skipped when the element size
  /// is 1) + gmir.ptr_add. Bails (falls back to the legacy selector) for:
  /// vector GEPs (result or pointer-operand type isa<VectorType> --
  /// catches scalable vectors too, since ScalableVectorType inherits
  /// VectorType; stricter than this checkout's own
  /// translateGetElementPtr, which has a latent unguarded
  /// cast<FixedVectorType> for the scalable case); and any variable index
  /// whose integer bit width doesn't match the pointer-index type's
  /// width, since gmir has no sext/trunc op yet to fix that up (see
  /// gmir.ptr_add's doc comment).
  ///
  /// Unlike computeGMIRLeafTypes (which now delegates its struct/array
  /// walk to the shared llvm::ComputeValueTypes), this function's
  /// coalescing loop is a genuine hand-port rather than a call into
  /// IRTranslator::translateGetElementPtr, and deliberately stays that
  /// way: the type-system primitives it walks (gep_type_iterator,
  /// DataLayout::getStructLayout) are already shared as-is, but the
  /// coalescing *algorithm* itself is entangled with MachineIRBuilder's
  /// G_PTR_ADD/G_MUL/G_CONSTANT builder calls in the real function, with
  /// no existing builder-agnostic abstraction to call into instead.
  /// Extracting one would mean refactoring a core, heavily-used
  /// GlobalISel function used by every in-tree target to be templated or
  /// callback-parameterized over "how do I emit an add/mul/constant" --
  /// a meaningfully larger and riskier change than a mechanical
  /// extraction (contrast createMIRBuilder in GlobalISel/Utils.cpp, a
  /// true behavior-preserving extraction with no design decisions),
  /// for a ~50-line algorithm that changes rarely. Not attempted here;
  /// worth reconsidering if it ever proves to actually drift.
  bool importGEP(GetElementPtrInst &GEP) {
    // llvm::VectorType vs. mlir::VectorType collide under this file's
    // blanket `using namespace llvm;`/`using namespace mlir;` -- same
    // class of ambiguity as Value/Type/Attribute/DenseMap noted elsewhere
    // in this codebase; explicit `llvm::` qualification required.
    if (isa<llvm::VectorType>(GEP.getType()) ||
        isa<llvm::VectorType>(GEP.getPointerOperandType()))
      return false;

    mlir::Value BaseVal;
    if (!getScalarOperand(GEP.getPointerOperand(), BaseVal))
      return false;

    auto &GEPOp = cast<GEPOperator>(GEP);
    if (GEPOp.hasAllZeroIndices()) {
      ValueMap[&GEP] = {BaseVal};
      return true;
    }

    gmir::LLTType PtrTy = convertType(Context, GEP.getType());
    if (!PtrTy)
      return false;

    llvm::Type *OffsetIRTy = DL->getIndexType(GEP.getPointerOperandType());
    unsigned IndexBitWidth = OffsetIRTy->getIntegerBitWidth();
    gmir::LLTType OffsetTy = convertType(Context, OffsetIRTy);

    bool NoUWrap = GEPOp.hasNoUnsignedWrap();
    bool NoUSWrap = GEPOp.hasNoUnsignedSignedWrap();
    bool InBounds = GEPOp.isInBounds();

    // A nonnegative constant offset added on top of a nusw/inbounds
    // pointer can't unsigned-wrap either -- IRTranslator.cpp's
    // PtrAddFlagsWithConst upgrade, applied only at constant-offset flush
    // points below, not the variable-index gmir.ptr_add further down.
    auto EmitConstOffset = [&](int64_t Offset) {
      auto ConstOp =
          gmir::ConstantOp::create(Builder, Builder.getUnknownLoc(), OffsetTy,
                                   Builder.getI64IntegerAttr(Offset));
      bool UpgradedNoUWrap = NoUWrap || (NoUSWrap && Offset >= 0);
      auto AddOp = gmir::PtrAddOp::create(
          Builder, Builder.getUnknownLoc(), PtrTy, BaseVal, ConstOp.getResult(),
          UpgradedNoUWrap ? Builder.getUnitAttr() : mlir::UnitAttr(),
          NoUSWrap ? Builder.getUnitAttr() : mlir::UnitAttr(),
          InBounds ? Builder.getUnitAttr() : mlir::UnitAttr());
      BaseVal = AddOp.getResult();
    };

    int64_t Offset = 0;
    for (auto GTI = gep_type_begin(GEP), GTE = gep_type_end(GEP); GTI != GTE;
         ++GTI) {
      llvm::Value *Idx = GTI.getOperand();
      if (llvm::StructType *StTy = GTI.getStructTypeOrNull()) {
        unsigned Field = cast<Constant>(Idx)->getUniqueInteger().getZExtValue();
        Offset += DL->getStructLayout(StTy)->getElementOffset(Field);
        continue;
      }
      // getSequentialElementStride returns a TypeSize; a scalable stride
      // (e.g. this step indexes into a <vscale x N x T> element) would
      // fatally abort the compiler on the old implicit conversion to
      // uint64_t below rather than falling back gracefully -- same class
      // of gap as computeGMIRLeafTypes's array-element case.
      TypeSize ElementSizeTS = GTI.getSequentialElementStride(*DL);
      if (ElementSizeTS.isScalable())
        return false;
      uint64_t ElementSize = ElementSizeTS.getFixedValue();
      if (auto *CI = dyn_cast<ConstantInt>(Idx)) {
        if (auto Val = CI->getValue().trySExtValue()) {
          Offset += ElementSize * *Val;
          continue;
        }
      }

      // Variable index: flush any accumulated constant offset first, then
      // emit Idx * ElementSize (if needed) + a ptr_add for Idx itself.
      if (Idx->getType()->getIntegerBitWidth() != IndexBitWidth)
        return false;
      if (Offset != 0) {
        EmitConstOffset(Offset);
        Offset = 0;
      }

      mlir::Value IdxVal;
      if (!getScalarOperand(Idx, IdxVal))
        return false;

      mlir::Value ScaledVal = IdxVal;
      if (ElementSize != 1) {
        auto ElemSizeConst = gmir::ConstantOp::create(
            Builder, Builder.getUnknownLoc(), OffsetTy,
            Builder.getI64IntegerAttr(static_cast<int64_t>(ElementSize)));
        auto MulOp =
            gmir::MulOp::create(Builder, Builder.getUnknownLoc(), OffsetTy,
                                IdxVal, ElemSizeConst.getResult());
        ScaledVal = MulOp.getResult();
      }

      // Raw (non-const-upgraded) flags for the variable-index add, per
      // IRTranslator.cpp.
      auto AddOp = gmir::PtrAddOp::create(
          Builder, Builder.getUnknownLoc(), PtrTy, BaseVal, ScaledVal,
          NoUWrap ? Builder.getUnitAttr() : mlir::UnitAttr(),
          NoUSWrap ? Builder.getUnitAttr() : mlir::UnitAttr(),
          InBounds ? Builder.getUnitAttr() : mlir::UnitAttr());
      BaseVal = AddOp.getResult();
    }

    if (Offset != 0)
      EmitConstOffset(Offset);

    ValueMap[&GEP] = {BaseVal};
    return true;
  }

  /// Imports a direct, non-vararg, non-tail/musttail CallInst with no
  /// byval/sret/inalloca/preallocated/byref/swifterror/swiftself/
  /// swiftasync parameter attributes, no operand bundles, that isn't an
  /// intrinsic or inline-asm call, and whose return type is void or
  /// convertType-representable (aggregate returns needing sret-demotion
  /// are conservatively out of scope for this milestone slice -- see
  /// GMIRDialect.td's gmir.call doc comment). Arguments flatten via
  /// computeGMIRLeafTypes/getOperands, the same mechanism as aggregate
  /// load/store. Sets the op's location to an mlir::OpaqueLoc wrapping
  /// &CI, the only way MLIRToGMIRTranslator can recover the original
  /// llvm::CallInst it needs for CallLowering::lowerCall's CallBase-taking
  /// overload -- see GMIRDialect.td's gmir.call doc comment for why that
  /// overload is used instead of hand-building a CallLoweringInfo.
  bool importCall(CallInst &CI) {
    if (CI.isInlineAsm() || CI.isTailCall() || CI.isMustTailCall() ||
        CI.hasOperandBundles() || CI.getFunctionType()->isVarArg())
      return false;

    llvm::Value *CalleeV = CI.getCalledOperand()->stripPointerCasts();
    if (!isa<Function, GlobalIFunc, GlobalAlias>(CalleeV))
      return false; // indirect call
    if (auto *F = dyn_cast<Function>(CalleeV))
      if (F->getIntrinsicID() != Intrinsic::not_intrinsic)
        return false;

    // llvm::Attribute vs. mlir::Attribute collide under this file's
    // blanket `using namespace llvm;`/`using namespace mlir;` -- same
    // class of ambiguity as VectorType/Value/Type/DenseMap noted
    // elsewhere in this codebase; explicit `llvm::` qualification
    // required.
    static constexpr llvm::Attribute::AttrKind BannedParamAttrs[] = {
        llvm::Attribute::ByVal,     llvm::Attribute::StructRet,
        llvm::Attribute::InAlloca,  llvm::Attribute::Preallocated,
        llvm::Attribute::ByRef,     llvm::Attribute::SwiftError,
        llvm::Attribute::SwiftSelf, llvm::Attribute::SwiftAsync};
    for (unsigned I = 0, E = CI.arg_size(); I != E; ++I) {
      // Fetch this argument's AttributeSet once rather than calling
      // CI.paramHasAttr(I, Kind) per banned kind -- paramHasAttr rederives
      // getAttributes().getParamAttrs(I) internally on every call.
      llvm::AttributeSet ParamAttrs = CI.getAttributes().getParamAttrs(I);
      for (llvm::Attribute::AttrKind Kind : BannedParamAttrs)
        if (ParamAttrs.hasAttribute(Kind))
          return false;
    }

    gmir::LLTType RetTy;
    if (!CI.getType()->isVoidTy()) {
      RetTy = convertType(Context, CI.getType());
      if (!RetTy)
        return false; // aggregate/vector/float return: out of scope
    }

    SmallVector<mlir::Value, 8> FlatArgs;
    SmallVector<int32_t, 8> LeafCounts;
    for (llvm::Value *Arg : CI.args()) {
      SmallVector<gmir::LLTType, 1> LeafTypes;
      SmallVector<uint64_t, 1> Offsets;
      if (!computeGMIRLeafTypes(Context, *DL, Arg->getType(), LeafTypes,
                                Offsets))
        return false;
      SmallVector<mlir::Value, 1> ArgLeaves;
      if (!getOperands(Arg, ArgLeaves) || ArgLeaves.size() != LeafTypes.size())
        return false;
      LeafCounts.push_back(static_cast<int32_t>(ArgLeaves.size()));
      FlatArgs.append(ArgLeaves.begin(), ArgLeaves.end());
    }

    mlir::Location Loc = mlir::OpaqueLoc::get<llvm::CallInst *>(&CI, &Context);
    auto Op = gmir::CallOp::create(
        Builder, Loc, RetTy ? mlir::TypeRange(RetTy) : mlir::TypeRange(),
        mlir::FlatSymbolRefAttr::get(&Context,
                                     cast<GlobalValue>(CalleeV)->getName()),
        Builder.getI64IntegerAttr(static_cast<int64_t>(CI.getCallingConv())),
        FlatArgs, Builder.getDenseI32ArrayAttr(LeafCounts));
    if (RetTy)
      ValueMap[&CI] = {Op.getResult()};
    return true;
  }

  bool importReturn(ReturnInst &Ret) {
    llvm::Value *RetVal = Ret.getReturnValue();
    if (!RetVal) {
      func::ReturnOp::create(Builder, Builder.getUnknownLoc());
      return true;
    }
    mlir::Value MLIRRetVal;
    if (!getScalarOperand(RetVal, MLIRRetVal))
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
    if (!getScalarOperand(Br.getCondition(), Cond))
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
      if (!getScalarOperand(PN.getIncomingValueForBlock(&FromBB), V))
        return false;
      Operands.push_back(V);
    }
    return true;
  }

  /// Resolves an llvm::Value operand to its full list of leaf mlir::Values
  /// (see ValueMap below): an already-mapped value (a formal arg, a PHI's
  /// block argument, or a prior instruction's result -- possibly multiple
  /// leaves for an aggregate, mirroring IRTranslator's ArrayRef<Register>-
  /// per-Value model, since !gmir.llt has no aggregate representation), or
  /// materializes a single-leaf gmir.constant (memoized in ValueMap) the
  /// first time a scalar ConstantInt is seen. Returns false if the operand
  /// can't be represented (e.g. an aggregate constant -- ConstantStruct/
  /// ConstantArray/ConstantAggregateZero are not materialized here, only
  /// ConstantInt is -- a ConstantInt too wide for 64 bits, or some other
  /// unmapped/unsupported value).
  bool getOperands(llvm::Value *V, SmallVectorImpl<mlir::Value> &Out) {
    // A single try_emplace both checks for and (speculatively) reserves
    // V's slot, rather than a find() here plus a separate operator[]
    // insert below -- two independent DenseMap probes on the (rare)
    // miss-then-materialize path. On any failure path below, the
    // speculative empty entry is erased again so a later lookup for the
    // same unresolvable V doesn't wrongly find a stale empty "resolved to
    // zero values" entry.
    auto [It, Inserted] = ValueMap.try_emplace(V);
    if (!Inserted) {
      Out.append(It->second.begin(), It->second.end());
      return true;
    }
    auto *CI = dyn_cast<ConstantInt>(V);
    if (!CI || CI->getValue().getSignificantBits() > 64) {
      ValueMap.erase(It);
      return false;
    }
    gmir::LLTType Ty = convertType(Context, CI->getType());
    if (!Ty) {
      ValueMap.erase(It);
      return false;
    }
    // Insert at the entry block's front-growing cursor, not wherever the
    // caller's Builder happens to be pointed -- see import()'s comment on
    // ConstantInsertPt for why: this constant may be memoized and reused
    // from a block that doesn't dominate the current one.
    mlir::OpBuilder::InsertionGuard Guard(Builder);
    Builder.setInsertionPoint(EntryBlock, ConstantInsertPt);
    auto ConstOp = gmir::ConstantOp::create(
        Builder, Builder.getUnknownLoc(), Ty,
        Builder.getI64IntegerAttr(CI->getSExtValue()));
    ConstantInsertPt = std::next(mlir::Block::iterator(ConstOp));
    It->second.push_back(ConstOp.getResult());
    Out.push_back(ConstOp.getResult());
    return true;
  }

  /// Thin single-leaf convenience wrapper over getOperands, for every call
  /// site that's structurally guaranteed non-aggregate (arithmetic/icmp
  /// operands, branch conditions, PHI incoming values, alloca/GEP pointer
  /// and index operands, a function's own return value -- convertType-able
  /// types are always exactly one leaf, so this asserts via the size
  /// check rather than silently truncating).
  bool getScalarOperand(llvm::Value *V, mlir::Value &Out) {
    SmallVector<mlir::Value, 1> Leaves;
    if (!getOperands(V, Leaves) || Leaves.size() != 1)
      return false;
    Out = Leaves[0];
    return true;
  }

  MLIRContext &Context;
  OpBuilder Builder;
  const llvm::DataLayout *DL = nullptr;
  /// Where the next memoized constant gets inserted -- see import()'s
  /// comment and getOperands's use of these.
  mlir::Block *EntryBlock = nullptr;
  mlir::Block::iterator ConstantInsertPt;
  llvm::DenseMap<BasicBlock *, Block *> BlockMap;
  /// Every SSA value maps to a list of leaf mlir::Values -- almost always
  /// exactly one, except for an aggregate-typed load result or store
  /// value, which has one leaf per computeGMIRLeafTypes leaf.
  llvm::DenseMap<llvm::Value *, SmallVector<mlir::Value, 1>> ValueMap;
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
