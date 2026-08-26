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
#include "llvm/ADT/STLExtras.h"
#include "llvm/CodeGen/Analysis.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GetElementPtrTypeIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Operator.h"
#include "llvm/Support/Alignment.h"
#include "llvm/Support/MathExtras.h"
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
/// and ::mlir, so this file explicitly qualifies every mlir:: type/call
/// instead of also pulling in `using namespace mlir;` alongside `using
/// namespace llvm;` below -- this file implements code in ::llvm, not
/// ::mlir, so only the former is the standard-sanctioned exception to
/// LLVM's "don't `using namespace`" rule (see CodingStandards.md); the
/// latter would just be inviting exactly this Value/Type-style
/// collision instead of avoiding it.
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
// otherwise silently become a negative size. importGEP's offset
// accumulation below needs the same guard: casting an out-of-range
// uint64_t field/element offset straight into the int64_t
// AddOverflow/MulOverflow calls below would corrupt the value *before*
// those calls ever run, defeating the overflow check they're there to
// provide -- the check must see the original, correctly-signed
// magnitude, not a value already mangled by an unchecked narrowing cast.
static std::optional<int64_t> checkedU64ToI64(uint64_t V) {
  if (V > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()))
    return std::nullopt;
  return static_cast<int64_t>(V);
}

namespace {

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
/// IRTranslator's own computeValueLLTs is a thin wrapper over (it just maps
/// ComputeValueTypes's leaf llvm::Types through getLLTForType; this does
/// the same through convertType instead). Reusing it directly, rather than
/// hand-porting the struct/array walk, means any future correctness fix to
/// it benefits every caller without a separate re-port -- and it already
/// handles a scalable type nested inside a struct/array correctly via
/// TypeSize-preserving arithmetic throughout, not just a direct array
/// element.
bool computeGMIRLeafTypes(mlir::MLIRContext &Context,
                          const llvm::DataLayout &DL, llvm::Type *Ty,
                          SmallVectorImpl<gmir::LLTType> &Types,
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
    // Memoized constants (see getOperands) are materialized at a fixed,
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
      ValueMap[&Arg] = {FuncOp.getArgument(ArgIdx++)};

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
      // Anything else (calls, switches, casts, selects, ...) is out of
      // scope for now -- fall back rather than mistranslate. Note:
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
    if (!getScalarOperand(BinOp.getOperand(0), LHS) ||
        !getScalarOperand(BinOp.getOperand(1), RHS))
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
    ValueMap[&AI] = {Op.getResult()};
    return true;
  }

  /// Returns BasePtr unchanged when Offset==0 (skip-if-zero, mirroring
  /// MachineIRBuilder::materializePtrAdd's short-circuit -- the common
  /// case, since most loads/stores are scalar/pointer with a single
  /// zero-offset leaf), else emits gmir.constant(Offset) +
  /// gmir.ptr_add(BasePtr, that constant) with NoUWrap+InBounds set
  /// unconditionally (mirrors MachineIRBuilder::materializeObjectPtrOffset's
  /// fixed flags -- this is sub-object offset math within an already-valid
  /// object, not a user GEP, so it always gets the strong nuw+inbounds
  /// guarantee regardless of anything else).
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
    // Unlike PtrTy (pointers always convert successfully -- convertType has
    // no width restriction for them), the pointer's index type is an
    // *integer* whose width is datalayout-defined, not statically bounded,
    // so it can hit convertType's >64-bit rejection on an unusual custom
    // datalayout (e.g. `p:128:128`). Bail like every other convertType-
    // consuming path in this file, rather than handing a null type to gmir
    // op construction below.
    gmir::LLTType OffsetTy = convertType(Context, DL->getIndexType(PtrIRTy));
    if (!OffsetTy)
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
    // See importLoad's identical check above for why OffsetTy (unlike
    // PtrTy) needs one -- its width is datalayout-defined, not statically
    // bounded.
    gmir::LLTType OffsetTy = convertType(Context, DL->getIndexType(PtrIRTy));
    if (!OffsetTy)
      return false;
    // Mirrors TargetLoweringBase::getStoreMemOperandFlags's MONonTemporal
    // derivation exactly (stores have no invariant or dereferenceable
    // concept, unlike loads).
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
  /// VectorType); a variable index whose integer bit width doesn't match
  /// the pointer-index type's width, since gmir has no sext/trunc op yet
  /// to fix that up (see gmir.ptr_add's doc comment); and int64_t overflow
  /// while accumulating the constant offset (a plain += could wrap for a
  /// pathological but constructible GEP chain, e.g. deeply nested/huge
  /// structs, producing a wrong, in-bounds-looking offset instead of
  /// failing closed -- same bug class as importAlloca's overflow guard).
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
    // Unlike PtrTy (pointers always convert successfully -- convertType has
    // no width restriction for them), the pointer's index type is an
    // *integer* whose width is datalayout-defined, not statically bounded,
    // so it can hit convertType's >64-bit rejection on an unusual custom
    // datalayout (e.g. `p:128:128`). Bail like every other convertType-
    // consuming path in this file, rather than handing a null type to gmir
    // op construction below.
    gmir::LLTType OffsetTy = convertType(Context, OffsetIRTy);
    if (!OffsetTy)
      return false;

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
        uint64_t FieldOff = DL->getStructLayout(StTy)->getElementOffset(Field);
        auto SignedFieldOff = checkedU64ToI64(FieldOff);
        if (!SignedFieldOff || AddOverflow(Offset, *SignedFieldOff, Offset))
          return false;
        continue;
      }
      // getSequentialElementStride returns a TypeSize; a scalable stride
      // (e.g. this step indexes into a <vscale x N x T> element) would
      // fatally abort the compiler on an implicit conversion to uint64_t
      // rather than falling back gracefully -- same class of gap as
      // computeGMIRLeafTypes's array-element case.
      TypeSize ElementSizeTS = GTI.getSequentialElementStride(*DL);
      if (ElementSizeTS.isScalable())
        return false;
      uint64_t ElementSize = ElementSizeTS.getFixedValue();
      if (auto *CI = dyn_cast<ConstantInt>(Idx)) {
        if (auto Val = CI->getValue().trySExtValue()) {
          // A large constant array index times a large element size can
          // overflow the product, and/or overflow Offset when added in --
          // same overflow concern as the struct-offset case above.
          auto SignedElementSize = checkedU64ToI64(ElementSize);
          int64_t Prod;
          if (!SignedElementSize ||
              MulOverflow(*SignedElementSize, *Val, Prod) ||
              AddOverflow(Offset, Prod, Offset))
            return false;
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

  bool importReturn(ReturnInst &Ret) {
    llvm::Value *RetVal = Ret.getReturnValue();
    if (!RetVal) {
      mlir::func::ReturnOp::create(Builder, Builder.getUnknownLoc());
      return true;
    }
    mlir::Value MLIRRetVal;
    if (!getScalarOperand(RetVal, MLIRRetVal))
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
    if (!getScalarOperand(Br.getCondition(), Cond))
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
    auto It = ValueMap.find(V);
    if (It != ValueMap.end()) {
      Out.append(It->second.begin(), It->second.end());
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
    ValueMap[V] = {ConstOp.getResult()};
    Out.push_back(ConstOp.getResult());
    return true;
  }

  /// Thin single-leaf convenience wrapper over getOperands, for every call
  /// site that's structurally guaranteed non-aggregate (arithmetic/icmp
  /// operands, branch conditions, PHI incoming values, alloca/GEP pointer
  /// and index operands, a function's own return value -- convertType-able
  /// types are always exactly one leaf, so this asserts via the size check
  /// rather than silently truncating).
  bool getScalarOperand(llvm::Value *V, mlir::Value &Out) {
    SmallVector<mlir::Value, 1> Leaves;
    if (!getOperands(V, Leaves) || Leaves.size() != 1)
      return false;
    Out = Leaves[0];
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
  /// Every SSA value maps to a list of leaf mlir::Values -- almost always
  /// exactly one, except for an aggregate-typed load result or store
  /// value, which has one leaf per computeGMIRLeafTypes leaf.
  llvm::DenseMap<llvm::Value *, SmallVector<mlir::Value, 1>> ValueMap;
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
