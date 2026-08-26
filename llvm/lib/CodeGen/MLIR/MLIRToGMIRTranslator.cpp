//===- MLIRToGMIRTranslator.cpp - gmir -> MIR bridge ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "MLIRToGMIRTranslator.h"
#include "GMIRLLTConversion.h"
#include "IR/GMIRDialect.h"
#include "mlir/IR/Location.h"
#include "llvm/ADT/APInt.h"
#include "llvm/Analysis/BranchProbabilityInfo.h"
#include "llvm/CodeGen/FunctionLoweringInfo.h"
#include "llvm/CodeGen/GlobalISel/CallLowering.h"
#include "llvm/CodeGen/GlobalISel/MachineIRBuilder.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/TargetSubtargetInfo.h"
#include "llvm/CodeGenTypes/LowLevelType.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"

using namespace llvm;
// convertLLT's unqualified calls below resolve to gmir::convertLLT (see
// GMIRLLTConversion.h) via argument-dependent lookup on its gmir::LLTType
// parameter -- every call site passes one, so no `gmir::` qualification is
// needed despite the function living in a nested namespace that `using
// namespace llvm;` above doesn't itself bring into scope.

namespace {
/// Bridges a `gmir`-only mlir::func::FuncOp into MF's MachineIR.
///
/// Two-pass structure: pass 1 creates every MachineBasicBlock and, for
/// every non-entry block's arguments, a skeleton G_PHI (result register,
/// zero source operands) -- so branch translation in pass 2 can always
/// resolve a forward or back-edge target. Pass 2 translates each block's
/// real ops; whenever a gmir.br/gmir.brcond is translated, the
/// *predecessor* directly appends (vreg, MBB) operand pairs onto the
/// destination's already-created skeleton G_PHIs, rather than a fully
/// separate final patching pass -- correct as long as skeleton creation
/// (pass 1) fully precedes any patching (pass 2), which it does either
/// way; a fully separate pass would additionally be needed to handle
/// critical-edge splitting and multi-edge predecessors, which this
/// (structured, non-critical-edge) subset doesn't need.
class GMIRToMIRWalker {
public:
  GMIRToMIRWalker(MachineFunction &MF, MachineIRBuilder &MIRBuilder,
                  const CallLowering &CLI, const BranchProbabilityInfo &BPI,
                  const gmir::CallInstMap &CallInsts)
      : MF(MF), MIRBuilder(MIRBuilder), CLI(CLI), BPI(BPI),
        DL(MF.getDataLayout()), CallInsts(CallInsts) {}

  bool run(mlir::func::FuncOp FuncOp, Function &F,
           FunctionLoweringInfo &FuncInfo) {
    mlir::Block &EntryBB = FuncOp.getFunctionBody().front();

    // Pass 1: MBBs for every block, then skeleton G_PHIs for every
    // non-entry block's arguments (entry's arguments are formal
    // parameters, handled by CallLowering below, not phis). Associates
    // each MBB with its originating llvm::BasicBlock (mirrors
    // IRTranslator's own CreateMachineBasicBlock(&BB) usage) purely so
    // downstream passes' assembly-comment annotations and block-placement
    // heuristics (e.g. loop-header alignment) see the same IR linkage
    // GlobalISel's own pipeline does -- confirmed by diffing output
    // against -global-isel, which was otherwise identical without this.
    {
      auto IRBlockIt = F.begin();
      for (mlir::Block &BB : FuncOp.getFunctionBody()) {
        BasicBlock &IRBB = *IRBlockIt++;
        MachineBasicBlock *MBB = MF.CreateMachineBasicBlock(&IRBB);
        MF.push_back(MBB);
        BlockMap[&BB] = MBB;
      }
    }
    for (mlir::Block &BB : FuncOp.getFunctionBody()) {
      if (&BB == &EntryBB)
        continue;
      MIRBuilder.setMBB(*BlockMap[&BB]);
      SmallVector<MachineInstr *, 4> Skeletons;
      for (mlir::BlockArgument Arg : BB.getArguments()) {
        LLT Ty = convertLLT(cast<gmir::LLTType>(Arg.getType()), DL);
        Register Reg = MIRBuilder.getMRI()->createGenericVirtualRegister(Ty);
        ValueToReg[Arg] = Reg;
        auto MIB = MIRBuilder.buildInstr(TargetOpcode::G_PHI, {Reg}, {});
        Skeletons.push_back(MIB.getInstr());
      }
      SkeletonPhis[&BB] = std::move(Skeletons);
    }

    if (!lowerFormalArguments(FuncOp, F, FuncInfo))
      return false;

    // Pass 2: translate each block's ops, in the same order as F's basic
    // blocks (guaranteed by GMIRImporter: FuncOp's block order is entry,
    // then F's remaining blocks in F's own iteration order), so a plain
    // parallel walk recovers each mlir::Block's original llvm::BasicBlock
    // (needed by lowerReturn, which wants the IR-level `ret`'s operand).
    auto IRBlockIt = F.begin();
    for (mlir::Block &BB : FuncOp.getFunctionBody()) {
      BasicBlock &IRBB = *IRBlockIt++;
      MIRBuilder.setMBB(*BlockMap[&BB]);
      for (mlir::Operation &Op : BB) {
        if (auto Ret = dyn_cast<mlir::func::ReturnOp>(&Op)) {
          if (!lowerReturn(Ret, IRBB, FuncInfo))
            return false;
        } else if (auto Br = dyn_cast<gmir::BrOp>(&Op)) {
          if (!translateBr(Br, BlockMap[&BB], IRBB))
            return false;
        } else if (auto CondBr = dyn_cast<gmir::CondBrOp>(&Op)) {
          if (!translateCondBr(CondBr, BlockMap[&BB], IRBB))
            return false;
        } else if (!translateOp(Op)) {
          return false;
        }
      }
    }
    return true;
  }

private:
  bool lowerFormalArguments(mlir::func::FuncOp FuncOp, Function &F,
                            FunctionLoweringInfo &FuncInfo) {
    // Stable storage for the ArrayRef<Register>s CallLowering expects --
    // one single-element vector per argument (no multi-register/aggregate
    // args in the current scalar-integer-only subset).
    mlir::Block &EntryBB = FuncOp.getFunctionBody().front();
    ArgRegStorage.reserve(EntryBB.getNumArguments());
    SmallVector<ArrayRef<Register>, 8> VRegArgs;
    for (mlir::BlockArgument Arg : EntryBB.getArguments()) {
      Register Reg = MIRBuilder.getMRI()->createGenericVirtualRegister(
          convertLLT(cast<gmir::LLTType>(Arg.getType()), DL));
      ValueToReg[Arg] = Reg;
      ArgRegStorage.push_back({Reg});
      VRegArgs.push_back(ArgRegStorage.back());
    }
    MIRBuilder.setMBB(*BlockMap[&EntryBB]);
    return CLI.lowerFormalArguments(MIRBuilder, F, VRegArgs, FuncInfo);
  }

  bool lowerReturn(mlir::func::ReturnOp Ret, BasicBlock &IRBB,
                   FunctionLoweringInfo &FuncInfo) {
    SmallVector<Register, 1> RetRegs;
    if (Ret.getNumOperands() == 1)
      RetRegs.push_back(ValueToReg.lookup(Ret.getOperand(0)));

    // lowerReturn wants the original llvm::Value*, not an mlir::Value --
    // re-fetch it directly from IRBB (the llvm::BasicBlock this mlir::Block
    // was imported from) rather than threading provenance through gmir.
    auto *OrigRet = cast<ReturnInst>(IRBB.getTerminator());
    // CallLowering::lowerReturn has a 4-arg overload and a 5-arg one
    // (trailing SwiftErrorVReg); the base class's 4-arg default just
    // returns false, and its 5-arg default only forwards to the 4-arg one
    // when the target doesn't support swifterror. Calling with exactly 4
    // arguments is an *exact* overload match, so it always resolves to
    // whichever one the target itself overrides -- for a target that only
    // overrides the 5-arg form (e.g. AArch64), that silently picks the
    // base class's unconditional `return false` stub instead of the real
    // implementation. Passing an explicit null SwiftErrorVReg forces the
    // 5-arg overload, which correctly reaches every target's real
    // implementation regardless of which one it happens to override.
    return CLI.lowerReturn(MIRBuilder, OrigRet->getReturnValue(), RetRegs,
                           FuncInfo, /*SwiftErrorVReg=*/Register());
  }

  bool translateBr(gmir::BrOp Br, MachineBasicBlock *CurMBB, BasicBlock &IRBB) {
    mlir::Block *DestBB = Br.getDest();
    MachineBasicBlock *DestMBB = BlockMap[DestBB];
    MIRBuilder.buildBr(*DestMBB);
    CurMBB->addSuccessor(
        DestMBB, BPI.getEdgeProbability(&IRBB, DestMBB->getBasicBlock()));
    patchPhis(DestBB, Br.getDestOperands(), CurMBB);
    return true;
  }

  bool translateCondBr(gmir::CondBrOp Br, MachineBasicBlock *CurMBB,
                       BasicBlock &IRBB) {
    Register CondReg = ValueToReg.lookup(Br.getCondition());
    mlir::Block *TrueBB = Br.getTrueDest();
    mlir::Block *FalseBB = Br.getFalseDest();
    MachineBasicBlock *TrueMBB = BlockMap[TrueBB];
    MachineBasicBlock *FalseMBB = BlockMap[FalseBB];
    // Always emit both terminators (G_BRCOND to true, unconditional G_BR
    // to false) rather than eliding a fallthrough -- correctness over
    // branch-layout optimality; downstream branch-folding/block-placement
    // passes (unchanged, already in the pipeline) clean this up same as
    // they would for GlobalISel-selected code.
    MIRBuilder.buildBrCond(CondReg, *TrueMBB);
    MIRBuilder.buildBr(*FalseMBB);
    CurMBB->addSuccessor(
        TrueMBB, BPI.getEdgeProbability(&IRBB, TrueMBB->getBasicBlock()));
    CurMBB->addSuccessor(
        FalseMBB, BPI.getEdgeProbability(&IRBB, FalseMBB->getBasicBlock()));
    patchPhis(TrueBB, Br.getTrueDestOperands(), CurMBB);
    patchPhis(FalseBB, Br.getFalseDestOperands(), CurMBB);
    return true;
  }

  /// Appends this predecessor's (vreg, MBB) pair onto each of DestBB's
  /// already-created skeleton G_PHIs (one per block argument, matching
  /// Operands' order -- both ultimately derive from the same `BB.phis()`
  /// order in GMIRImporter).
  void patchPhis(mlir::Block *DestBB, mlir::Operation::operand_range Operands,
                 MachineBasicBlock *PredMBB) {
    ArrayRef<MachineInstr *> Skeletons = SkeletonPhis[DestBB];
    assert(Skeletons.size() == Operands.size() &&
           "gmir.br/brcond operand count must match dest block arg count");
    for (auto [Skeleton, Operand] : zip(Skeletons, Operands)) {
      Register Reg = ValueToReg.lookup(Operand);
      MachineInstrBuilder(MF, Skeleton).addUse(Reg).addMBB(PredMBB);
    }
  }

  /// Shared plumbing for gmir.uaddo/usubo (the "O" -- no incoming carry --
  /// variants): allocates the dst/carryOut registers, calls Build to emit
  /// the actual instruction, then records both results. Build takes a
  /// lambda rather than this being a template over the GlobalISel builder
  /// method directly so callers can pass buildUAddo/buildUSubo's differing
  /// argument order-independent call sites uniformly.
  template <typename OpTy, typename BuildFn>
  bool translateCarryOOp(OpTy CarryOp, BuildFn Build) {
    Register LHS = ValueToReg.lookup(CarryOp.getLhs());
    Register RHS = ValueToReg.lookup(CarryOp.getRhs());
    LLT DstTy = convertLLT(cast<gmir::LLTType>(CarryOp.getDst().getType()), DL);
    LLT CarryTy =
        convertLLT(cast<gmir::LLTType>(CarryOp.getCarryOut().getType()), DL);
    Register DstReg = MIRBuilder.getMRI()->createGenericVirtualRegister(DstTy);
    Register CarryReg =
        MIRBuilder.getMRI()->createGenericVirtualRegister(CarryTy);
    Build(DstReg, CarryReg, LHS, RHS);
    ValueToReg[CarryOp.getDst()] = DstReg;
    ValueToReg[CarryOp.getCarryOut()] = CarryReg;
    return true;
  }

  /// Same as translateCarryOOp, for gmir.uadde/usube (the "E" variants,
  /// which also consume an incoming carry).
  template <typename OpTy, typename BuildFn>
  bool translateCarryEOp(OpTy CarryOp, BuildFn Build) {
    Register LHS = ValueToReg.lookup(CarryOp.getLhs());
    Register RHS = ValueToReg.lookup(CarryOp.getRhs());
    Register CarryInReg = ValueToReg.lookup(CarryOp.getCarryIn());
    LLT DstTy = convertLLT(cast<gmir::LLTType>(CarryOp.getDst().getType()), DL);
    LLT CarryTy =
        convertLLT(cast<gmir::LLTType>(CarryOp.getCarryOut().getType()), DL);
    Register DstReg = MIRBuilder.getMRI()->createGenericVirtualRegister(DstTy);
    Register CarryReg =
        MIRBuilder.getMRI()->createGenericVirtualRegister(CarryTy);
    Build(DstReg, CarryReg, LHS, RHS, CarryInReg);
    ValueToReg[CarryOp.getDst()] = DstReg;
    ValueToReg[CarryOp.getCarryOut()] = CarryReg;
    return true;
  }

  bool translateOp(mlir::Operation &Op) {
    if (auto ConstOp = dyn_cast<gmir::ConstantOp>(&Op)) {
      LLT Ty =
          convertLLT(cast<gmir::LLTType>(ConstOp.getResult().getType()), DL);
      // GMIRImporter always stores gmir.constant's value sign-extended to
      // 64 bits (I64Attr), regardless of the actual !gmir.llt width, so it
      // must be truncated back down here -- buildConstant asserts the
      // APInt's bit width matches Ty exactly.
      APInt Val =
          ConstOp.getValueAttr().getValue().trunc(Ty.getScalarSizeInBits());
      auto MIB = MIRBuilder.buildConstant(Ty, Val);
      ValueToReg[ConstOp.getResult()] = MIB.getReg(0);
      return true;
    }

    if (auto ICmp = dyn_cast<gmir::ICmpOp>(&Op)) {
      Register LHS = ValueToReg.lookup(ICmp.getLhs());
      Register RHS = ValueToReg.lookup(ICmp.getRhs());
      LLT Ty = convertLLT(cast<gmir::LLTType>(ICmp.getResult().getType()), DL);
      auto Pred = static_cast<CmpInst::Predicate>(
          ICmp.getPredicateAttr().getValue().getSExtValue());
      auto MIB = MIRBuilder.buildICmp(Pred, Ty, LHS, RHS);
      ValueToReg[ICmp.getResult()] = MIB.getReg(0);
      return true;
    }

#define GMIR_BINOP_CASE(OpTy, Build)                                           \
  if (auto BinOp = dyn_cast<gmir::OpTy>(&Op)) {                                \
    Register LHS = ValueToReg.lookup(BinOp.getLhs());                          \
    Register RHS = ValueToReg.lookup(BinOp.getRhs());                          \
    LLT Ty = convertLLT(cast<gmir::LLTType>(BinOp.getResult().getType()), DL); \
    auto MIB = Build;                                                          \
    ValueToReg[BinOp.getResult()] = MIB.getReg(0);                             \
    return true;                                                               \
  }
    GMIR_BINOP_CASE(AddOp, MIRBuilder.buildAdd(Ty, LHS, RHS))
    GMIR_BINOP_CASE(SubOp, MIRBuilder.buildSub(Ty, LHS, RHS))
    GMIR_BINOP_CASE(MulOp, MIRBuilder.buildMul(Ty, LHS, RHS))
    GMIR_BINOP_CASE(AndOp, MIRBuilder.buildAnd(Ty, LHS, RHS))
    GMIR_BINOP_CASE(OrOp, MIRBuilder.buildOr(Ty, LHS, RHS))
    GMIR_BINOP_CASE(XorOp, MIRBuilder.buildXor(Ty, LHS, RHS))
    GMIR_BINOP_CASE(
        SDivOp, MIRBuilder.buildInstr(TargetOpcode::G_SDIV, {Ty}, {LHS, RHS}))
#undef GMIR_BINOP_CASE

    // gmir.uaddo/uadde/usubo/usube -> G_U{ADD,SUB}{O,E}, and gmir.unmerge/
    // merge -> G_UNMERGE_VALUES/G_MERGE_VALUES: emitted only by
    // GMIRLegalizer's NarrowScalar patterns (see GMIRLegalizer.cpp), never
    // by GMIRImporter directly. Must be handled here regardless: once the
    // legalizer rewrites a gmir.add/sub/and/or/xor into this sequence,
    // this translator is the only thing standing between it and a "leave
    // it for the legacy selector" fallback, which would silently defeat
    // the whole point of running the legalizer at all.
    if (auto CarryOp = dyn_cast<gmir::UAddOOp>(&Op))
      return translateCarryOOp(
          CarryOp, [&](Register D, Register C, Register L, Register R) {
            MIRBuilder.buildUAddo(D, C, L, R);
          });
    if (auto CarryOp = dyn_cast<gmir::USubOOp>(&Op))
      return translateCarryOOp(
          CarryOp, [&](Register D, Register C, Register L, Register R) {
            MIRBuilder.buildUSubo(D, C, L, R);
          });
    if (auto CarryOp = dyn_cast<gmir::UAddEOp>(&Op))
      return translateCarryEOp(
          CarryOp, [&](Register D, Register C, Register L, Register R,
                       Register CI) { MIRBuilder.buildUAdde(D, C, L, R, CI); });
    if (auto CarryOp = dyn_cast<gmir::USubEOp>(&Op))
      return translateCarryEOp(
          CarryOp, [&](Register D, Register C, Register L, Register R,
                       Register CI) { MIRBuilder.buildUSube(D, C, L, R, CI); });

    if (auto Unmerge = dyn_cast<gmir::UnmergeOp>(&Op)) {
      Register SrcReg = ValueToReg.lookup(Unmerge.getSrc());
      SmallVector<Register, 4> DstRegs;
      for (mlir::Value Dst : Unmerge.getDsts()) {
        LLT Ty = convertLLT(cast<gmir::LLTType>(Dst.getType()), DL);
        DstRegs.push_back(
            MIRBuilder.getMRI()->createGenericVirtualRegister(Ty));
      }
      MIRBuilder.buildUnmerge(DstRegs, SrcReg);
      for (auto [Dst, Reg] : zip(Unmerge.getDsts(), DstRegs))
        ValueToReg[Dst] = Reg;
      return true;
    }

    if (auto Merge = dyn_cast<gmir::MergeOp>(&Op)) {
      SmallVector<Register, 4> SrcRegs;
      for (mlir::Value Src : Merge.getSrcs())
        SrcRegs.push_back(ValueToReg.lookup(Src));
      LLT Ty = convertLLT(cast<gmir::LLTType>(Merge.getResult().getType()), DL);
      Register Res = MIRBuilder.getMRI()->createGenericVirtualRegister(Ty);
      MIRBuilder.buildMergeValues(Res, SrcRegs);
      ValueToReg[Merge.getResult()] = Res;
      return true;
    }

    if (auto Alloca = dyn_cast<gmir::AllocaOp>(&Op)) {
      LLT Ty =
          convertLLT(cast<gmir::LLTType>(Alloca.getResult().getType()), DL);
      int FI = MF.getFrameInfo().CreateStackObject(
          Alloca.getSizeAttr().getInt(), Align(Alloca.getAlignAttr().getInt()),
          /*isSpillSlot=*/false);
      Register Res = MIRBuilder.getMRI()->createGenericVirtualRegister(Ty);
      MIRBuilder.buildFrameIndex(Res, FI);
      ValueToReg[Alloca.getResult()] = Res;
      return true;
    }

    if (auto Load = dyn_cast<gmir::LoadOp>(&Op)) {
      Register PtrReg = ValueToReg.lookup(Load.getPtr());
      LLT Ty = convertLLT(cast<gmir::LLTType>(Load.getResult().getType()), DL);
      Register Res = MIRBuilder.getMRI()->createGenericVirtualRegister(Ty);
      MachineMemOperand::Flags ExtraFlags = MachineMemOperand::MONone;
      if (Load.getIsInvariant())
        ExtraFlags |= MachineMemOperand::MOInvariant;
      if (Load.getIsNonTemporal())
        ExtraFlags |= MachineMemOperand::MONonTemporal;
      MachineMemOperand *MMO = buildMMO(
          MachineMemOperand::MOLoad | ExtraFlags, Ty,
          Load.getAlignAttr().getInt(), Load.getOrderingAttr().getInt(),
          Load.getSyncscopeAttr().getInt(), Load.getIsVolatile());
      MIRBuilder.buildLoad(Res, PtrReg, *MMO);
      ValueToReg[Load.getResult()] = Res;
      return true;
    }

    if (auto Store = dyn_cast<gmir::StoreOp>(&Op)) {
      Register ValReg = ValueToReg.lookup(Store.getValue());
      Register PtrReg = ValueToReg.lookup(Store.getPtr());
      LLT Ty = convertLLT(cast<gmir::LLTType>(Store.getValue().getType()), DL);
      MachineMemOperand::Flags ExtraFlags = MachineMemOperand::MONone;
      if (Store.getIsNonTemporal())
        ExtraFlags |= MachineMemOperand::MONonTemporal;
      MachineMemOperand *MMO = buildMMO(
          MachineMemOperand::MOStore | ExtraFlags, Ty,
          Store.getAlignAttr().getInt(), Store.getOrderingAttr().getInt(),
          Store.getSyncscopeAttr().getInt(), Store.getIsVolatile());
      MIRBuilder.buildStore(ValReg, PtrReg, *MMO);
      return true;
    }

    if (auto PtrAdd = dyn_cast<gmir::PtrAddOp>(&Op)) {
      Register PtrReg = ValueToReg.lookup(PtrAdd.getPtr());
      Register OffReg = ValueToReg.lookup(PtrAdd.getOffset());
      LLT Ty =
          convertLLT(cast<gmir::LLTType>(PtrAdd.getResult().getType()), DL);
      unsigned Flags = 0;
      if (PtrAdd.getNoUWrap())
        Flags |= MachineInstr::MIFlag::NoUWrap;
      if (PtrAdd.getNoUSWrap())
        Flags |= MachineInstr::MIFlag::NoUSWrap;
      if (PtrAdd.getInBounds())
        Flags |= MachineInstr::MIFlag::InBounds;
      auto MIB = MIRBuilder.buildPtrAdd(Ty, PtrReg, OffReg, Flags);
      ValueToReg[PtrAdd.getResult()] = MIB.getReg(0);
      return true;
    }

    if (auto Call = dyn_cast<gmir::CallOp>(&Op))
      return translateCall(Call);

    // GMIRImporter only ever emits the ops handled above (plus
    // func::ReturnOp/gmir.br/gmir.brcond, handled directly in run()).
    return false;
  }

  /// Translates gmir.call by recovering the original llvm::CallInst from
  /// CallInsts (populated at import time -- see GMIRImporter.cpp's
  /// importCall) and calling CallLowering::lowerCall's CallBase-taking
  /// overload directly on it, rather than hand-building a
  /// CallLoweringInfo -- see GMIRDialect.td's gmir.call doc comment for
  /// why: that overload already derives argument flags from IR
  /// Attributes, recomputes tail-call eligibility, and handles
  /// sret-demotion, all logic worth reusing rather than duplicating. The
  /// zero/nullopt/unreachable arguments below are each provably unused
  /// given importCall's bail-out list: no swifterror, no ptrauth bundle,
  /// no convergencectrl bundle, and the callee is always direct (so
  /// GetCalleeReg is never actually invoked).
  bool translateCall(gmir::CallOp Call) {
    auto *CI = CallInsts.lookup(Call.getOperation());
    // Every gmir.call GMIRImporter ever produces carries an entry by
    // construction (see importCall), so this "should" always be non-null
    // -- but unlike everywhere else in this file that assumes an
    // invariant impossible to violate today, a future legalizer pattern
    // or rewriter helper cloning/rebuilding a gmir.call could plausibly
    // produce one CallInsts never learns about. A real check-and-bail
    // costs nothing here and keeps this path inside the same "never
    // crash, always fall back gracefully" discipline every other
    // unsupported-construct path in this file follows.
    if (!CI)
      return false;

    SmallVector<Register, 1> ResRegs;
    if (Call.getNumResults() == 1) {
      LLT Ty = convertLLT(cast<gmir::LLTType>(Call.getResult().getType()), DL);
      ResRegs.push_back(MIRBuilder.getMRI()->createGenericVirtualRegister(Ty));
    }

    // Regroup the flat leaf-operand list back into one ArrayRef<Register>
    // per original IR argument, using argLeafCounts. CallArgRegStorage
    // provides one flat, stable backing store for every ArgRegs slice --
    // all leaf registers are pushed up front (reserve()'d to their exact
    // final count, so no reallocation ever occurs mid-loop), then ArgRegs
    // is built as a second pass of non-overlapping slices into it. Unlike
    // a vector-of-vectors, there's no inner SmallVector whose own
    // relocation could dangle an already-captured ArrayRef<Register>.
    SmallVector<Register, 8> CallArgRegStorage;
    CallArgRegStorage.reserve(Call.getArgs().size());
    for (mlir::Value Arg : Call.getArgs())
      CallArgRegStorage.push_back(ValueToReg.lookup(Arg));

    SmallVector<ArrayRef<Register>, 8> ArgRegs;
    unsigned FlatIdx = 0;
    for (int32_t Count : Call.getArgLeafCounts()) {
      ArgRegs.push_back(
          ArrayRef<Register>(CallArgRegStorage).slice(FlatIdx, Count));
      FlatIdx += Count;
    }

    if (!CLI.lowerCall(
            MIRBuilder, *CI, ResRegs, ArgRegs, /*SwiftErrorVReg=*/Register(),
            /*PAI=*/std::nullopt, /*ConvergenceCtrlToken=*/Register(),
            /*GetCalleeReg=*/[]() -> Register {
              llvm_unreachable("gmir.call is always a direct call in this "
                               "subset -- see importCall's bail-out list");
            }))
      return false;

    if (Call.getNumResults() == 1)
      ValueToReg[Call.getResult()] = ResRegs[0];
    return true;
  }

  /// Shared MMO construction for gmir.load/gmir.store. MOInvariant/
  /// MONonTemporal (set by the two callers above, from gmir.load/store's
  /// isInvariant/isNonTemporal attrs) are passed in via BaseFlags, mirroring
  /// TargetLoweringBase::getLoadMemOperandFlags/getStoreMemOperandFlags
  /// exactly. MachinePointerInfo is left "unknown" (no Value-based
  /// provenance) and AAMDNodes are left empty -- AA metadata is a pure
  /// optimization hint, safe to omit, and not carried through gmir today,
  /// unlike ordering (a correctness-affecting field once atomics are in
  /// play). MODereferenceable is never set either: its real derivation
  /// needs isDereferenceableAndAlignedPointer, which needs an
  /// AssumptionCache/TargetLibraryInfo the importer doesn't have threaded
  /// in (see gmir.load's doc comment) -- all deliberate, still-scoped-out
  /// exclusions, not oversights.
  MachineMemOperand *buildMMO(MachineMemOperand::Flags BaseFlags, LLT Ty,
                              int64_t AlignBytes, int64_t OrderingVal,
                              int64_t SyncScopeVal, bool IsVolatile) {
    MachineMemOperand::Flags Flags = BaseFlags;
    if (IsVolatile)
      Flags |= MachineMemOperand::MOVolatile;
    return MF.getMachineMemOperand(MachinePointerInfo(), Flags, Ty,
                                   Align(AlignBytes), MMOMetadata(),
                                   static_cast<SyncScope::ID>(SyncScopeVal),
                                   static_cast<AtomicOrdering>(OrderingVal));
  }

  MachineFunction &MF;
  MachineIRBuilder &MIRBuilder;
  const CallLowering &CLI;
  const BranchProbabilityInfo &BPI;
  const llvm::DataLayout &DL;
  const gmir::CallInstMap &CallInsts;
  llvm::DenseMap<mlir::Value, Register> ValueToReg;
  llvm::DenseMap<mlir::Block *, MachineBasicBlock *> BlockMap;
  llvm::DenseMap<mlir::Block *, SmallVector<MachineInstr *, 4>> SkeletonPhis;
  SmallVector<SmallVector<Register, 1>, 8> ArgRegStorage;
};
} // namespace

bool gmir::translate(mlir::func::FuncOp FuncOp, Function &F,
                     MachineFunction &MF, const BranchProbabilityInfo &BPI,
                     MachineIRBuilder &MIRBuilder,
                     const CallInstMap &CallInsts) {
  const CallLowering *CLI = MF.getSubtarget().getCallLowering();
  // getCallLowering() defaults to nullptr (TargetSubtargetInfo's base
  // implementation) and isn't overridden by every in-tree target (e.g.
  // SystemZ, Hexagon, Sparc, XCore, VE, LoongArch, NVPTX); -enable-mlir-isel
  // has no target allowlist, so a null CLI here is a real, reachable case,
  // not a should-never-happen one -- fall back gracefully, exactly what
  // this function's own header doc comment already promises callers.
  if (!CLI)
    return false;

  FunctionLoweringInfo FuncInfo;
  FuncInfo.clear();
  FuncInfo.MF = &MF;
  // FunctionLoweringInfo::BPI is non-const (used internally by some
  // CallLowering implementations for sret-demotion heuristics, per its
  // doc comment); not applicable to the scalar-only subset this bridges,
  // so left null rather than const_cast-ing BPI away just to populate it.
  FuncInfo.BPI = nullptr;
  FuncInfo.CanLowerReturn = CLI->checkReturnTypeForCallConv(MF);

  return GMIRToMIRWalker(MF, MIRBuilder, *CLI, BPI, CallInsts)
      .run(FuncOp, F, FuncInfo);
}
