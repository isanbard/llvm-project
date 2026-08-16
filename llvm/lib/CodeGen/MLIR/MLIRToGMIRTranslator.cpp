//===- MLIRToGMIRTranslator.cpp - gmir -> MIR bridge ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "MLIRToGMIRTranslator.h"
#include "IR/GMIRDialect.h"
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
using namespace mlir;

// scalarSizeInBits == 0 means "pointer in addressSpace" (see GMIRDialect.td's
// !gmir.llt doc comment) -- pointer size itself needs a DataLayout, hence
// the extra parameter (M1/M2 never needed one; only scalar ints existed).
//
// This checkout's LLT (LowLevelType.h) is a real fork-specific extension
// beyond upstream LLVM: it distinguishes Kind::INTEGER (LLT::integer(N),
// prints "iN") from a generic untyped Kind::ANY_SCALAR (LLT::scalar(N),
// prints "sN") -- upstream LLT has no such split. Using LLT::scalar() here
// (as M1/M2 always did) silently produced ANY_SCALAR operands that AArch64's
// InstructionSelect tablegen patterns don't match (they require isInteger()
// specifically) while X86's happened to be permissive enough not to care --
// masked entirely until now because a separate CallLowering::lowerReturn
// overload-resolution bug (see lowerReturn's comment) made every non-void
// AArch64 function fail earlier, before InstructionSelect ever got to
// reject the wrong LLT kind. Every gmir integer value must use
// LLT::integer(), not LLT::scalar().
static LLT convertLLT(gmir::LLTType Ty, const llvm::DataLayout &DL) {
  if (Ty.getScalarSizeInBits() == 0)
    return LLT::pointer(Ty.getAddressSpace(),
                         DL.getPointerSizeInBits(Ty.getAddressSpace()));
  return LLT::integer(Ty.getScalarSizeInBits());
}

namespace {
/// Bridges a `gmir`-only mlir::func::FuncOp into MF's MachineIR.
///
/// Two-pass structure mirrors GlobalISel::IRTranslator (see
/// ~/llvm/mlir_instruction_selection_plan.md, decision #5's refinement):
/// pass 1 creates every MachineBasicBlock and, for every non-entry block's
/// arguments, a skeleton G_PHI (result register, zero source operands) --
/// so branch translation in pass 2 can always resolve a forward or
/// back-edge target. Pass 2 translates each block's real ops; whenever a
/// gmir.br/gmir.brcond is translated, the *predecessor* directly appends
/// (vreg, MBB) operand pairs onto the destination's already-created
/// skeleton G_PHIs (IRTranslator.cpp:3740-3743's pattern), rather than
/// IRTranslator's separate final finishPendingPhis pass -- correct as long
/// as skeleton creation (pass 1) fully precedes any patching (pass 2),
/// which it does either way; the fully-separate pass additionally handles
/// critical-edge splitting and multi-edge predecessors, which this
/// (structured, non-critical-edge) subset doesn't need.
class GMIRToMIRWalker {
public:
  GMIRToMIRWalker(MachineFunction &MF, MachineIRBuilder &MIRBuilder,
                   const CallLowering &CLI, const BranchProbabilityInfo &BPI)
      : MF(MF), MIRBuilder(MIRBuilder), CLI(CLI), BPI(BPI),
        DL(MF.getDataLayout()) {}

  bool run(func::FuncOp FuncOp, Function &F, FunctionLoweringInfo &FuncInfo) {
    Block &EntryBB = FuncOp.getFunctionBody().front();

    // Pass 1: MBBs for every block, then skeleton G_PHIs for every
    // non-entry block's arguments (entry's arguments are formal
    // parameters, handled by CallLowering below, not phis). Associates
    // each MBB with its originating llvm::BasicBlock (mirrors
    // IRTranslator.cpp:4320-4339's CreateMachineBasicBlock(&BB)) purely so
    // downstream passes' assembly-comment annotations and block-placement
    // heuristics (e.g. loop-header alignment) see the same IR linkage
    // GlobalISel's own pipeline does -- confirmed by diffing output
    // against -global-isel, which was otherwise identical without this.
    {
      auto IRBlockIt = F.begin();
      for (Block &BB : FuncOp.getFunctionBody()) {
        BasicBlock &IRBB = *IRBlockIt++;
        MachineBasicBlock *MBB = MF.CreateMachineBasicBlock(&IRBB);
        MF.push_back(MBB);
        BlockMap[&BB] = MBB;
      }
    }
    for (Block &BB : FuncOp.getFunctionBody()) {
      if (&BB == &EntryBB)
        continue;
      MIRBuilder.setMBB(*BlockMap[&BB]);
      SmallVector<MachineInstr *, 4> Skeletons;
      for (BlockArgument Arg : BB.getArguments()) {
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
    for (Block &BB : FuncOp.getFunctionBody()) {
      BasicBlock &IRBB = *IRBlockIt++;
      MIRBuilder.setMBB(*BlockMap[&BB]);
      for (Operation &Op : BB) {
        if (auto Ret = dyn_cast<func::ReturnOp>(&Op)) {
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
  bool lowerFormalArguments(func::FuncOp FuncOp, Function &F,
                             FunctionLoweringInfo &FuncInfo) {
    // Stable storage for the ArrayRef<Register>s CallLowering expects --
    // one single-element vector per argument (no multi-register/aggregate
    // args in the current scalar-integer-only subset).
    Block &EntryBB = FuncOp.getFunctionBody().front();
    ArgRegStorage.reserve(EntryBB.getNumArguments());
    SmallVector<ArrayRef<Register>, 8> VRegArgs;
    for (BlockArgument Arg : EntryBB.getArguments()) {
      Register Reg = MIRBuilder.getMRI()->createGenericVirtualRegister(
          convertLLT(cast<gmir::LLTType>(Arg.getType()), DL));
      ValueToReg[Arg] = Reg;
      ArgRegStorage.push_back({Reg});
      VRegArgs.push_back(ArgRegStorage.back());
    }
    MIRBuilder.setMBB(*BlockMap[&EntryBB]);
    return CLI.lowerFormalArguments(MIRBuilder, F, VRegArgs, FuncInfo);
  }

  bool lowerReturn(func::ReturnOp Ret, BasicBlock &IRBB,
                    FunctionLoweringInfo &FuncInfo) {
    SmallVector<Register, 1> RetRegs;
    if (Ret.getNumOperands() == 1)
      RetRegs.push_back(ValueToReg.lookup(Ret.getOperand(0)));

    // lowerReturn wants the original llvm::Value*, not an mlir::Value --
    // re-fetch it directly from IRBB (the llvm::BasicBlock this mlir::Block
    // was imported from) rather than threading provenance through gmir.
    auto *OrigRet = cast<ReturnInst>(IRBB.getTerminator());
    // CallLowering::lowerReturn is overloaded: a 4-arg version (base class
    // default body: `return false;`) and a 5-arg version taking a trailing
    // SwiftErrorVReg (base class default: delegates to the 4-arg one when
    // !supportSwiftError()). Calling with exactly 4 args binds to the 4-arg
    // overload -- an EXACT match beats the 5-arg overload's default
    // argument in overload resolution -- so any target that only overrides
    // the 5-arg form (e.g. AArch64CallLowering) would otherwise silently
    // hit the base class's unconditional `return false` stub instead of
    // the real implementation. X86CallLowering happens to override the
    // 4-arg form directly, which is why this was masked there. Passing an
    // explicit null SwiftErrorVReg forces the 5-arg overload.
    return CLI.lowerReturn(MIRBuilder, OrigRet->getReturnValue(), RetRegs,
                            FuncInfo, /*SwiftErrorVReg=*/Register());
  }

  bool translateBr(gmir::BrOp Br, MachineBasicBlock *CurMBB,
                    BasicBlock &IRBB) {
    Block *DestBB = Br.getDest();
    MachineBasicBlock *DestMBB = BlockMap[DestBB];
    MIRBuilder.buildBr(*DestMBB);
    CurMBB->addSuccessor(DestMBB,
                          BPI.getEdgeProbability(&IRBB, DestMBB->getBasicBlock()));
    patchPhis(DestBB, Br.getDestOperands(), CurMBB);
    return true;
  }

  bool translateCondBr(gmir::CondBrOp Br, MachineBasicBlock *CurMBB,
                        BasicBlock &IRBB) {
    Register CondReg = ValueToReg.lookup(Br.getCondition());
    Block *TrueBB = Br.getTrueDest();
    Block *FalseBB = Br.getFalseDest();
    MachineBasicBlock *TrueMBB = BlockMap[TrueBB];
    MachineBasicBlock *FalseMBB = BlockMap[FalseBB];
    // Always emit both terminators (G_BRCOND to true, unconditional G_BR
    // to false) rather than eliding a fallthrough -- correctness over
    // branch-layout optimality; downstream branch-folding/block-placement
    // passes (unchanged, already in the pipeline) clean this up same as
    // they would for GlobalISel-selected code.
    MIRBuilder.buildBrCond(CondReg, *TrueMBB);
    MIRBuilder.buildBr(*FalseMBB);
    CurMBB->addSuccessor(TrueMBB,
                          BPI.getEdgeProbability(&IRBB, TrueMBB->getBasicBlock()));
    CurMBB->addSuccessor(FalseMBB,
                          BPI.getEdgeProbability(&IRBB, FalseMBB->getBasicBlock()));
    patchPhis(TrueBB, Br.getTrueDestOperands(), CurMBB);
    patchPhis(FalseBB, Br.getFalseDestOperands(), CurMBB);
    return true;
  }

  /// Appends this predecessor's (vreg, MBB) pair onto each of DestBB's
  /// already-created skeleton G_PHIs (one per block argument, matching
  /// Operands' order -- both ultimately derive from the same `BB.phis()`
  /// order in GMIRImporter).
  void patchPhis(Block *DestBB, Operation::operand_range Operands,
                 MachineBasicBlock *PredMBB) {
    ArrayRef<MachineInstr *> Skeletons = SkeletonPhis[DestBB];
    assert(Skeletons.size() == Operands.size() &&
           "gmir.br/brcond operand count must match dest block arg count");
    for (auto [Skeleton, Operand] : zip(Skeletons, Operands)) {
      Register Reg = ValueToReg.lookup(Operand);
      MachineInstrBuilder(MF, Skeleton).addUse(Reg).addMBB(PredMBB);
    }
  }

  bool translateOp(Operation &Op) {
    if (auto ConstOp = dyn_cast<gmir::ConstantOp>(&Op)) {
      LLT Ty = convertLLT(cast<gmir::LLTType>(ConstOp.getResult().getType()), DL);
      // GMIRImporter always stores gmir.constant's value sign-extended to
      // 64 bits (I64Attr), regardless of the actual !gmir.llt width, so it
      // must be truncated back down here -- buildConstant asserts the
      // APInt's bit width matches Ty exactly.
      APInt Val = ConstOp.getValueAttr().getValue().trunc(Ty.getScalarSizeInBits());
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

#define GMIR_BINOP_CASE(OpTy, Build)                                         \
  if (auto BinOp = dyn_cast<gmir::OpTy>(&Op)) {                              \
    Register LHS = ValueToReg.lookup(BinOp.getLhs());                        \
    Register RHS = ValueToReg.lookup(BinOp.getRhs());                        \
    LLT Ty = convertLLT(cast<gmir::LLTType>(BinOp.getResult().getType()), DL);\
    auto MIB = Build;                                                        \
    ValueToReg[BinOp.getResult()] = MIB.getReg(0);                           \
    return true;                                                             \
  }
    GMIR_BINOP_CASE(AddOp, MIRBuilder.buildAdd(Ty, LHS, RHS))
    GMIR_BINOP_CASE(SubOp, MIRBuilder.buildSub(Ty, LHS, RHS))
    GMIR_BINOP_CASE(MulOp, MIRBuilder.buildMul(Ty, LHS, RHS))
    GMIR_BINOP_CASE(AndOp, MIRBuilder.buildAnd(Ty, LHS, RHS))
    GMIR_BINOP_CASE(OrOp, MIRBuilder.buildOr(Ty, LHS, RHS))
    GMIR_BINOP_CASE(XorOp, MIRBuilder.buildXor(Ty, LHS, RHS))
    GMIR_BINOP_CASE(SDivOp,
                     MIRBuilder.buildInstr(TargetOpcode::G_SDIV, {Ty},
                                           {LHS, RHS}))
#undef GMIR_BINOP_CASE

    if (auto Alloca = dyn_cast<gmir::AllocaOp>(&Op)) {
      LLT Ty = convertLLT(cast<gmir::LLTType>(Alloca.getResult().getType()), DL);
      int FI = MF.getFrameInfo().CreateStackObject(
          Alloca.getSizeAttr().getInt(),
          Align(Alloca.getAlignAttr().getInt()), /*isSpillSlot=*/false);
      Register Res = MIRBuilder.getMRI()->createGenericVirtualRegister(Ty);
      MIRBuilder.buildFrameIndex(Res, FI);
      ValueToReg[Alloca.getResult()] = Res;
      return true;
    }

    if (auto Load = dyn_cast<gmir::LoadOp>(&Op)) {
      Register PtrReg = ValueToReg.lookup(Load.getPtr());
      LLT Ty = convertLLT(cast<gmir::LLTType>(Load.getResult().getType()), DL);
      Register Res = MIRBuilder.getMRI()->createGenericVirtualRegister(Ty);
      MachineMemOperand *MMO = buildMMO(
          MachineMemOperand::MOLoad, Ty, Load.getAlignAttr().getInt(),
          Load.getOrderingAttr().getInt(), Load.getSyncscopeAttr().getInt(),
          Load.getIsVolatile());
      MIRBuilder.buildLoad(Res, PtrReg, *MMO);
      ValueToReg[Load.getResult()] = Res;
      return true;
    }

    if (auto Store = dyn_cast<gmir::StoreOp>(&Op)) {
      Register ValReg = ValueToReg.lookup(Store.getValue());
      Register PtrReg = ValueToReg.lookup(Store.getPtr());
      LLT Ty = convertLLT(cast<gmir::LLTType>(Store.getValue().getType()), DL);
      MachineMemOperand *MMO = buildMMO(
          MachineMemOperand::MOStore, Ty, Store.getAlignAttr().getInt(),
          Store.getOrderingAttr().getInt(), Store.getSyncscopeAttr().getInt(),
          Store.getIsVolatile());
      MIRBuilder.buildStore(ValReg, PtrReg, *MMO);
      return true;
    }

    if (auto PtrAdd = dyn_cast<gmir::PtrAddOp>(&Op)) {
      Register PtrReg = ValueToReg.lookup(PtrAdd.getPtr());
      Register OffReg = ValueToReg.lookup(PtrAdd.getOffset());
      LLT Ty = convertLLT(cast<gmir::LLTType>(PtrAdd.getResult().getType()), DL);
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

    // GMIRImporter only ever emits the ops handled above (plus
    // func::ReturnOp/gmir.br/gmir.brcond, handled directly in run()).
    return false;
  }

  /// Shared MMO construction for gmir.load/gmir.store. MachinePointerInfo
  /// is left "unknown" (no Value-based provenance) and AAMDNodes are left
  /// empty -- a deliberate M3 scope exclusion (see design doc §1.5):
  /// unlike ordering (a correctness-affecting field once atomics are in
  /// play), AA metadata is a pure optimization hint, safe to omit, and not
  /// carried through gmir today.
  MachineMemOperand *buildMMO(MachineMemOperand::Flags BaseFlags, LLT Ty,
                               int64_t AlignBytes, int64_t OrderingVal,
                               int64_t SyncScopeVal, bool IsVolatile) {
    MachineMemOperand::Flags Flags = BaseFlags;
    if (IsVolatile)
      Flags |= MachineMemOperand::MOVolatile;
    return MF.getMachineMemOperand(
        MachinePointerInfo(), Flags, Ty, Align(AlignBytes), AAMDNodes(),
        /*Ranges=*/nullptr, static_cast<SyncScope::ID>(SyncScopeVal),
        static_cast<AtomicOrdering>(OrderingVal));
  }

  MachineFunction &MF;
  MachineIRBuilder &MIRBuilder;
  const CallLowering &CLI;
  const BranchProbabilityInfo &BPI;
  const llvm::DataLayout &DL;
  llvm::DenseMap<mlir::Value, Register> ValueToReg;
  llvm::DenseMap<Block *, MachineBasicBlock *> BlockMap;
  llvm::DenseMap<Block *, SmallVector<MachineInstr *, 4>> SkeletonPhis;
  SmallVector<SmallVector<Register, 1>, 8> ArgRegStorage;
};
} // namespace

bool gmir::translate(func::FuncOp FuncOp, Function &F, MachineFunction &MF,
                      const BranchProbabilityInfo &BPI,
                      MachineIRBuilder &MIRBuilder) {
  const CallLowering *CLI = MF.getSubtarget().getCallLowering();

  FunctionLoweringInfo FuncInfo;
  FuncInfo.clear();
  FuncInfo.MF = &MF;
  // FunctionLoweringInfo::BPI is non-const (used internally by some
  // CallLowering implementations for sret-demotion heuristics, per its
  // doc comment); not applicable to the scalar-only subset this bridges,
  // so left null rather than const_cast-ing BPI away just to populate it.
  FuncInfo.BPI = nullptr;
  FuncInfo.CanLowerReturn = CLI->checkReturnTypeForCallConv(MF);

  return GMIRToMIRWalker(MF, MIRBuilder, *CLI, BPI).run(FuncOp, F, FuncInfo);
}
