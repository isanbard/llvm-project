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
#include "llvm/CodeGen/GlobalISel/Utils.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/TargetSubtargetInfo.h"
#include "llvm/CodeGenTypes/LowLevelType.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"

using namespace llvm;

static LLT convertLLT(gmir::LLTType Ty) {
  // GMIRImporter only ever produces scalar-integer !gmir.llt values. Uses
  // LLT::integer() specifically, not the more general LLT::scalar(): some
  // targets' InstructionSelect patterns require isInteger() on their
  // generic MIR operands, which only LLT::integer() satisfies.
  return LLT::integer(Ty.getScalarSizeInBits());
}

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
                  const CallLowering &CLI, const BranchProbabilityInfo &BPI)
      : MF(MF), MIRBuilder(MIRBuilder), CLI(CLI), BPI(BPI) {}

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
        LLT Ty = convertLLT(cast<gmir::LLTType>(Arg.getType()));
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
          convertLLT(cast<gmir::LLTType>(Arg.getType())));
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

  bool translateOp(mlir::Operation &Op) {
    if (auto ConstOp = dyn_cast<gmir::ConstantOp>(&Op)) {
      LLT Ty = convertLLT(cast<gmir::LLTType>(ConstOp.getResult().getType()));
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
      LLT Ty = convertLLT(cast<gmir::LLTType>(ICmp.getResult().getType()));
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
    LLT Ty = convertLLT(cast<gmir::LLTType>(BinOp.getResult().getType()));     \
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

    // GMIRImporter only ever emits the ops handled above (plus
    // func::ReturnOp/gmir.br/gmir.brcond, handled directly in run()).
    return false;
  }

  MachineFunction &MF;
  MachineIRBuilder &MIRBuilder;
  const CallLowering &CLI;
  const BranchProbabilityInfo &BPI;
  llvm::DenseMap<mlir::Value, Register> ValueToReg;
  llvm::DenseMap<mlir::Block *, MachineBasicBlock *> BlockMap;
  llvm::DenseMap<mlir::Block *, SmallVector<MachineInstr *, 4>> SkeletonPhis;
  SmallVector<SmallVector<Register, 1>, 8> ArgRegStorage;
};
} // namespace

bool gmir::translate(mlir::func::FuncOp FuncOp, Function &F,
                     MachineFunction &MF, const BranchProbabilityInfo &BPI) {
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

  // CSE isn't enabled yet -- see createMIRBuilder's doc comment; this
  // translator switches it on once it has a reassociation-sensitive
  // consumer that benefits from it.
  std::unique_ptr<MachineIRBuilder> MIRBuilder =
      createMIRBuilder(MF, /*CSEInfo=*/nullptr);

  return GMIRToMIRWalker(MF, *MIRBuilder, *CLI, BPI).run(FuncOp, F, FuncInfo);
}
