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
#include "llvm/CodeGen/FunctionLoweringInfo.h"
#include "llvm/CodeGen/GlobalISel/CallLowering.h"
#include "llvm/CodeGen/GlobalISel/MachineIRBuilder.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/TargetSubtargetInfo.h"
#include "llvm/CodeGenTypes/LowLevelType.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"

using namespace llvm;
using namespace mlir;

static LLT convertLLT(gmir::LLTType Ty) {
  // M1 only ever produces scalar-integer !gmir.llt values.
  return LLT::scalar(Ty.getScalarSizeInBits());
}

namespace {
class GMIRToMIRWalker {
public:
  GMIRToMIRWalker(MachineIRBuilder &MIRBuilder, const CallLowering &CLI)
      : MIRBuilder(MIRBuilder), CLI(CLI) {}

  bool run(func::FuncOp FuncOp, Function &F, FunctionLoweringInfo &FuncInfo) {
    if (!lowerFormalArguments(FuncOp, F, FuncInfo))
      return false;

    for (Operation &Op : FuncOp.getFunctionBody().front()) {
      if (auto Ret = dyn_cast<func::ReturnOp>(&Op))
        return lowerReturn(Ret, F, FuncInfo);
      if (!translateOp(Op))
        return false;
    }
    // Should always end with a func::ReturnOp (GMIRImporter guarantees
    // this); fall back gracefully rather than assert if that invariant
    // is ever violated.
    return false;
  }

private:
  bool lowerFormalArguments(func::FuncOp FuncOp, Function &F,
                             FunctionLoweringInfo &FuncInfo) {
    // Stable storage for the ArrayRef<Register>s CallLowering expects --
    // one single-element vector per argument (no multi-register/aggregate
    // args in M1's scalar-integer-only subset).
    ArgRegStorage.reserve(FuncOp.getNumArguments());
    SmallVector<ArrayRef<Register>, 8> VRegArgs;
    for (BlockArgument Arg : FuncOp.getArguments()) {
      Register Reg = MIRBuilder.getMRI()->createGenericVirtualRegister(
          convertLLT(cast<gmir::LLTType>(Arg.getType())));
      ValueToReg[Arg] = Reg;
      ArgRegStorage.push_back({Reg});
      VRegArgs.push_back(ArgRegStorage.back());
    }
    return CLI.lowerFormalArguments(MIRBuilder, F, VRegArgs, FuncInfo);
  }

  bool lowerReturn(func::ReturnOp Ret, Function &F,
                    FunctionLoweringInfo &FuncInfo) {
    SmallVector<Register, 1> RetRegs;
    if (Ret.getNumOperands() == 1)
      RetRegs.push_back(ValueToReg.lookup(Ret.getOperand(0)));

    // lowerReturn wants the original llvm::Value*, not an mlir::Value --
    // re-fetch it directly rather than threading provenance through gmir
    // (M1 is single-basic-block only, so this is always F's one and only
    // terminator).
    auto *OrigRet = cast<ReturnInst>(F.getEntryBlock().getTerminator());
    return CLI.lowerReturn(MIRBuilder, OrigRet->getReturnValue(), RetRegs,
                            FuncInfo);
  }

  bool translateOp(Operation &Op) {
    if (auto ConstOp = dyn_cast<gmir::ConstantOp>(&Op)) {
      LLT Ty = convertLLT(cast<gmir::LLTType>(ConstOp.getResult().getType()));
      // GMIRImporter always stores gmir.constant's value sign-extended to
      // 64 bits (I64Attr), regardless of the actual !gmir.llt width, so it
      // must be truncated back down here -- buildConstant asserts the
      // APInt's bit width matches Ty exactly.
      APInt Val = ConstOp.getValueAttr().getValue().trunc(Ty.getScalarSizeInBits());
      auto MIB = MIRBuilder.buildConstant(Ty, Val);
      ValueToReg[ConstOp.getResult()] = MIB.getReg(0);
      return true;
    }

#define GMIR_BINOP_CASE(OpTy, Build)                                         \
  if (auto BinOp = dyn_cast<gmir::OpTy>(&Op)) {                              \
    Register LHS = ValueToReg.lookup(BinOp.getLhs());                        \
    Register RHS = ValueToReg.lookup(BinOp.getRhs());                        \
    LLT Ty = convertLLT(cast<gmir::LLTType>(BinOp.getResult().getType()));   \
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

    // GMIRImporter only ever emits the ops handled above.
    return false;
  }

  MachineIRBuilder &MIRBuilder;
  const CallLowering &CLI;
  llvm::DenseMap<mlir::Value, Register> ValueToReg;
  SmallVector<SmallVector<Register, 1>, 8> ArgRegStorage;
};
} // namespace

bool gmir::translate(func::FuncOp FuncOp, Function &F, MachineFunction &MF) {
  const CallLowering *CLI = MF.getSubtarget().getCallLowering();

  FunctionLoweringInfo FuncInfo;
  FuncInfo.clear();
  FuncInfo.MF = &MF;
  FuncInfo.BPI = nullptr;
  FuncInfo.CanLowerReturn = CLI->checkReturnTypeForCallConv(MF);

  MachineBasicBlock *MBB = MF.CreateMachineBasicBlock();
  MF.push_back(MBB);

  MachineIRBuilder MIRBuilder(MF);
  MIRBuilder.setMBB(*MBB);

  return GMIRToMIRWalker(MIRBuilder, *CLI).run(FuncOp, F, FuncInfo);
}
