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
class GMIRToMIRWalker {
public:
  GMIRToMIRWalker(MachineIRBuilder &MIRBuilder, const CallLowering &CLI)
      : MIRBuilder(MIRBuilder), CLI(CLI) {}

  bool run(mlir::func::FuncOp FuncOp, Function &F,
           FunctionLoweringInfo &FuncInfo) {
    if (!lowerFormalArguments(FuncOp, F, FuncInfo))
      return false;

    for (mlir::Operation &Op : FuncOp.getFunctionBody().front()) {
      if (auto Ret = dyn_cast<mlir::func::ReturnOp>(&Op))
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
  bool lowerFormalArguments(mlir::func::FuncOp FuncOp, Function &F,
                            FunctionLoweringInfo &FuncInfo) {
    // Stable storage for the ArrayRef<Register>s CallLowering expects --
    // one single-element vector per argument (no multi-register/aggregate
    // args in this currently-supported subset).
    ArgRegStorage.reserve(FuncOp.getNumArguments());
    SmallVector<ArrayRef<Register>, 8> VRegArgs;
    for (mlir::BlockArgument Arg : FuncOp.getArguments()) {
      Register Reg = MIRBuilder.getMRI()->createGenericVirtualRegister(
          convertLLT(cast<gmir::LLTType>(Arg.getType())));
      ValueToReg[Arg] = Reg;
      ArgRegStorage.push_back({Reg});
      VRegArgs.push_back(ArgRegStorage.back());
    }
    return CLI.lowerFormalArguments(MIRBuilder, F, VRegArgs, FuncInfo);
  }

  bool lowerReturn(mlir::func::ReturnOp Ret, Function &F,
                   FunctionLoweringInfo &FuncInfo) {
    SmallVector<Register, 1> RetRegs;
    if (Ret.getNumOperands() == 1)
      RetRegs.push_back(ValueToReg.lookup(Ret.getOperand(0)));

    // lowerReturn wants the original llvm::Value*, not an mlir::Value --
    // re-fetch it directly rather than threading provenance through gmir
    // (single-basic-block only for now, so this is always F's one and
    // only terminator).
    auto *OrigRet = cast<ReturnInst>(F.getEntryBlock().getTerminator());
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

    // GMIRImporter only ever emits the ops handled above.
    return false;
  }

  MachineIRBuilder &MIRBuilder;
  const CallLowering &CLI;
  llvm::DenseMap<mlir::Value, Register> ValueToReg;
  SmallVector<SmallVector<Register, 1>, 8> ArgRegStorage;
};
} // namespace

bool gmir::translate(mlir::func::FuncOp FuncOp, Function &F,
                     MachineFunction &MF) {
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
  FuncInfo.BPI = nullptr;
  FuncInfo.CanLowerReturn = CLI->checkReturnTypeForCallConv(MF);

  MachineBasicBlock *MBB = MF.CreateMachineBasicBlock();
  MF.push_back(MBB);

  // CSE isn't enabled yet -- see createMIRBuilder's doc comment; this
  // translator switches it on once it has a reassociation-sensitive
  // consumer that benefits from it.
  std::unique_ptr<MachineIRBuilder> MIRBuilder =
      createMIRBuilder(MF, /*CSEInfo=*/nullptr);
  MIRBuilder->setMBB(*MBB);

  return GMIRToMIRWalker(*MIRBuilder, *CLI).run(FuncOp, F, FuncInfo);
}
