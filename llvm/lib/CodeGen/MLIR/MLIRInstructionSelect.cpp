//===- MLIRInstructionSelect.cpp - MLIR ISel pass ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the M1-scaffolding version of the MLIR ISel pass: it registers
// and loads the `gmir` MLIR dialect (proving the dialect + MLIRContext
// wiring builds and runs), then unconditionally defers to the existing
// selector by marking the MachineFunction's ISel as failed. Real op
// translation is deferred to a later iteration -- see
// ~/llvm/mlir_instruction_selection_plan.md, "Deferred to next iteration".
//
//===----------------------------------------------------------------------===//

#include "IR/GMIRDialect.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/CodeGen/GlobalISel/Utils.h"
#include "llvm/CodeGen/MLIRISel.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "mlir/IR/MLIRContext.h"

using namespace llvm;

namespace {
// Deliberately skips the usual INITIALIZE_PASS/PassRegistry registration:
// that would need llvm/lib/CodeGen/CodeGen.cpp (always compiled into
// LLVMCodeGen) to call an initializeMLIRInstructionSelectPass() that only
// this optional component defines -- the same symbol-resolution problem
// MLIRISel.h's factory-pointer seam exists to avoid. Registration isn't
// required for a pass that's only ever constructed directly and addPass()'d
// (see llvm::Pass's constructor); it's only needed for CLI pass-name lookup
// (e.g. -run-pass=mlir-isel), which isn't a scaffolding requirement. Add it
// later via a runtime PassRegistry::registerPass() call from
// InitializeMLIRISel() if/when that's needed, rather than the static macro.
class MLIRInstructionSelect : public MachineFunctionPass {
public:
  static char ID;

  MLIRInstructionSelect() : MachineFunctionPass(ID) {}

  StringRef getPassName() const override { return "MLIR Instruction Select"; }

  // Mirrors GlobalISel::IRTranslator::getAnalysisUsage(): this pass occupies
  // the same pipeline position (first MachineFunctionPass before the
  // SelectionDAG fallback), and without declaring the same requirements the
  // legacy PassManager fails to keep 'Function Alias Analysis Results' alive
  // for the fallback X86 DAG selector, aborting with "Unable to schedule
  // pass" -- getSelectionDAGFallbackAnalysisUsage() is the fix GlobalISel's
  // own passes use for exactly this fallback-path scheduling issue.
  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<TargetPassConfig>();
    AU.addRequired<AAResultsWrapperPass>();
    getSelectionDAGFallbackAnalysisUsage(AU);
    MachineFunctionPass::getAnalysisUsage(AU);
  }

  bool runOnMachineFunction(MachineFunction &MF) override {
    // Proves the dialect registers and loads cleanly; nothing is
    // translated yet, so every function currently falls back.
    mlir::MLIRContext Context;
    Context.getOrLoadDialect<gmir::GMIRDialect>();

    MF.getProperties().setFailedISel();
    return false;
  }
};
} // namespace

char MLIRInstructionSelect::ID = 0;

static MachineFunctionPass *createMLIRInstructionSelectPassImpl() {
  return new MLIRInstructionSelect();
}

void llvm::InitializeMLIRISel() {
  setMLIRInstructionSelectFactory(&createMLIRInstructionSelectPassImpl);
}
