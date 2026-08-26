//===- MLIRInstructionSelect.cpp - MLIR ISel pass ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Imports the current MachineFunction's llvm::Function into `gmir` ops
// (GMIRImporter), legalizes them against the target's real legality rules
// (GMIRLegalizer) and, if both succeed, translates them into real generic
// MIR (MLIRToGMIRTranslator). All three steps only ever handle the
// currently-supported subset (straight-line scalar-integer arithmetic);
// anything else makes the importer fail, in which case this pass defers to
// the existing selector exactly as it always has.
//
//===----------------------------------------------------------------------===//

#include "GMIRImporter.h"
#include "GMIRLegalizer.h"
#include "IR/GMIRDialect.h"
#include "MLIRToGMIRTranslator.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/BranchProbabilityInfo.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/CodeGen/GlobalISel/CSEInfo.h"
#include "llvm/CodeGen/GlobalISel/MachineIRBuilder.h"
#include "llvm/CodeGen/GlobalISel/Utils.h"
#include "llvm/CodeGen/LibcallLoweringInfo.h"
#include "llvm/CodeGen/MLIRISel.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/StackProtector.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/Support/CommandLine.h"

using namespace llvm;

// The permanent downstream target Legalizer MachineFunctionPass makes the
// usual -global-isel asm-diff oracle unable to distinguish "GMIRLegalizer's
// own pattern ran" from "the safety-net pass quietly did the work instead"
// for any correctly-scoped legalization case. Dumping the gmir IR right
// after legalize(), before translation, lets a FileCheck test grep for the
// ops (e.g. gmir.uaddo/uadde/merge/unmerge) that only GMIRLegalizer's
// patterns emit -- the real proof mechanism.
static cl::opt<bool> PrintGMIRAfterLegalize(
    "print-gmir-after-legalize", cl::Hidden,
    cl::desc("Print the gmir IR right after GMIRLegalizer runs, before "
             "MLIRToGMIRTranslator lowers it to real MIR"));

namespace {
// Deliberately skips the usual INITIALIZE_PASS/PassRegistry registration:
// that would need llvm/lib/CodeGen/CodeGen.cpp (always compiled into
// LLVMCodeGen) to call an initializeMLIRInstructionSelectPass() that only
// this optional component defines -- the same symbol-resolution problem
// MLIRISel.h's factory-pointer seam exists to avoid. Registration isn't
// required for a pass that's only ever constructed directly and addPass()'d
// (see llvm::Pass's constructor); it's only needed for CLI pass-name lookup
// (e.g. -run-pass=mlir-isel), which isn't needed here. Add it later via a
// runtime PassRegistry::registerPass() call from InitializeMLIRISel() if/
// when that's needed, rather than the static macro.
class MLIRInstructionSelect : public MachineFunctionPass {
public:
  static char ID;

  // MLIRContext/dialect loading and the legalizer's pattern cache are
  // pass-instance state, constructed once and reused across every
  // runOnMachineFunction call (i.e. once per module, not once per
  // function) -- a MachineFunctionPass instance is never shared across
  // compilation threads in any parallel-codegen configuration, so no
  // locking is needed for either. This is also a correctness prerequisite
  // for the pattern cache (see GMIRLegalizer.h's LegalizerPatternCache
  // doc): the cached patterns capture this Context by pointer, so it must
  // outlive every function the cache serves.
  MLIRInstructionSelect() : MachineFunctionPass(ID) {
    Context.getOrLoadDialect<gmir::GMIRDialect>();
    Context.getOrLoadDialect<mlir::func::FuncDialect>();
  }

  StringRef getPassName() const override { return "MLIR Instruction Select"; }

  // Mirrors GlobalISel::IRTranslator::getAnalysisUsage() exactly (down to
  // requirements this pass doesn't itself use yet, like
  // GISelCSEAnalysisWrapperPass and LibcallLoweringInfoWrapper): this pass
  // occupies the same pipeline position (first MachineFunctionPass before
  // Legalize/RegBankSelect/InstructionSelect and the SelectionDAG
  // fallback), and a *partial* match to IRTranslator's declared
  // requirements was empirically insufficient -- the legacy PassManager
  // still failed to keep 'Function Alias Analysis Results' alive for the
  // fallback X86 DAG selector once Legalize/RegBankSelect/
  // InstructionSelect were chained after this pass (see
  // TargetPassConfig::addCoreISelPasses), aborting with "Unable to
  // schedule pass". Matching IRTranslator's requirements fully avoids
  // whatever legacy-PM scheduling decision that partial match was causing.
  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<StackProtector>();
    AU.addRequired<TargetPassConfig>();
    AU.addRequired<GISelCSEAnalysisWrapperPass>();
    AU.addRequired<AssumptionCacheTracker>();
    AU.addRequired<BranchProbabilityInfoWrapperPass>();
    AU.addRequired<AAResultsWrapperPass>();
    AU.addRequired<TargetLibraryInfoWrapperPass>();
    AU.addPreserved<TargetLibraryInfoWrapperPass>();
    AU.addRequired<LibcallLoweringInfoWrapper>();
    getSelectionDAGFallbackAnalysisUsage(AU);
    MachineFunctionPass::getAnalysisUsage(AU);
  }

  bool runOnMachineFunction(MachineFunction &MF) override {
    mlir::OwningOpRef<mlir::ModuleOp> Module(
        mlir::ModuleOp::create(mlir::UnknownLoc::get(&Context)));
    gmir::CallInstMap CallInsts;
    mlir::func::FuncOp FuncOp =
        gmir::importFunction(*Module, MF.getFunction(), CallInsts);

    if (!FuncOp || !gmir::legalize(FuncOp, MF, PatternCache)) {
      // Outside the currently-supported subset: defer to the existing
      // selector, same as always.
      MF.getProperties().setFailedISel();
      return false;
    }

    if (PrintGMIRAfterLegalize) {
      FuncOp.print(llvm::errs());
      llvm::errs() << '\n';
    }

    // Everything below is only needed once import/legalization succeeded
    // -- fetched here, after that check, rather than unconditionally up
    // front, so a function outside the supported subset (the common case
    // for real-world code today) doesn't pay for an unused BPI lookup, a
    // TargetPassConfig analysis fetch, a CSE-config query, and
    // createMIRBuilder's allocation before falling back.
    const auto &BPI = getAnalysis<BranchProbabilityInfoWrapperPass>().getBPI();

    // Match IRTranslatorLegacy::runOnMachineFunction's own choice of
    // builder exactly (see MLIRToGMIRTranslator.h's translate() doc
    // comment for why a plain MachineIRBuilder isn't just a style
    // difference here): CSE is on unconditionally here, the same default
    // IRTranslator itself uses now that TargetPassConfig no longer
    // exposes a CSE-enabled query of its own.
    auto &TPC = getAnalysis<TargetPassConfig>();
    GISelCSEAnalysisWrapper &Wrapper =
        getAnalysis<GISelCSEAnalysisWrapperPass>().getCSEWrapper();
    GISelCSEInfo *CSEInfo = &Wrapper.get(TPC.getCSEConfig());
    std::unique_ptr<MachineIRBuilder> Builder = createMIRBuilder(MF, CSEInfo);

    if (!gmir::translate(FuncOp, MF.getFunction(), MF, BPI, *Builder,
                         CallInsts)) {
      // CallLowering itself declined: defer to the existing selector,
      // same as always.
      MF.getProperties().setFailedISel();
      return false;
    }
    // MF's MIR was just rebuilt from scratch above -- the legacy
    // PassManager only calls removeNotPreservedAnalysis() for this pass
    // when the return value says something changed, so returning false
    // here (as if translate() were a no-op) would leave every analysis
    // this pass doesn't explicitly preserve looking valid to downstream
    // passes even though it now describes a stale, pre-translation MF.
    return true;
  }

private:
  mlir::MLIRContext Context;
  gmir::LegalizerPatternCache PatternCache;
};
} // namespace

char MLIRInstructionSelect::ID = 0;

static MachineFunctionPass *createMLIRInstructionSelectPassImpl() {
  return new MLIRInstructionSelect();
}

void llvm::InitializeMLIRISel() {
  setMLIRInstructionSelectFactory(&createMLIRInstructionSelectPassImpl);
}
