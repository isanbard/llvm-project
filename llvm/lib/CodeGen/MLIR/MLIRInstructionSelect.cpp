//===- MLIRInstructionSelect.cpp - MLIR ISel pass ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Imports the current MachineFunction's llvm::Function into `gmir` ops
// (GMIRImporter) and, if that succeeds, translates them into real generic
// MIR (MLIRToGMIRTranslator). Both steps only ever handle M1's supported
// subset (straight-line scalar-integer arithmetic); anything else makes
// the importer fail, in which case this pass defers to the existing
// selector exactly as it always has. See
// ~/llvm/mlir_instruction_selection_plan.md for the full architecture.
//
//===----------------------------------------------------------------------===//

#include "GMIRCombiner.h"
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

// M4 slice 1's proof mechanism (see GMIRLegalizer.h): the permanent
// downstream target Legalizer MachineFunctionPass makes the asm-diff
// oracle alone unable to distinguish "GMIRLegalizer's own pattern ran" from
// "the safety-net pass quietly did the work instead" for any correctly-
// scoped legalization slice. Dumping the gmir IR right after legalize(),
// before translation, lets a FileCheck test grep for the ops (e.g.
// gmir.uaddo/uadde/merge/unmerge) that only GMIRLegalizer's patterns emit.
static cl::opt<bool> PrintGMIRAfterLegalize(
    "print-gmir-after-legalize", cl::Hidden,
    cl::desc("Print the gmir IR right after GMIRLegalizer runs, before "
             "MLIRToGMIRTranslator lowers it to real MIR"));

// M5 slice 1's proof mechanism, same rationale as PrintGMIRAfterLegalize
// above: on AArch64 (unlike X86 at -O0), a real downstream GICombiner
// pass runs even at -O0-equivalent (AArch64O0PreLegalizerCombiner), so
// the usual -global-isel asm-diff oracle can't by itself distinguish
// "GMIRCombiner's own fold()/CSE ran" from "the safety-net combiner
// quietly did the work instead" -- see GMIRCombiner.h.
static cl::opt<bool> PrintGMIRAfterCombine(
    "print-gmir-after-combine", cl::Hidden,
    cl::desc("Print the gmir IR right after GMIRCombiner runs, before "
             "MLIRToGMIRTranslator lowers it to real MIR"));

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

  // MLIRContext/dialect loading and the legalizer's and combiner's
  // pattern caches are pass-instance state, constructed once and reused
  // across every runOnMachineFunction call (i.e. once per module, not
  // once per function) -- a MachineFunctionPass instance is never shared
  // across compilation threads in any parallel-codegen configuration, so
  // no locking is needed for any of them. See GMIRLegalizer.h's
  // LegalizerPatternCache doc (GMIRCombiner.h's CombinerPatternCache is
  // the same shape) for why this is also a correctness prerequisite for
  // those caches (the cached patterns capture this Context by pointer).
  MLIRInstructionSelect() : MachineFunctionPass(ID) {
    Context.getOrLoadDialect<gmir::GMIRDialect>();
    Context.getOrLoadDialect<mlir::func::FuncDialect>();
  }

  StringRef getPassName() const override { return "MLIR Instruction Select"; }

  // Mirrors GlobalISel::IRTranslator::getAnalysisUsage() exactly (down to
  // requirements this pass doesn't itself use, like GISelCSEAnalysisWrapperPass
  // and LibcallLoweringInfoWrapper): this pass occupies the same pipeline
  // position (first MachineFunctionPass before Legalize/RegBankSelect/
  // InstructionSelect and the SelectionDAG fallback), and a *partial* match
  // to IRTranslator's declared requirements was empirically insufficient --
  // the legacy PassManager still failed to keep 'Function Alias Analysis
  // Results' alive for the fallback X86 DAG selector once
  // Legalize/RegBankSelect/InstructionSelect were chained after this pass
  // (see TargetPassConfig::addCoreISelPasses), aborting with "Unable to
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
    mlir::func::FuncOp FuncOp =
        gmir::importFunction(*Module, MF.getFunction());

    if (!FuncOp || !gmir::legalize(FuncOp, MF, PatternCache)) {
      // Outside the supported subset: defer to the existing selector, same
      // as always.
      MF.getProperties().setFailedISel();
      return false;
    }

    if (PrintGMIRAfterLegalize) {
      FuncOp.print(llvm::errs());
      llvm::errs() << '\n';
    }

    if (!gmir::combine(FuncOp, CombinerCache)) {
      MF.getProperties().setFailedISel();
      return false;
    }

    if (PrintGMIRAfterCombine) {
      FuncOp.print(llvm::errs());
      llvm::errs() << '\n';
    }

    // Everything below is only needed once import/legalization/combining
    // succeeded -- fetched here, after that check, rather than
    // unconditionally up front, so a function outside the supported
    // subset (the common case for real-world code today) doesn't pay for
    // an unused BPI lookup, a TargetPassConfig analysis fetch, a
    // CSE-config query, and createMIRBuilder's allocation before falling
    // back.
    const auto &BPI = getAnalysis<BranchProbabilityInfoWrapperPass>().getBPI();

    // Match IRTranslator::translate's own choice of builder exactly (see
    // MLIRToGMIRTranslator.h's translate() doc comment for why a plain
    // MachineIRBuilder isn't just a style difference here): downstream
    // combiner passes reused unchanged from the real pipeline have
    // reassociation rules that measurably behave differently on
    // non-CSE'd input, e.g. a multi-index GEP's constant-offset
    // G_PTR_ADD chain selected a different (but semantically equivalent)
    // AArch64 addressing mode without this. createMIRBuilder is shared
    // with IRTranslator itself (Utils.h) for exactly this "CSEMIRBuilder
    // wired to CSEInfo, or plain MachineIRBuilder" mechanics.
    auto &TPC = getAnalysis<TargetPassConfig>();
    GISelCSEInfo *CSEInfo = nullptr;
    if (TPC.isGISelCSEEnabled())
      CSEInfo = &getAnalysis<GISelCSEAnalysisWrapperPass>().getCSEWrapper().get(
          TPC.getCSEConfig());

    std::unique_ptr<MachineIRBuilder> Builder = createMIRBuilder(MF, CSEInfo);

    if (!gmir::translate(FuncOp, MF.getFunction(), MF, BPI, *Builder)) {
      // CallLowering itself declined: defer to the existing selector, same
      // as always.
      MF.getProperties().setFailedISel();
      return false;
    }

    return false;
  }

private:
  mlir::MLIRContext Context;
  gmir::LegalizerPatternCache PatternCache;
  gmir::CombinerPatternCache CombinerCache;
};
} // namespace

char MLIRInstructionSelect::ID = 0;

static MachineFunctionPass *createMLIRInstructionSelectPassImpl() {
  return new MLIRInstructionSelect();
}

void llvm::InitializeMLIRISel() {
  setMLIRInstructionSelectFactory(&createMLIRInstructionSelectPassImpl);
}
