; REQUIRES: llvm_enable_mlir_isel
; RUN: llc -mtriple=x86_64-unknown-linux-gnu -enable-mlir-isel \
; RUN:     -debug-pass=Executions %s -o /dev/null 2>&1 | FileCheck %s
;
; Regression test for MLIRInstructionSelect::runOnMachineFunction: a
; MachineFunctionPass must return true when it mutates the
; MachineFunction, or the legacy PassManager's removeNotPreservedAnalysis
; -- gated on that return value, not on what the pass actually did -- never
; runs, leaving every analysis this pass doesn't explicitly preserve
; looking valid to later passes even though the MIR it describes was just
; rebuilt from scratch. "Made Modification" is the legacy PassManager's
; own -debug-pass=Executions text for a true return; its presence here is
; a direct, non-fragile check on that return value (see
; PMDataManager::dumpPassInfo in llvm/lib/IR/LegacyPassManager.cpp).
define i32 @add2(i32 %a, i32 %b) {
entry:
  %r = add i32 %a, %b
  ret i32 %r
}

; CHECK: Made Modification 'MLIR Instruction Select' on Function 'add2'
