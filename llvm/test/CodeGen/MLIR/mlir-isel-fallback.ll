; RUN: llc -mtriple=x86_64-unknown-linux-gnu %s -o %t.normal.s
; RUN: llc -mtriple=x86_64-unknown-linux-gnu -enable-mlir-isel %s -o %t.mlir.s
; RUN: diff %t.normal.s %t.mlir.s
;
; M1 scaffolding: the experimental MLIR ISel pass currently always defers
; to the normal instruction selector (see llvm/lib/CodeGen/MLIR/
; MLIRInstructionSelect.cpp), so -enable-mlir-isel must produce byte-identical
; output to a normal llc invocation, in every build configuration:
;  - builds without -DLLVM_ENABLE_MLIR_ISEL=ON degrade immediately
;    (createMLIRInstructionSelectPass() returns nullptr);
;  - builds with it on run the real pass, which always marks the
;    MachineFunction's ISel as failed and falls back via the same
;    ResetMachineFunctionPass + SelectionDAG path GlobalISel uses.

define i32 @add(i32 %a, i32 %b) {
  %sum = add i32 %a, %b
  ret i32 %sum
}
