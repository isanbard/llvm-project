; RUN: llc -mtriple=x86_64-unknown-linux-gnu %s -o %t.normal.s
; RUN: llc -mtriple=x86_64-unknown-linux-gnu -enable-mlir-isel %s -o %t.mlir.s
; RUN: diff %t.normal.s %t.mlir.s
;
; Verifies the fallback path for functions genuinely outside the supported
; subset (a call, and -- since M2 -- a switch; see
; llvm/lib/CodeGen/MLIR/GMIRImporter.cpp, whose per-instruction dispatch
; has no case for either). -enable-mlir-isel must produce byte-identical
; output to a normal llc invocation here in every build configuration:
;  - builds without -DLLVM_ENABLE_MLIR_ISEL=ON degrade immediately
;    (createMLIRInstructionSelectPass() returns nullptr);
;  - builds with it on run the real pass, whose importer declines (neither
;    function is in the supported subset), so it marks the
;    MachineFunction's ISel as failed and falls back via the same
;    ResetMachineFunctionPass + SelectionDAG path GlobalISel uses.
;
; (Straight-line scalar arithmetic and structured if/else/loop control
; flow are now genuinely translated instead of always falling back; see
; scalar-arith.ll and control-flow.ll.)

declare i32 @callee(i32)

define i32 @calls_something(i32 %a) {
  %r = call i32 @callee(i32 %a)
  ret i32 %r
}

define i32 @switcher(i32 %x) {
entry:
  switch i32 %x, label %default [
    i32 0, label %case0
    i32 1, label %case1
  ]
case0:
  ret i32 10
case1:
  ret i32 20
default:
  ret i32 30
}
