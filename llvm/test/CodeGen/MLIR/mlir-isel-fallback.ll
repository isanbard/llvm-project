; RUN: llc -mtriple=x86_64-unknown-linux-gnu %s -o %t.normal.s
; RUN: llc -mtriple=x86_64-unknown-linux-gnu -enable-mlir-isel %s -o %t.mlir.s
; RUN: diff %t.normal.s %t.mlir.s
;
; Verifies the fallback path for a function genuinely outside M1's
; supported subset (a call -- see llvm/lib/CodeGen/MLIR/GMIRImporter.cpp,
; which only imports single-basic-block, scalar-integer-arithmetic-only
; functions). -enable-mlir-isel must produce byte-identical output to a
; normal llc invocation here in every build configuration:
;  - builds without -DLLVM_ENABLE_MLIR_ISEL=ON degrade immediately
;    (createMLIRInstructionSelectPass() returns nullptr);
;  - builds with it on run the real pass, whose importer declines (this
;    function isn't in the M1 subset), so it marks the MachineFunction's
;    ISel as failed and falls back via the same ResetMachineFunctionPass +
;    SelectionDAG path GlobalISel uses.
;
; (Straight-line scalar arithmetic -- e.g. a trivial `add` -- is now
; genuinely translated instead of always falling back; see scalar-arith.ll.)

declare i32 @callee(i32)

define i32 @calls_something(i32 %a) {
  %r = call i32 @callee(i32 %a)
  ret i32 %r
}
