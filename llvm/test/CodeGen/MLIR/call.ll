; REQUIRES: llvm_enable_mlir_isel
; RUN: llc -mtriple=x86_64-unknown-linux-gnu -global-isel %s -o %t.gisel.s
; RUN: llc -mtriple=x86_64-unknown-linux-gnu -enable-mlir-isel %s -o %t.mlir.s
; RUN: diff %t.gisel.s %t.mlir.s
;
; See call-aarch64.ll for the AArch64 variant -- see scalar-arith.ll for
; the rationale on why REQUIRES/the -global-isel oracle need a separate
; file per target rather than extra RUN lines in this one. Unlike other
; genuinely-translated-feature test pairs, this X86 file is deliberately
; narrower than the AArch64 one: it omits a struct-argument call case
; because X86's own -global-isel IRTranslator can't translate a
; struct-by-value call argument at all (confirmed: it report_fatal_errors
; with "unable to translate instruction: call" on a minimal repro,
; independent of gmir) -- there is no working -global-isel oracle to diff
; against on this target, so it isn't testable here. See
; call-aarch64.ll's header for that case (AArch64's own IRTranslator does
; support it).
;
; Verifies direct-call translation (llvm/lib/CodeGen/MLIR/GMIRImporter.cpp's
; importCall, MLIRToGMIRTranslator.cpp's translateCall): a plain
; scalar-args-and-return call, and a void-returning call.

declare i32 @callee(i32, i32)
declare void @callee_void(i32)

define i32 @call_scalar_args(i32 %a, i32 %b) {
entry:
  %r = call i32 @callee(i32 %a, i32 %b)
  ret i32 %r
}

define void @call_void(i32 %a) {
entry:
  call void @callee_void(i32 %a)
  ret void
}
