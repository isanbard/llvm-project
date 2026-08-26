; REQUIRES: aarch64-registered-target
; RUN: llc -mtriple=aarch64-unknown-linux-gnu %s -o %t.normal.s
; RUN: llc -mtriple=aarch64-unknown-linux-gnu -enable-mlir-isel %s -o %t.mlir.s
; RUN: diff %t.normal.s %t.mlir.s
;
; AArch64 variant of mlir-isel-fallback.ll -- see that file for the full
; rationale (same content, different mtriple; kept as a separate file
; rather than extra RUN lines in the same file for consistency with
; scalar-arith-aarch64.ll, even though this particular test doesn't
; currently need a file-global feature requirement beyond the target).

declare i32 @callee(i32)

define i32 @calls_something(i32 %a) {
  %r = call i32 @callee(i32 %a)
  ret i32 %r
}

; See mlir-isel-fallback.ll for the full rationale.
define i128 @add_i128(i128 %a) {
entry:
  %r = add i128 %a, 5
  ret i128 %r
}
