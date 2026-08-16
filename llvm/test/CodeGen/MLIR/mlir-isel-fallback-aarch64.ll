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
