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

define i32 @dynamic_alloca(i32 %n) {
entry:
  %p = alloca i32, i32 %n
  store i32 0, ptr %p
  %v = load i32, ptr %p
  ret i32 %v
}

define i8 @huge_alloca() {
entry:
  %p = alloca i8, i64 9223372036854775808
  store i8 0, ptr %p
  %v = load i8, ptr %p
  ret i8 %v
}
