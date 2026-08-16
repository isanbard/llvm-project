; REQUIRES: llvm_enable_mlir_isel
; REQUIRES: aarch64-registered-target
; RUN: llc -mtriple=aarch64-unknown-linux-gnu -global-isel %s -o %t.gisel.s
; RUN: llc -mtriple=aarch64-unknown-linux-gnu -enable-mlir-isel %s -o %t.mlir.s
; RUN: diff %t.gisel.s %t.mlir.s
;
; AArch64 variant of scalar-arith.ll -- see that file for the full rationale
; (same content, different mtriple; kept as a separate file rather than
; extra RUN lines in the same file because lit's REQUIRES is file-global,
; not scoped to individual RUN lines).

define i32 @add32(i32 %a, i32 %b) {
  %r = add i32 %a, %b
  ret i32 %r
}

define i64 @add64(i64 %a, i64 %b) {
  %r = add i64 %a, %b
  ret i64 %r
}

define i32 @sub(i32 %a, i32 %b) {
  %r = sub i32 %a, %b
  ret i32 %r
}

define i32 @mul(i32 %a, i32 %b) {
  %r = mul i32 %a, %b
  ret i32 %r
}

define i32 @sdiv(i32 %a, i32 %b) {
  %r = sdiv i32 %a, %b
  ret i32 %r
}

define i32 @and(i32 %a, i32 %b) {
  %r = and i32 %a, %b
  ret i32 %r
}

define i32 @or(i32 %a, i32 %b) {
  %r = or i32 %a, %b
  ret i32 %r
}

define i32 @xor(i32 %a, i32 %b) {
  %r = xor i32 %a, %b
  ret i32 %r
}

define i32 @constant(i32 %a) {
  %r = add i32 %a, 42
  ret i32 %r
}

define i32 @multi_op(i32 %a, i32 %b, i32 %c) {
  %t0 = add i32 %a, %b
  %t1 = mul i32 %t0, %c
  %t2 = xor i32 %t1, 7
  %r = sub i32 %t2, %a
  ret i32 %r
}
