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

define <4 x ptr> @vector_gep(<4 x ptr> %p, i64 %i) {
entry:
  %g = getelementptr i32, <4 x ptr> %p, i64 %i
  ret <4 x ptr> %g
}

define i32 @narrow_index_gep(ptr %p, i32 %i) {
entry:
  %g = getelementptr i32, ptr %p, i32 %i
  %v = load i32, ptr %g
  ret i32 %v
}

define void @load_struct_with_float(ptr %src, ptr %dst) {
entry:
  %s = load {i32, float}, ptr %src
  store {i32, float} %s, ptr %dst
  ret void
}

define i32 @indirect_call(ptr %fp, i32 %a) {
entry:
  %r = call i32 %fp(i32 %a)
  ret i32 %r
}

declare i32 @vararg_callee(i32, ...)

define i32 @vararg_call(i32 %a, i32 %b) {
entry:
  %r = call i32 (i32, ...) @vararg_callee(i32 %a, i32 %b)
  ret i32 %r
}
