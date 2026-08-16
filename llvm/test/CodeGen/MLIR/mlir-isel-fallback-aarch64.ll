; REQUIRES: aarch64-registered-target
; RUN: llc -mtriple=aarch64-unknown-linux-gnu -mattr=+sve %s -o %t.normal.s
; RUN: llc -mtriple=aarch64-unknown-linux-gnu -mattr=+sve -enable-mlir-isel %s -o %t.mlir.s
; RUN: diff %t.normal.s %t.mlir.s
;
; +sve is needed for @scalable_alloca/@scalable_array_gep below (harmless
; for every other function in this file, none of which use scalable
; vectors).
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

; See mlir-isel-fallback.ll for the full rationale.
define i128 @add_i128(i128 %a) {
entry:
  %r = add i128 %a, 5
  ret i128 %r
}

; Regression test: AllocaInst::isStaticAlloca() only checks the array-size
; operand and entry-block placement -- it says nothing about whether the
; *allocated type* is scalable, so `alloca <vscale x 4 x i32>` (no
; array-size operand at all) passed isStaticAlloca() cleanly. importAlloca
; would then store only Size.getKnownMinValue() (the vscale=1 size) into
; gmir.alloca's plain I64Attr, silently under-sizing the real (vscale
; times that) stack object on hardware where vscale > 1 -- a silent
; miscompile, not a crash.
define <vscale x 4 x i32> @scalable_alloca() {
entry:
  %p = alloca <vscale x 4 x i32>
  %v = load <vscale x 4 x i32>, ptr %p
  ret <vscale x 4 x i32> %v
}

; Regression test: DataLayout::getTypeAllocSize/
; gep_type_iterator::getSequentialElementStride return a TypeSize, not a
; plain integer; computeGMIRLeafTypes's array-element case and importGEP's
; sequential-index case both used to implicitly convert straight to
; uint64_t, which calls reportFatalInternalError and aborts the whole
; compiler for a scalable element type (unlike every other unsupported
; construct in this file, which falls back gracefully). GEP's own
; top-of-function vector-type guard doesn't catch this case: the GEP's own
; result/pointer-operand types here are plain `ptr`, not vectors -- it's
; the array *element* type that's scalable.
define <vscale x 4 x i32> @scalable_array_gep(ptr %p, i64 %i) {
entry:
  %g = getelementptr [4 x <vscale x 4 x i32>], ptr %p, i64 0, i64 %i
  %v = load <vscale x 4 x i32>, ptr %g
  ret <vscale x 4 x i32> %v
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
