; REQUIRES: llvm_enable_mlir_isel
; REQUIRES: aarch64-registered-target
; RUN: llc -mtriple=aarch64-unknown-linux-gnu -global-isel %s -o %t.gisel.s
; RUN: llc -mtriple=aarch64-unknown-linux-gnu -enable-mlir-isel %s -o %t.mlir.s
; RUN: diff %t.gisel.s %t.mlir.s
;
; AArch64 variant of combine-identities.ll -- see that file for the full
; rationale (same content, different mtriple; kept as a separate file
; rather than extra RUN lines in the same file since lit's REQUIRES is
; file-global, not scoped to individual RUN lines).
;
; RUN: llc -mtriple=aarch64-unknown-linux-gnu -enable-mlir-isel \
; RUN:     -print-gmir-after-combine %s -o /dev/null 2>&1 | FileCheck %s
;
; Unlike X86_64, the byte-diff above is NOT by itself trustworthy proof
; that GMIRCombiner's own fold()s did this work: AArch64PassConfig runs
; a real GICombiner pass (AArch64O0PreLegalizerCombiner) even at
; -O0-equivalent, gated on isGlobalISelOptNone() -- the same check
; design doc §1.7 already found behaves differently under
; -enable-mlir-isel than under -global-isel. So the real downstream
; combiner could in principle be doing these simplifications "for
; free," the same masking risk M4 hit with the real target Legalizer
; (see narrow-scalar-i686.ll). The -print-gmir-after-combine FileCheck
; below is the actual proof here -- same CHECK patterns as
; combine-identities.ll.

define i32 @add_zero(i32 %x) {
  %r = add i32 %x, 0
  ret i32 %r
}
; CHECK-LABEL: func.func @add_zero
; CHECK-NOT: gmir.add
; CHECK: return %arg0

define i32 @add_zero_lhs(i32 %x) {
  %r = add i32 0, %x
  ret i32 %r
}
; CHECK-LABEL: func.func @add_zero_lhs
; CHECK-NOT: gmir.add
; CHECK: return %arg0

define i32 @sub_self(i32 %x) {
  %r = sub i32 %x, %x
  ret i32 %r
}
; CHECK-LABEL: func.func @sub_self
; CHECK-NOT: gmir.sub
; CHECK: gmir.constant 0
; CHECK: return

define i32 @mul_zero(i32 %x) {
  %r = mul i32 %x, 0
  ret i32 %r
}
; CHECK-LABEL: func.func @mul_zero
; CHECK-NOT: gmir.mul
; CHECK: gmir.constant 0
; CHECK: return

define i32 @mul_one(i32 %x) {
  %r = mul i32 1, %x
  ret i32 %r
}
; CHECK-LABEL: func.func @mul_one
; CHECK-NOT: gmir.mul
; CHECK: return %arg0

define i32 @and_self(i32 %x) {
  %r = and i32 %x, %x
  ret i32 %r
}
; CHECK-LABEL: func.func @and_self
; CHECK-NOT: gmir.and
; CHECK: return %arg0

define i32 @and_allones(i32 %x) {
  %r = and i32 %x, -1
  ret i32 %r
}
; CHECK-LABEL: func.func @and_allones
; CHECK-NOT: gmir.and
; CHECK: return %arg0

define i32 @and_zero(i32 %x) {
  %r = and i32 %x, 0
  ret i32 %r
}
; CHECK-LABEL: func.func @and_zero
; CHECK-NOT: gmir.and
; CHECK: gmir.constant 0
; CHECK: return

define i32 @or_self(i32 %x) {
  %r = or i32 %x, %x
  ret i32 %r
}
; CHECK-LABEL: func.func @or_self
; CHECK-NOT: gmir.or
; CHECK: return %arg0

define i32 @or_zero(i32 %x) {
  %r = or i32 %x, 0
  ret i32 %r
}
; CHECK-LABEL: func.func @or_zero
; CHECK-NOT: gmir.or
; CHECK: return %arg0

define i32 @or_allones(i32 %x) {
  %r = or i32 %x, -1
  ret i32 %r
}
; CHECK-LABEL: func.func @or_allones
; CHECK-NOT: gmir.or
; CHECK: gmir.constant -1
; CHECK: return

define i32 @xor_self(i32 %x) {
  %r = xor i32 %x, %x
  ret i32 %r
}
; CHECK-LABEL: func.func @xor_self
; CHECK-NOT: gmir.xor
; CHECK: gmir.constant 0
; CHECK: return

define i32 @xor_zero(i32 %x) {
  %r = xor i32 %x, 0
  ret i32 %r
}
; CHECK-LABEL: func.func @xor_zero
; CHECK-NOT: gmir.xor
; CHECK: return %arg0

define i32 @const_add(i32 %unused) {
  %r = add i32 3, 4
  ret i32 %r
}
; CHECK-LABEL: func.func @const_add
; CHECK-NOT: gmir.add
; CHECK: gmir.constant 7
; CHECK: return

define i32 @const_add_overflow(i32 %unused) {
  %r = add i32 2147483647, 1
  ret i32 %r
}
; CHECK-LABEL: func.func @const_add_overflow
; CHECK-NOT: gmir.add
; CHECK: gmir.constant -2147483648
; CHECK: return

; M5 slice 2's first real OpRewritePattern (MulNegOneToSubPattern, not
; a fold()) -- see combine-identities.ll for the full rationale. Unlike
; that file's mul_neg_one, this one's byte-diff RUN lines above DO hold
; here: confirmed empirically that AArch64's real O0-equivalent
; combiner (AArch64O0PreLegalizerCombiner) already does this exact
; mul-by-negative-one-to-negate simplification independently, same
; masking risk as every other identity in this file -- the
; -print-gmir-after-combine FileCheck below is still the real proof
; gmir's own pattern ran, not the byte-diff alone.
define i32 @mul_neg_one(i32 %x) {
  %r = mul i32 %x, -1
  ret i32 %r
}
; CHECK-LABEL: func.func @mul_neg_one
; CHECK-NOT: gmir.mul
; CHECK: gmir.constant 0
; CHECK: gmir.sub
; CHECK: return

define i32 @mul_neg_one_lhs(i32 %x) {
  %r = mul i32 -1, %x
  ret i32 %r
}
; CHECK-LABEL: func.func @mul_neg_one_lhs
; CHECK-NOT: gmir.mul
; CHECK: gmir.constant 0
; CHECK: gmir.sub
; CHECK: return
