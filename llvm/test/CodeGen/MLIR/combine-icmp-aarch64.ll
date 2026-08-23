; REQUIRES: llvm_enable_mlir_isel
; REQUIRES: aarch64-registered-target
; RUN: llc -mtriple=aarch64-unknown-linux-gnu -enable-mlir-isel \
; RUN:     -print-gmir-after-combine %s -o /dev/null 2>&1 | FileCheck --check-prefix=GMIR %s
; RUN: llc -mtriple=aarch64-unknown-linux-gnu -enable-mlir-isel %s -o - | FileCheck --check-prefix=ASM %s
;
; AArch64 variant of combine-identities.ll's M5 slice 6 additions (gmir's
; first gmir.icmp combiner coverage -- see that file for the full
; rationale on the ported rules themselves). A separate dedicated file
; rather than more RUN lines in combine-identities-aarch64.ll: confirmed
; empirically (`diff` against plain -global-isel) that ICmpOp::fold's
; reflexive icmp-cc-X,X identity, plus ICmpSameBinOpPattern's
; (X op Y)==(X op Z)->Y==Z family, genuinely diverge on AArch64 -- plain
; -global-isel doesn't fold a self-comparison at all here (it still
; emits `cmp w0, w0; cset w0, cc`), nor does it cancel the shared operand
; out of two same-shaped binops feeding an icmp, unlike AArch64's own
; downstream GICombiner, which apparently doesn't wire up
; CombinerHelper::matchICmpToTrueFalseKnownBits (or an equivalent) at O0
; either. ICmpBinOpEqOtherPattern's (X op Y)==X/Y->Y/X==0 family (and the
; sub-Y==N1 negative test) turn out byte-diff clean individually, but
; stay here too rather than being split into combine-identities-aarch64.ll
; -- same "don't fragment one coherent test set" call as
; combine-reassociate-aarch64.ll. ICmpConstAdjustPattern/
; ICmpSubConstPattern's three positive cases diverge the same way (a
; genuine gmir-only optimization); their hasOneUse()-negative and
; wrong-side tests are byte-diff clean but stay here for the same reason.

define i1 @icmp_eq_self(i32 %x) {
  %r = icmp eq i32 %x, %x
  ret i1 %r
}
; GMIR-LABEL: func.func @icmp_eq_self
; GMIR-NOT: gmir.icmp
; GMIR: gmir.constant -1
; GMIR: return
; ASM-LABEL: icmp_eq_self:
; ASM-NOT: cmp w
; ASM: mov w8, #1
; ASM: and w0, w8, #0x1

define i1 @icmp_ne_self(i32 %x) {
  %r = icmp ne i32 %x, %x
  ret i1 %r
}
; GMIR-LABEL: func.func @icmp_ne_self
; GMIR-NOT: gmir.icmp
; GMIR: gmir.constant 0
; GMIR: return
; ASM-LABEL: icmp_ne_self:
; ASM-NOT: cmp w
; ASM: and w0, wzr, #0x1

define i1 @icmp_slt_self(i32 %x) {
  %r = icmp slt i32 %x, %x
  ret i1 %r
}
; GMIR-LABEL: func.func @icmp_slt_self
; GMIR-NOT: gmir.icmp
; GMIR: gmir.constant 0
; GMIR: return
; ASM-LABEL: icmp_slt_self:
; ASM-NOT: cmp w
; ASM: and w0, wzr, #0x1

define i1 @icmp_sle_self(i32 %x) {
  %r = icmp sle i32 %x, %x
  ret i1 %r
}
; GMIR-LABEL: func.func @icmp_sle_self
; GMIR-NOT: gmir.icmp
; GMIR: gmir.constant -1
; GMIR: return
; ASM-LABEL: icmp_sle_self:
; ASM-NOT: cmp w
; ASM: mov w8, #1
; ASM: and w0, w8, #0x1

; ICmpSameBinOpPattern: (X op Y) == (X op Z) -> Y == Z. See
; combine-identities.ll for the full rationale.
define i1 @icmp_same_add(i32 %x, i32 %y, i32 %z) {
  %a = add i32 %x, %y
  %b = add i32 %x, %z
  %r = icmp eq i32 %a, %b
  ret i1 %r
}
; GMIR-LABEL: func.func @icmp_same_add
; GMIR-NOT: gmir.add
; GMIR: gmir.icmp"(%arg1, %arg2)
; GMIR: return
; ASM-LABEL: icmp_same_add:
; ASM-NOT: add w
; ASM: cmp w1, w2
; ASM: cset w0, eq

define i1 @icmp_same_add_swapped(i32 %x, i32 %y, i32 %z) {
  %a = add i32 %x, %y
  %b = add i32 %z, %x
  %r = icmp ne i32 %a, %b
  ret i1 %r
}
; GMIR-LABEL: func.func @icmp_same_add_swapped
; GMIR-NOT: gmir.add
; GMIR: gmir.icmp"(%arg1, %arg2)
; GMIR: return
; ASM-LABEL: icmp_same_add_swapped:
; ASM-NOT: add w
; ASM: cmp w1, w2
; ASM: cset w0, ne

define i1 @icmp_same_sub(i32 %x, i32 %y, i32 %z) {
  %a = sub i32 %x, %y
  %b = sub i32 %x, %z
  %r = icmp eq i32 %a, %b
  ret i1 %r
}
; GMIR-LABEL: func.func @icmp_same_sub
; GMIR-NOT: gmir.sub
; GMIR: gmir.icmp"(%arg1, %arg2)
; GMIR: return
; ASM-LABEL: icmp_same_sub:
; ASM-NOT: sub w
; ASM: cmp w1, w2
; ASM: cset w0, eq

define i1 @icmp_same_xor(i32 %x, i32 %y, i32 %z) {
  %a = xor i32 %x, %y
  %b = xor i32 %x, %z
  %r = icmp eq i32 %a, %b
  ret i1 %r
}
; GMIR-LABEL: func.func @icmp_same_xor
; GMIR-NOT: gmir.xor
; GMIR: gmir.icmp"(%arg1, %arg2)
; GMIR: return
; ASM-LABEL: icmp_same_xor:
; ASM-NOT: eor
; ASM: cmp w1, w2
; ASM: cset w0, eq

; ICmpBinOpEqOtherPattern: (X op Y)==X -> Y==0, (X op Y)==Y -> X==0 (add/
; xor only). See combine-identities.ll for the full rationale.
define i1 @icmp_addy_eqx(i32 %x, i32 %y) {
  %a = add i32 %x, %y
  %r = icmp eq i32 %a, %x
  ret i1 %r
}
; GMIR-LABEL: func.func @icmp_addy_eqx
; GMIR-NOT: gmir.add
; GMIR: gmir.constant 0
; GMIR: gmir.icmp"(%arg1
; GMIR: return
; ASM-LABEL: icmp_addy_eqx:
; ASM-NOT: add w
; ASM: cmp w1, #0
; ASM: cset w0, eq

define i1 @icmp_addy_eqy(i32 %x, i32 %y) {
  %a = add i32 %x, %y
  %r = icmp eq i32 %a, %y
  ret i1 %r
}
; GMIR-LABEL: func.func @icmp_addy_eqy
; GMIR-NOT: gmir.add
; GMIR: gmir.constant 0
; GMIR: gmir.icmp"(%arg0
; GMIR: return
; ASM-LABEL: icmp_addy_eqy:
; ASM-NOT: add w
; ASM: cmp w0, #0
; ASM: cset w0, eq

define i1 @icmp_subxy_eqx(i32 %x, i32 %y) {
  %a = sub i32 %x, %y
  %r = icmp eq i32 %a, %x
  ret i1 %r
}
; GMIR-LABEL: func.func @icmp_subxy_eqx
; GMIR-NOT: gmir.sub
; GMIR: gmir.constant 0
; GMIR: gmir.icmp"(%arg1
; GMIR: return
; ASM-LABEL: icmp_subxy_eqx:
; ASM-NOT: sub w
; ASM: cmp w1, #0
; ASM: cset w0, eq

; Negative test: sub's Y==N1 form is deliberately not implemented (would
; need a shift op) -- both the sub and the icmp must survive unchanged.
define i1 @icmp_subxy_eqy(i32 %x, i32 %y) {
  %a = sub i32 %x, %y
  %r = icmp eq i32 %a, %y
  ret i1 %r
}
; GMIR-LABEL: func.func @icmp_subxy_eqy
; GMIR: gmir.sub
; GMIR: gmir.icmp
; GMIR: return
; ASM-LABEL: icmp_subxy_eqy:
; ASM: sub w8, w0, w1
; ASM: cmp w8, w1

define i1 @icmp_xor_eqx(i32 %x, i32 %y) {
  %a = xor i32 %x, %y
  %r = icmp eq i32 %a, %x
  ret i1 %r
}
; GMIR-LABEL: func.func @icmp_xor_eqx
; GMIR-NOT: gmir.xor
; GMIR: gmir.constant 0
; GMIR: gmir.icmp"(%arg1
; GMIR: return
; ASM-LABEL: icmp_xor_eqx:
; ASM-NOT: eor
; ASM: cmp w1, #0
; ASM: cset w0, eq

define i1 @icmp_xor_eqy(i32 %x, i32 %y) {
  %a = xor i32 %x, %y
  %r = icmp eq i32 %a, %y
  ret i1 %r
}
; GMIR-LABEL: func.func @icmp_xor_eqy
; GMIR-NOT: gmir.xor
; GMIR: gmir.constant 0
; GMIR: gmir.icmp"(%arg0
; GMIR: return
; ASM-LABEL: icmp_xor_eqy:
; ASM-NOT: eor
; ASM: cmp w0, #0
; ASM: cset w0, eq

; ICmpConstAdjustPattern/ICmpSubConstPattern: (X op C1)==C2 ->
; X==combine(C1,C2). See combine-identities.ll for the full rationale.
define i1 @icmp_add_const_adjust(i32 %x) {
  %a = add i32 %x, 5
  %r = icmp eq i32 %a, 10
  ret i1 %r
}
; GMIR-LABEL: func.func @icmp_add_const_adjust
; GMIR-NOT: gmir.add
; GMIR: gmir.constant 5
; GMIR: gmir.icmp"(%arg0
; GMIR: return
; ASM-LABEL: icmp_add_const_adjust:
; ASM-NOT: add w
; ASM: cmp w0, #5
; ASM: cset w0, eq

define i1 @icmp_xor_const_adjust(i32 %x) {
  %a = xor i32 %x, 5
  %r = icmp eq i32 %a, 10
  ret i1 %r
}
; GMIR-LABEL: func.func @icmp_xor_const_adjust
; GMIR-NOT: gmir.xor
; GMIR: gmir.constant 15
; GMIR: gmir.icmp"(%arg0
; GMIR: return
; ASM-LABEL: icmp_xor_const_adjust:
; ASM-NOT: eor
; ASM: cmp w0, #15
; ASM: cset w0, eq

define i1 @icmp_sub_const_adjust(i32 %x) {
  %a = sub i32 20, %x
  %r = icmp eq i32 %a, 10
  ret i1 %r
}
; GMIR-LABEL: func.func @icmp_sub_const_adjust
; GMIR-NOT: gmir.sub
; GMIR: gmir.constant 10
; GMIR: gmir.icmp"(%arg0
; GMIR: return
; ASM-LABEL: icmp_sub_const_adjust:
; ASM: cmp w0, #10
; ASM: cset w0, eq

; Negative tests: %a is also stored, so hasOneUse() must block the
; rewrite; these three are byte-diff clean.
define i1 @icmp_add_const_multiuse(i32 %x, ptr %out) {
  %a = add i32 %x, 5
  store i32 %a, ptr %out
  %r = icmp eq i32 %a, 10
  ret i1 %r
}
; GMIR-LABEL: func.func @icmp_add_const_multiuse
; GMIR: gmir.add
; GMIR: gmir.icmp
; GMIR: return
; ASM-LABEL: icmp_add_const_multiuse:
; ASM: add w8, w0, #5
; ASM: cmp w8, #10

define i1 @icmp_xor_const_multiuse(i32 %x, ptr %out) {
  %a = xor i32 %x, 5
  store i32 %a, ptr %out
  %r = icmp eq i32 %a, 10
  ret i1 %r
}
; GMIR-LABEL: func.func @icmp_xor_const_multiuse
; GMIR: gmir.xor
; GMIR: gmir.icmp
; GMIR: return
; ASM-LABEL: icmp_xor_const_multiuse:
; ASM: eor
; ASM: cmp w8, #10

define i1 @icmp_sub_const_multiuse(i32 %x, ptr %out) {
  %a = sub i32 20, %x
  store i32 %a, ptr %out
  %r = icmp eq i32 %a, 10
  ret i1 %r
}
; GMIR-LABEL: func.func @icmp_sub_const_multiuse
; GMIR: gmir.sub
; GMIR: gmir.icmp
; GMIR: return
; ASM-LABEL: icmp_sub_const_multiuse:
; ASM: cmp w8, #10

; Wrong-side test: sub(X, C1) must NOT trigger ICmpSubConstPattern.
define i1 @icmp_sub_wrong_side(i32 %x) {
  %a = sub i32 %x, 20
  %r = icmp eq i32 %a, 10
  ret i1 %r
}
; GMIR-LABEL: func.func @icmp_sub_wrong_side
; GMIR: gmir.sub
; GMIR: gmir.icmp
; GMIR: return
; ASM-LABEL: icmp_sub_wrong_side:
; ASM: cmp w8, #10
