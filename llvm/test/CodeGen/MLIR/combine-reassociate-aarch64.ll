; REQUIRES: llvm_enable_mlir_isel
; REQUIRES: aarch64-registered-target
; RUN: llc -mtriple=aarch64-unknown-linux-gnu -enable-mlir-isel \
; RUN:     -print-gmir-after-combine %s -o /dev/null 2>&1 | FileCheck --check-prefix=GMIR %s
; RUN: llc -mtriple=aarch64-unknown-linux-gnu -enable-mlir-isel %s -o - | FileCheck --check-prefix=ASM %s
;
; AArch64 variant of combine-identities.ll's M5 slice 4 additions (see
; that file for the full rationale on the ported rules themselves).
; Unlike combine-identities-aarch64.ll, this is a separate dedicated
; file rather than more RUN lines there: confirmed empirically (`diff`
; against plain -global-isel) that 6 of these 8 functions' mlir-isel
; output genuinely diverges from plain -global-isel on AArch64 --
; AArch64O0PreLegalizerCombiner does not do constant reassociation for
; mul/or/xor, nor the repeated-operand/self-cancel identities,
; independently (it does happen to do it for add/and, but splitting
; those two out to stay byte-diffable while the other six sit elsewhere
; would fragment one coherent set of tests for no real benefit) -- a
; file-wide byte-diff RUN line would fail here the same way it would
; have for combine-disjoint-add-aarch64.ll's add_disjoint_and. Uses the
; same GMIR + ASM FileCheck pair for the same reason.

define i32 @add_reassoc_const(i32 %x) {
  %a = add i32 %x, 3
  %r = add i32 %a, 4
  ret i32 %r
}
; GMIR-LABEL: func.func @add_reassoc_const
; GMIR: gmir.constant 7
; GMIR: gmir.add
; GMIR-NOT: gmir.add
; GMIR: return
; ASM-LABEL: add_reassoc_const:
; ASM: add w0, w0, #7

define i32 @mul_reassoc_const(i32 %x) {
  %a = mul i32 %x, 3
  %r = mul i32 %a, 4
  ret i32 %r
}
; GMIR-LABEL: func.func @mul_reassoc_const
; GMIR: gmir.constant 12
; GMIR: gmir.mul
; GMIR-NOT: gmir.mul
; GMIR: return
; ASM-LABEL: mul_reassoc_const:
; ASM: mul w0, w0, w8

define i32 @and_reassoc_const(i32 %x) {
  %a = and i32 %x, 240
  %r = and i32 %a, 255
  ret i32 %r
}
; GMIR-LABEL: func.func @and_reassoc_const
; GMIR: gmir.constant 240
; GMIR: gmir.and
; GMIR-NOT: gmir.and
; GMIR: return
; ASM-LABEL: and_reassoc_const:
; ASM: and w0, w0, #0xf0

define i32 @or_reassoc_const(i32 %x) {
  %a = or i32 %x, 1
  %r = or i32 %a, 2
  ret i32 %r
}
; GMIR-LABEL: func.func @or_reassoc_const
; GMIR: gmir.constant 3
; GMIR: gmir.or
; GMIR-NOT: gmir.or
; GMIR: return
; ASM-LABEL: or_reassoc_const:
; ASM: orr w0, w0, #0x3

define i32 @xor_reassoc_const(i32 %x) {
  %a = xor i32 %x, 1
  %r = xor i32 %a, 3
  ret i32 %r
}
; GMIR-LABEL: func.func @xor_reassoc_const
; GMIR: gmir.constant 2
; GMIR: gmir.xor
; GMIR-NOT: gmir.xor
; GMIR: return
; ASM-LABEL: xor_reassoc_const:
; ASM: eor w0, w0, #0x2

define i32 @and_repeated_operand(i32 %a, i32 %b) {
  %m = and i32 %a, %b
  %r = and i32 %m, %a
  ret i32 %r
}
; GMIR-LABEL: func.func @and_repeated_operand
; GMIR: gmir.and %arg0, %arg1
; GMIR-NOT: gmir.and
; GMIR: return
; ASM-LABEL: and_repeated_operand:
; ASM: and w0, w0, w1

define i32 @or_repeated_operand(i32 %a, i32 %b) {
  %m = or i32 %a, %b
  %r = or i32 %m, %b
  ret i32 %r
}
; GMIR-LABEL: func.func @or_repeated_operand
; GMIR: gmir.or %arg0, %arg1
; GMIR-NOT: gmir.or
; GMIR: return
; ASM-LABEL: or_repeated_operand:
; ASM: orr w0, w0, w1

define i32 @xor_self_cancel(i32 %a, i32 %b) {
  %m = xor i32 %a, %b
  %r = xor i32 %m, %a
  ret i32 %r
}
; GMIR-LABEL: func.func @xor_self_cancel
; GMIR-NOT: gmir.xor
; GMIR: return %arg1
; ASM-LABEL: xor_self_cancel:
; ASM-NOT: eor
; ASM: mov w0, w1

; Regression case for a real bug an expert review found: without
; XorOp::fold's x^x->0 identity (added alongside this test, see
; combine-identities.ll), reducing %n to %b here (via
; XorSelfCancelPattern above) rewires %r's operands to (b, b) -- a live
; x^x that nothing folded away. With the fix, the whole chain collapses
; to a constant. Confirmed empirically that this genuinely diverges
; from plain -global-isel on AArch64 too (its combiner leaves a real
; `eor w0, w8, w8` rather than folding to zero), same reason the rest
; of this file avoids the byte-diff oracle.
define i32 @xor_chain_self_cancel_then_zero(i32 %a, i32 %b) {
  %m = xor i32 %a, %b
  %n = xor i32 %m, %a
  %r = xor i32 %n, %b
  ret i32 %r
}
; GMIR-LABEL: func.func @xor_chain_self_cancel_then_zero
; GMIR-NOT: gmir.xor
; GMIR: gmir.constant 0
; GMIR: return
; ASM-LABEL: xor_chain_self_cancel_then_zero:
; ASM: mov w0, wzr
