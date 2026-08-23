; REQUIRES: llvm_enable_mlir_isel
; RUN: llc -mtriple=x86_64-unknown-linux-gnu -enable-mlir-isel \
; RUN:     -print-gmir-after-combine %s -o /dev/null 2>&1 | FileCheck --check-prefix=GMIR %s
; RUN: llc -mtriple=x86_64-unknown-linux-gnu -enable-mlir-isel %s -o - | FileCheck --check-prefix=ASM %s
;
; See combine-identities-aarch64.ll for the AArch64 variant. Unlike
; every prior M1-M4 test in this directory, this one does NOT use the
; usual -global-isel byte-diff oracle -- confirmed empirically (not
; assumed) that it's the wrong invariant here: X86PassConfig's real
; GICombiner passes (addPreLegalizeMachineIR/addPreRegBankSelect) are
; both gated on getOptLevel() != CodeGenOptLevel::None, so plain
; -global-isel at this opt level genuinely does NOT simplify e.g.
; `mul x, 1` or `and x, -1` -- it emits a real imull/andl. gmir's own
; new fold()-based combiner (see IR/GMIRDialect.cpp, GMIRCombiner.cpp)
; DOES simplify these, so -enable-mlir-isel's output correctly and
; intentionally diverges from plain -global-isel's for exactly the
; functions below that exercise a real simplification -- a genuine new
; optimization, not something M4's "reproduce what the safety net would
; do anyway" byte-diff discipline applies to. Verified two ways instead:
; the GMIR-prefixed FileCheck below (gmir-level proof the new fold()s
; ran, same -print-gmir-after-X discipline every M4 slice used) and the
; ASM-prefixed FileCheck (proof the simplification survives all the way
; to correct final codegen) -- mirroring vector-scalarize.ll's precedent
; of FileChecking final assembly directly when the usual oracle doesn't
; apply.
;
; mul_neg_one/mul_neg_one_lhs (M5 slice 2) are this file's first case
; exercising a genuine OpRewritePattern (MulNegOneToSubPattern,
; GMIRCombiner.cpp) rather than a fold() -- `mul x, -1 -> sub(0, x)`
; needs a brand-new gmir.sub op, which fold() can't emit (it can only
; return an existing operand or a constant). Confirmed empirically,
; not assumed, that these two functions' divergence from plain
; -global-isel is asymmetric: X86's own InstructionSelect TableGen
; patterns already recognize `mul x, -1` (constant on the right) as a
; NEG idiom on their own, with zero combiner involved, so
; mul_neg_one's byte-diff against plain -global-isel would actually be
; empty too -- but `mul -1, x` (constant on the *left*) hits no such
; pattern (plain -global-isel emits a real imull there), unlike
; MulNegOneToSubPattern, which explicitly checks both operand orders.
; Both functions are FileChecked the same way regardless, for
; consistency with the rest of this file and because the ASM checks
; don't depend on which specific reason the divergence exists for.

define i32 @add_zero(i32 %x) {
  %r = add i32 %x, 0
  ret i32 %r
}
; GMIR-LABEL: func.func @add_zero
; GMIR-NOT: gmir.add
; GMIR: return %arg0
; ASM-LABEL: add_zero:
; ASM-NOT: addl
; ASM: movl %edi, %eax

define i32 @add_zero_lhs(i32 %x) {
  %r = add i32 0, %x
  ret i32 %r
}
; GMIR-LABEL: func.func @add_zero_lhs
; GMIR-NOT: gmir.add
; GMIR: return %arg0
; ASM-LABEL: add_zero_lhs:
; ASM-NOT: addl
; ASM: movl %edi, %eax

define i32 @sub_self(i32 %x) {
  %r = sub i32 %x, %x
  ret i32 %r
}
; GMIR-LABEL: func.func @sub_self
; GMIR-NOT: gmir.sub
; GMIR: gmir.constant 0
; GMIR: return
; ASM-LABEL: sub_self:
; ASM-NOT: subl
; ASM: xorl %eax, %eax

define i32 @mul_zero(i32 %x) {
  %r = mul i32 %x, 0
  ret i32 %r
}
; GMIR-LABEL: func.func @mul_zero
; GMIR-NOT: gmir.mul
; GMIR: gmir.constant 0
; GMIR: return
; ASM-LABEL: mul_zero:
; ASM-NOT: imull
; ASM: xorl %eax, %eax

define i32 @mul_one(i32 %x) {
  %r = mul i32 1, %x
  ret i32 %r
}
; GMIR-LABEL: func.func @mul_one
; GMIR-NOT: gmir.mul
; GMIR: return %arg0
; ASM-LABEL: mul_one:
; ASM-NOT: imull
; ASM: movl %edi, %eax

define i32 @and_self(i32 %x) {
  %r = and i32 %x, %x
  ret i32 %r
}
; GMIR-LABEL: func.func @and_self
; GMIR-NOT: gmir.and
; GMIR: return %arg0
; ASM-LABEL: and_self:
; ASM-NOT: andl
; ASM: movl %edi, %eax

define i32 @and_allones(i32 %x) {
  %r = and i32 %x, -1
  ret i32 %r
}
; GMIR-LABEL: func.func @and_allones
; GMIR-NOT: gmir.and
; GMIR: return %arg0
; ASM-LABEL: and_allones:
; ASM-NOT: andl
; ASM: movl %edi, %eax

define i32 @and_zero(i32 %x) {
  %r = and i32 %x, 0
  ret i32 %r
}
; GMIR-LABEL: func.func @and_zero
; GMIR-NOT: gmir.and
; GMIR: gmir.constant 0
; GMIR: return
; ASM-LABEL: and_zero:
; ASM-NOT: andl
; ASM: xorl %eax, %eax

define i32 @or_self(i32 %x) {
  %r = or i32 %x, %x
  ret i32 %r
}
; GMIR-LABEL: func.func @or_self
; GMIR-NOT: gmir.or
; GMIR: return %arg0
; ASM-LABEL: or_self:
; ASM-NOT: orl
; ASM: movl %edi, %eax

define i32 @or_zero(i32 %x) {
  %r = or i32 %x, 0
  ret i32 %r
}
; GMIR-LABEL: func.func @or_zero
; GMIR-NOT: gmir.or
; GMIR: return %arg0
; ASM-LABEL: or_zero:
; ASM-NOT: orl
; ASM: movl %edi, %eax

define i32 @or_allones(i32 %x) {
  %r = or i32 %x, -1
  ret i32 %r
}
; GMIR-LABEL: func.func @or_allones
; GMIR-NOT: gmir.or
; GMIR: gmir.constant -1
; GMIR: return
; ASM-LABEL: or_allones:
; ASM-NOT: orl
; ASM: movl $-1, %eax

define i32 @xor_self(i32 %x) {
  %r = xor i32 %x, %x
  ret i32 %r
}
; GMIR-LABEL: func.func @xor_self
; GMIR-NOT: gmir.xor
; GMIR: gmir.constant 0
; GMIR: return
; ASM-LABEL: xor_self:
; ASM-NOT: xorl %eax, %edi
; ASM: xorl %eax, %eax

define i32 @xor_zero(i32 %x) {
  %r = xor i32 %x, 0
  ret i32 %r
}
; GMIR-LABEL: func.func @xor_zero
; GMIR-NOT: gmir.xor
; GMIR: return %arg0
; ASM-LABEL: xor_zero:
; ASM-NOT: xorl %eax, %edi
; ASM: movl %edi, %eax

define i32 @const_add(i32 %unused) {
  %r = add i32 3, 4
  ret i32 %r
}
; GMIR-LABEL: func.func @const_add
; GMIR-NOT: gmir.add
; GMIR: gmir.constant 7
; GMIR: return
; ASM-LABEL: const_add:
; ASM: movl $7, %eax

define i32 @const_add_overflow(i32 %unused) {
  %r = add i32 2147483647, 1
  ret i32 %r
}
; GMIR-LABEL: func.func @const_add_overflow
; GMIR-NOT: gmir.add
; GMIR: gmir.constant -2147483648
; GMIR: return
; ASM-LABEL: const_add_overflow:
; ASM: movl $-2147483648, %eax

define i32 @mul_neg_one(i32 %x) {
  %r = mul i32 %x, -1
  ret i32 %r
}
; GMIR-LABEL: func.func @mul_neg_one
; GMIR-NOT: gmir.mul
; GMIR: gmir.constant 0
; GMIR: gmir.sub
; GMIR: return
; ASM-LABEL: mul_neg_one:
; ASM-NOT: imull
; ASM: negl %eax

define i32 @mul_neg_one_lhs(i32 %x) {
  %r = mul i32 -1, %x
  ret i32 %r
}
; GMIR-LABEL: func.func @mul_neg_one_lhs
; GMIR-NOT: gmir.mul
; GMIR: gmir.constant 0
; GMIR: gmir.sub
; GMIR: return
; ASM-LABEL: mul_neg_one_lhs:
; ASM-NOT: imull
; ASM: negl %eax

; M5 slice 4's constant-reassociation/repeated-operand identities
; (ReassociateConstOpPattern/RepeatedOperandIdempotentPattern/
; XorSelfCancelPattern, GMIRCombiner.cpp), ported from DAGCombiner.cpp's
; reassociateOpsCommutative -- shared identically by visitADD/MUL/AND/
; OR/XOR. (x+3)+4 -> x+7, etc.: both original adds/muls/etc. collapse
; into a single op against the combined constant.

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
; ASM: leal 7(%rdi), %eax

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
; ASM: imull $12, %edi, %eax

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
; ASM: andl $240, %eax

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
; ASM: orl $3, %eax

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
; ASM: xorl $2, %eax

; Repeated-operand identities: (a&b)&a -> a&b, (a|b)|b -> a|b,
; (a^b)^a -> b. Each original outer op vanishes entirely -- AND/OR
; collapse to the single inner op; XOR collapses all the way to a bare
; return of the surviving operand.

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
; ASM: andl %edi, %eax

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
; ASM: orl %edi, %eax

define i32 @xor_self_cancel(i32 %a, i32 %b) {
  %m = xor i32 %a, %b
  %r = xor i32 %m, %a
  ret i32 %r
}
; GMIR-LABEL: func.func @xor_self_cancel
; GMIR-NOT: gmir.xor
; GMIR: return %arg1
; ASM-LABEL: xor_self_cancel:
; ASM-NOT: xorl
; ASM: movl %esi, %eax

; Regression case for a real bug an expert review found: without
; XorOp::fold's x^x->0 identity (added alongside this test), reducing
; %n to %b here (via XorSelfCancelPattern above) rewires %r's operands
; to (b, b) -- a live x^x that nothing folded away. With the fix, the
; whole chain collapses to a constant.
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
; ASM: xorl %eax, %eax

; M5 slice 5's three new SubOp::fold arms (DAGCombiner.cpp:4406-4416):
; A-(A-B)->B, (A+B)-A->B, (A+B)-B->A. All SSA-equality-based, same shape
; as sub_self's existing x-x->0 above -- each collapses both the inner
; and outer sub/add entirely, down to a bare return of one operand.

define i32 @sub_a_minus_a_minus_b(i32 %a, i32 %b) {
  %t = sub i32 %a, %b
  %r = sub i32 %a, %t
  ret i32 %r
}
; GMIR-LABEL: func.func @sub_a_minus_a_minus_b
; GMIR-NOT: gmir.sub
; GMIR: return %arg1
; ASM-LABEL: sub_a_minus_a_minus_b:
; ASM-NOT: subl
; ASM: movl %esi, %eax

; A-(A-B)->B's A=0 instantiation (double negation, 0-(0-B)->B) --
; deliberately NOT a separate fold() arm, falls out of the same general
; rule above "for free" because GMIRImporter's constant memoization maps
; both literal `0` operands to the identical materialized gmir.constant
; Value, satisfying the general rule's SSA-equality check on A.
define i32 @sub_double_neg(i32 %b) {
  %t = sub i32 0, %b
  %r = sub i32 0, %t
  ret i32 %r
}
; GMIR-LABEL: func.func @sub_double_neg
; GMIR-NOT: gmir.sub
; GMIR: return %arg0
; ASM-LABEL: sub_double_neg:
; ASM-NOT: negl
; ASM: movl %edi, %eax

define i32 @sub_add_a(i32 %a, i32 %b) {
  %t = add i32 %a, %b
  %r = sub i32 %t, %a
  ret i32 %r
}
; GMIR-LABEL: func.func @sub_add_a
; GMIR-NOT: gmir.add
; GMIR-NOT: gmir.sub
; GMIR: return %arg1
; ASM-LABEL: sub_add_a:
; ASM-NOT: addl
; ASM-NOT: subl
; ASM: movl %esi, %eax

define i32 @sub_add_b(i32 %a, i32 %b) {
  %t = add i32 %a, %b
  %r = sub i32 %t, %b
  ret i32 %r
}
; GMIR-LABEL: func.func @sub_add_b
; GMIR-NOT: gmir.add
; GMIR-NOT: gmir.sub
; GMIR: return %arg0
; ASM-LABEL: sub_add_b:
; ASM-NOT: addl
; ASM-NOT: subl
; ASM: movl %edi, %eax

; M5 slice 5's SubMinusOneToXorPattern: sub(-1, x) -> xor(x, -1)
; (DAGCombiner.cpp:4398-4400). gmir.sub isn't Commutative, so unlike
; MulNegOneToSubPattern's both-orders test above, sub(x, -1) is a
; genuinely different value (x+1, not this identity) and deliberately
; has no parallel "other order" test here.
define i32 @sub_neg_one_to_xor(i32 %x) {
  %r = sub i32 -1, %x
  ret i32 %r
}
; GMIR-LABEL: func.func @sub_neg_one_to_xor
; GMIR-NOT: gmir.sub
; GMIR: gmir.constant -1
; GMIR: gmir.xor
; GMIR: return
; ASM-LABEL: sub_neg_one_to_xor:
; ASM-NOT: subl
; ASM: notl %eax

; M5 slice 5's XorAndDeMorganPattern: xor(and(x,y), y) -> and(xor(x,-1),
; y) (DAGCombiner.cpp:10638-10644, hasOneUse()-gated -- the first
; hasOneUse()-gated pattern in this file). DAGCombiner's own rule only
; checks one of 4 structurally equivalent operand-order combinations;
; this pattern deliberately generalizes to all 4 (see GMIRCombiner.cpp's
; doc comment) -- the three positive tests below cover 3 of those
; combinations (the 4th, and's-RHS-swapped + xor's-RHS-swapped, is
; algebraically identical to xor_and_demorgan_and_swapped modulo which
; xor operand is checked first, not a distinct code path worth a 4th
; test), plus a negative test proving the hasOneUse() gate actually
; gates something.

define i32 @xor_and_demorgan_lhs(i32 %x, i32 %y) {
  %m = and i32 %x, %y
  %r = xor i32 %m, %y
  ret i32 %r
}
; GMIR-LABEL: func.func @xor_and_demorgan_lhs
; GMIR-NOT: gmir.and %arg0, %arg1
; GMIR: gmir.constant -1
; GMIR: gmir.xor %arg0
; GMIR: gmir.and
; GMIR: return
; ASM-LABEL: xor_and_demorgan_lhs:
; ASM-NOT: xorl

define i32 @xor_and_demorgan_rhs(i32 %x, i32 %y) {
  %m = and i32 %x, %y
  %r = xor i32 %y, %m
  ret i32 %r
}
; GMIR-LABEL: func.func @xor_and_demorgan_rhs
; GMIR-NOT: gmir.and %arg0, %arg1
; GMIR: gmir.constant -1
; GMIR: gmir.xor %arg0
; GMIR: gmir.and
; GMIR: return
; ASM-LABEL: xor_and_demorgan_rhs:
; ASM-NOT: xorl

define i32 @xor_and_demorgan_and_swapped(i32 %x, i32 %y) {
  %m = and i32 %y, %x
  %r = xor i32 %m, %y
  ret i32 %r
}
; GMIR-LABEL: func.func @xor_and_demorgan_and_swapped
; GMIR-NOT: gmir.and %arg1, %arg0
; GMIR: gmir.constant -1
; GMIR: gmir.xor %arg0
; GMIR: gmir.and
; GMIR: return
; ASM-LABEL: xor_and_demorgan_and_swapped:
; ASM-NOT: xorl

; Negative test: %m is also returned directly (a second use), so the
; hasOneUse() gate must block the rewrite -- the original xor(and,y)
; shape must survive unchanged.
define i32 @xor_and_demorgan_multiuse(i32 %x, i32 %y, ptr %out) {
  %m = and i32 %x, %y
  store i32 %m, ptr %out
  %r = xor i32 %m, %y
  ret i32 %r
}
; GMIR-LABEL: func.func @xor_and_demorgan_multiuse
; GMIR: gmir.and %arg0, %arg1
; GMIR-NOT: gmir.constant -1
; GMIR: gmir.xor
; GMIR: return
; ASM-LABEL: xor_and_demorgan_multiuse:
; ASM: andl
; ASM: xorl
