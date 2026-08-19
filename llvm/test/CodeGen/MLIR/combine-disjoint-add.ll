; REQUIRES: llvm_enable_mlir_isel
; RUN: llc -mtriple=x86_64-unknown-linux-gnu -enable-mlir-isel \
; RUN:     -print-gmir-after-combine %s -o /dev/null 2>&1 | FileCheck --check-prefix=GMIR %s
; RUN: llc -mtriple=x86_64-unknown-linux-gnu -enable-mlir-isel %s -o - | FileCheck --check-prefix=ASM %s
;
; See combine-disjoint-add-aarch64.ll for the AArch64 variant. M5 slice
; 3's DisjointAddToOrPattern (GMIRCombiner.cpp) is the first gmir
; combiner pattern gated on a genuine TargetLowering query
; (GMIRTargetLoweringAdapter::isOperationLegal, mirroring
; GMIRLegalizerInfoAdapter's existing LegalizerInfo-wrapping precedent
; in GMIRLegalizer.cpp) rather than just an op's own intrinsic
; semantics: `add(and(a,C1), and(b,C2)) -> or(and(a,C1), and(b,C2))`
; when C1 and C2 are disjoint bitmasks and TLI reports OR legal for the
; result type -- DAGCombiner.cpp's visitADD, narrowed to the purely
; structural case gmir can prove without a general KnownBits-style
; value-tracking analysis (see GMIRCombiner.cpp's doc comment on
; DisjointAddToOrPattern).
;
; Kept in a dedicated file (not folded into combine-identities.ll/
; -aarch64.ll) because add_disjoint_and's mlir-isel output genuinely
; diverges from plain -global-isel on *both* X86 and AArch64 (confirmed
; empirically -- unlike every function in combine-identities-aarch64.ll,
; which byte-diff-matches there), which would break that file's
; single, file-wide byte-diff RUN line. Uses the same GMIR-prefixed
; -print-gmir-after-combine FileCheck (proof gmir's own pattern ran)
; plus an ASM-prefixed FileCheck (proof it survives to final codegen)
; as combine-identities.ll's mul_neg_one cases use for the same reason.
;
; add_overlapping_and is the negative case: masks 0xF and 0x1F overlap
; (share bit 3), so the pattern must not fire -- proves the
; disjointness check actually gates something, not just decoration.

define i32 @add_disjoint_and(i32 %a, i32 %b) {
  %m1 = and i32 %a, 15
  %m2 = and i32 %b, 240
  %r = add i32 %m1, %m2
  ret i32 %r
}
; GMIR-LABEL: func.func @add_disjoint_and
; GMIR-NOT: gmir.add
; GMIR: gmir.or
; GMIR: return
; ASM-LABEL: add_disjoint_and:
; ASM-NOT: addl
; ASM: orl

define i32 @add_overlapping_and(i32 %a, i32 %b) {
  %m1 = and i32 %a, 15
  %m2 = and i32 %b, 31
  %r = add i32 %m1, %m2
  ret i32 %r
}
; GMIR-LABEL: func.func @add_overlapping_and
; GMIR: gmir.add
; GMIR-NOT: gmir.or
; GMIR: return
; ASM-LABEL: add_overlapping_and:
; ASM-NOT: orl
