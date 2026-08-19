; REQUIRES: llvm_enable_mlir_isel
; REQUIRES: aarch64-registered-target
; RUN: llc -mtriple=aarch64-unknown-linux-gnu -enable-mlir-isel \
; RUN:     -print-gmir-after-combine %s -o /dev/null 2>&1 | FileCheck --check-prefix=GMIR %s
; RUN: llc -mtriple=aarch64-unknown-linux-gnu -enable-mlir-isel %s -o - | FileCheck --check-prefix=ASM %s
;
; AArch64 variant of combine-disjoint-add.ll -- see that file for the
; full rationale. Unlike combine-identities-aarch64.ll, this file does
; NOT use the usual -global-isel byte-diff oracle: confirmed empirically
; that add_disjoint_and's mlir-isel output (orr) genuinely diverges from
; plain -global-isel's (add) here too -- AArch64O0PreLegalizerCombiner
; does not do this simplification independently, unlike e.g.
; combine-identities-aarch64.ll's mul_neg_one. A file-wide byte-diff
; RUN line would fail on this function, so this file uses the same
; GMIR + ASM FileCheck pair combine-disjoint-add.ll uses for the
; identical reason on X86.

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
; ASM-NOT: {{^	add	}}
; ASM: orr

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
; ASM-NOT: orr
; ASM: {{^	add	}}
