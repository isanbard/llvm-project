; REQUIRES: llvm_enable_mlir_isel
; RUN: llc -mtriple=i686-unknown-linux-gnu -global-isel %s -o %t.gisel.s
; RUN: llc -mtriple=i686-unknown-linux-gnu -enable-mlir-isel %s -o %t.mlir.s
; RUN: diff %t.gisel.s %t.mlir.s
;
; The two RUN lines above are a regression net, not the proof that
; GMIRLegalizer/AddSubLegalizePattern's NarrowScalar case actually ran: the
; real target Legalizer MachineFunctionPass stays permanently in the
; pipeline after MLIRInstructionSelect (to catch Custom-resolving ops -- an
; already-locked design choice), so it would silently narrow the plain s64
; gmir.add/gmir.sub/gmir.and/gmir.or/gmir.xor/gmir.mul itself even with
; zero GMIRLegalizer code running, making this asm-diff alone unable to
; distinguish "the new pattern fired" from "the safety net quietly did the
; work instead". See GMIRLegalizer.h for the full rationale.
;
; RUN: llc -mtriple=i686-unknown-linux-gnu -enable-mlir-isel \
; RUN:     -print-gmir-after-legalize %s -o /dev/null 2>&1 | FileCheck %s
;
; This is the actual proof: i686's X86LegalizerInfo (Is64Bit=false) makes
; s64 G_ADD/G_SUB/G_AND/G_OR/G_XOR/G_MUL resolve to NarrowScalar(s32) via
; clampScalar -- verified directly against X86LegalizerInfo.cpp.
; GMIRLegalizerInfoAdapter queries that same real target LegalizerInfo, and
; AddSubLegalizePattern/BitwiseLegalizePattern/MulLegalizePattern
; (GMIRLegalizer.cpp) rewrite the plain s64 op into the exact
; hi/lo(+carry, for add/sub)/independent-chunk (for and/or/xor)/schoolbook
; 3-multiply-2-add (for mul) split LegalizerHelper itself would produce
; (gmir.unmerge -> per-chunk op(s) -> gmir.merge) -- these ops only ever
; come from these patterns, never from GMIRImporter directly, so their
; presence here is unambiguous evidence the new legalizer code path ran.
;
; mul64 below reaches MulLegalizePattern's NarrowScalar case (the schoolbook
; 3-multiply/2-add shape: Lo = ALo*BLo; Hi = umulh(ALo,BLo) + ALo*BHi +
; AHi*BLo -- no carry-propagation op needed, since an exact 2-limb split
; only ever executes the algorithm's last-limb step once).

define i64 @add64(i64 %a, i64 %b) {
entry:
  %c = add i64 %a, %b
  ret i64 %c
}
; CHECK-LABEL: func.func @add64
; CHECK: "gmir.unmerge"
; CHECK: "gmir.unmerge"
; CHECK: "gmir.uaddo"
; CHECK: "gmir.uadde"
; CHECK: "gmir.merge"

define i64 @sub64(i64 %a, i64 %b) {
entry:
  %c = sub i64 %a, %b
  ret i64 %c
}
; CHECK-LABEL: func.func @sub64
; CHECK: "gmir.unmerge"
; CHECK: "gmir.unmerge"
; CHECK: "gmir.usubo"
; CHECK: "gmir.usube"
; CHECK: "gmir.merge"

define i64 @and64(i64 %a, i64 %b) {
entry:
  %c = and i64 %a, %b
  ret i64 %c
}
; CHECK-LABEL: func.func @and64
; CHECK: "gmir.unmerge"
; CHECK: "gmir.unmerge"
; CHECK: gmir.and
; CHECK: gmir.and
; CHECK: "gmir.merge"

define i64 @or64(i64 %a, i64 %b) {
entry:
  %c = or i64 %a, %b
  ret i64 %c
}
; CHECK-LABEL: func.func @or64
; CHECK: "gmir.unmerge"
; CHECK: "gmir.unmerge"
; CHECK: gmir.or
; CHECK: gmir.or
; CHECK: "gmir.merge"

define i64 @xor64(i64 %a, i64 %b) {
entry:
  %c = xor i64 %a, %b
  ret i64 %c
}
; CHECK-LABEL: func.func @xor64
; CHECK: "gmir.unmerge"
; CHECK: "gmir.unmerge"
; CHECK: gmir.xor
; CHECK: gmir.xor
; CHECK: "gmir.merge"

define i64 @mul64(i64 %a, i64 %b) {
entry:
  %c = mul i64 %a, %b
  ret i64 %c
}
; CHECK-LABEL: func.func @mul64
; CHECK: "gmir.unmerge"
; CHECK: "gmir.unmerge"
; CHECK: gmir.mul
; CHECK: gmir.umulh
; CHECK: gmir.mul
; CHECK: gmir.mul
; CHECK: gmir.add
; CHECK: gmir.add
; CHECK: "gmir.merge"
