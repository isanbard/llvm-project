; REQUIRES: llvm_enable_mlir_isel
; RUN: llc -mtriple=i686-unknown-linux-gnu -global-isel %s -o %t.gisel.s
; RUN: llc -mtriple=i686-unknown-linux-gnu -enable-mlir-isel %s -o %t.mlir.s
; RUN: diff %t.gisel.s %t.mlir.s
;
; The two RUN lines above are a regression net, not the proof that M4
; slice 1's GMIRLegalizer/NarrowScalarAddSubPattern actually ran: the real
; target Legalizer MachineFunctionPass stays permanently in the pipeline
; after MLIRInstructionSelect (to catch Custom-resolving ops -- an
; already-locked design choice), so it would silently narrow the plain
; s64 gmir.add/gmir.sub itself even with zero new M4 code, making this
; asm-diff alone unable to distinguish "the new pattern fired" from "the
; safety net quietly did the work instead". See
; ~/llvm/mlir_instruction_selection_plan.md's M4 section and
; GMIRLegalizer.h for the full rationale.
;
; RUN: llc -mtriple=i686-unknown-linux-gnu -enable-mlir-isel \
; RUN:     -print-gmir-after-legalize %s -o /dev/null 2>&1 | FileCheck %s
;
; This is the actual proof: i686's X86LegalizerInfo (Is64Bit=false) makes
; s64 G_ADD/G_SUB resolve to NarrowScalar(s32) via clampScalar -- verified
; directly against X86LegalizerInfo.cpp:60,184-201. GMIRLegalizerInfoAdapter
; queries that same real target LegalizerInfo, and NarrowScalarAddSubPattern
; (GMIRLegalizer.cpp) rewrites the plain s64 gmir.add/gmir.sub into the
; exact hi/lo+carry split LegalizerHelper::narrowScalarAddSub itself would
; produce (gmir.unmerge -> gmir.uaddo/usubo (first chunk) ->
; gmir.uadde/usube (second chunk) -> gmir.merge) -- these ops only ever
; come from that pattern, never from GMIRImporter directly, so their
; presence here is unambiguous evidence the new legalizer code path ran.

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
