; REQUIRES: llvm_enable_mlir_isel
; RUN: llc -mtriple=x86_64-unknown-linux-gnu -global-isel %s -o %t.gisel.s
; RUN: llc -mtriple=x86_64-unknown-linux-gnu -enable-mlir-isel %s -o %t.mlir.s
; RUN: diff %t.gisel.s %t.mlir.s
;
; See widen-scalar-aarch64.ll for the AArch64 variant -- see scalar-arith.ll
; for the rationale on why REQUIRES/the -global-isel oracle need a separate
; file per target rather than extra RUN lines in this one, and on why the
; RUN lines above deliberately don't pin -O0 (doing so surfaces an
; unrelated, pre-existing basic-block-numbering divergence in -global-isel
; itself -- see the M3-era GEP finding in
; ~/llvm/mlir_instruction_selection_plan.md's design doc, confirmed still
; true here: `llc -O0 -global-isel` numbers even a single-block function's
; only block "# %bb.1:" instead of "# %bb.0:", with zero gmir involvement).
;
; RUN: llc -mtriple=x86_64-unknown-linux-gnu -enable-mlir-isel \
; RUN:     -print-gmir-after-legalize %s -o /dev/null 2>&1 | FileCheck %s
;
; This is the actual proof, not the asm-diff above (same reasoning as
; narrow-scalar-i686.ll -- the real target Legalizer stays in the pipeline
; as a safety net and would silently do this widening itself with zero new
; M4 code). Verified directly against X86LegalizerInfo.cpp: G_ADD/G_SUB/
; G_AND/G_OR/G_XOR/G_MUL all clampScalar(0, s8, sMaxScalar), and i1 isn't
; in any legalFor list, so it resolves to WidenScalar(s8) here (clampScalar
; widens any value below its floor -- confirmed empirically via
; -print-gmir-after-legalize, not assumed: i1 widens to s8 on X86_64 but
; s32 on AArch64, since AArch64's own clampScalar floor is s32; the
; FileCheck patterns below don't hardcode the wide width for exactly this
; reason). WidenScalarPattern (GMIRLegalizer.cpp) implements the single
; shared widenScalar switch case LegalizerHelper.cpp uses for all six of
; these opcodes: any-extend both operands to the wide type, re-run the
; same op there, truncate the result back down (gmir.anyext x2 -> op ->
; gmir.trunc) -- these two ops only ever come from that pattern, never
; from GMIRImporter directly, so their presence is unambiguous evidence
; the new legalizer code path ran.

define i1 @add_i1(i1 %a, i1 %b) {
  %c = add i1 %a, %b
  ret i1 %c
}
; CHECK-LABEL: func.func @add_i1
; CHECK: "gmir.anyext"
; CHECK: "gmir.anyext"
; CHECK: gmir.add
; CHECK: "gmir.trunc"

define i1 @sub_i1(i1 %a, i1 %b) {
  %c = sub i1 %a, %b
  ret i1 %c
}
; CHECK-LABEL: func.func @sub_i1
; CHECK: "gmir.anyext"
; CHECK: "gmir.anyext"
; CHECK: gmir.sub
; CHECK: "gmir.trunc"

define i1 @and_i1(i1 %a, i1 %b) {
  %c = and i1 %a, %b
  ret i1 %c
}
; CHECK-LABEL: func.func @and_i1
; CHECK: "gmir.anyext"
; CHECK: "gmir.anyext"
; CHECK: gmir.and
; CHECK: "gmir.trunc"

define i1 @or_i1(i1 %a, i1 %b) {
  %c = or i1 %a, %b
  ret i1 %c
}
; CHECK-LABEL: func.func @or_i1
; CHECK: "gmir.anyext"
; CHECK: "gmir.anyext"
; CHECK: gmir.or
; CHECK: "gmir.trunc"

define i1 @xor_i1(i1 %a, i1 %b) {
  %c = xor i1 %a, %b
  ret i1 %c
}
; CHECK-LABEL: func.func @xor_i1
; CHECK: "gmir.anyext"
; CHECK: "gmir.anyext"
; CHECK: gmir.xor
; CHECK: "gmir.trunc"

define i1 @mul_i1(i1 %a, i1 %b) {
  %c = mul i1 %a, %b
  ret i1 %c
}
; CHECK-LABEL: func.func @mul_i1
; CHECK: "gmir.anyext"
; CHECK: "gmir.anyext"
; CHECK: gmir.mul
; CHECK: "gmir.trunc"
