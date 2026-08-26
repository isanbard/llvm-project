; REQUIRES: llvm_enable_mlir_isel
; RUN: llc -mtriple=x86_64-unknown-linux-gnu -global-isel %s -o %t.gisel.s
; RUN: llc -mtriple=x86_64-unknown-linux-gnu -enable-mlir-isel %s -o %t.mlir.s
; RUN: diff %t.gisel.s %t.mlir.s
;
; See widen-scalar-aarch64.ll for the AArch64 variant -- kept as a separate
; file since lit's REQUIRES is file-global, not scoped to individual RUN
; lines (see scalar-arith.ll).
;
; RUN: llc -mtriple=x86_64-unknown-linux-gnu -enable-mlir-isel \
; RUN:     -print-gmir-after-legalize %s -o /dev/null 2>&1 | FileCheck %s
;
; The asm-diff above is a regression net, not proof that GMIRLegalizer's
; WidenScalar dispatch actually ran: the real target Legalizer
; MachineFunctionPass stays permanently in the pipeline after
; MLIRInstructionSelect (to catch Custom-resolving ops), so it would
; silently widen a plain i1 gmir.add/sub/and/or/xor/mul itself even with
; zero GMIRLegalizer code running -- see GMIRLegalizer.h for the full
; rationale. The -print-gmir-after-legalize FileCheck below is the actual
; proof: gmir.anyext/gmir.trunc only ever come from rewriteWidenScalar
; (GMIRLegalizer.cpp), never from GMIRImporter directly, so their presence
; here is unambiguous evidence the new legalizer code path ran.
;
; i1 arithmetic widens to a different scalar width per target
; (X86LegalizerInfo's clampScalar floor is s8, AArch64's is s32), so the
; CHECK patterns below intentionally don't hardcode the wide width.

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
