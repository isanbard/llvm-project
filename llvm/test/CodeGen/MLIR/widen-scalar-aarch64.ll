; REQUIRES: llvm_enable_mlir_isel
; REQUIRES: aarch64-registered-target
; RUN: llc -mtriple=aarch64-unknown-linux-gnu -global-isel %s -o %t.gisel.s
; RUN: llc -mtriple=aarch64-unknown-linux-gnu -enable-mlir-isel %s -o %t.mlir.s
; RUN: diff %t.gisel.s %t.mlir.s
;
; AArch64 variant of widen-scalar.ll -- see that file for the full
; rationale (same content, different mtriple; kept as a separate file
; since lit's REQUIRES is file-global, not scoped to individual RUN lines).
;
; RUN: llc -mtriple=aarch64-unknown-linux-gnu -enable-mlir-isel \
; RUN:     -print-gmir-after-legalize %s -o /dev/null 2>&1 | FileCheck %s

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
