; REQUIRES: llvm_enable_mlir_isel
; RUN: llc -mtriple=x86_64-unknown-linux-gnu -global-isel %s -o %t.gisel.s
; RUN: llc -mtriple=x86_64-unknown-linux-gnu -enable-mlir-isel %s -o %t.mlir.s
; RUN: diff %t.gisel.s %t.mlir.s
;
; See scalar-arith-aarch64.ll for the AArch64 variant of this test -- REQUIRES
; is file-global in lit (ANDed across the whole file, not scoped to
; individual RUN lines), so a second target needs a separate file to stay
; independently REQUIRES-gated rather than making this file's x86_64 lines
; also require aarch64-registered-target.
;
; Only meaningful when MLIR ISel support is actually built in --
; -enable-mlir-isel degrades to plain SelectionDAG (not GlobalISel) in
; builds without -DLLVM_ENABLE_MLIR_ISEL=ON, so the "-global-isel" oracle
; below wouldn't hold there.
;
; Verifies real translation for the currently-supported subset:
; straight-line, single-basic-block, scalar-integer arithmetic (see
; llvm/lib/CodeGen/MLIR/GMIRImporter.cpp and MLIRToGMIRTranslator.cpp).
;
; The oracle here is `-global-isel`, not plain default llc: the MLIR-based
; translator emits the same generic MIR shape IRTranslator would for these
; trivial ops, and both then run through the identical Legalize/
; RegBankSelect/InstructionSelect passes (see
; TargetPassConfig::addCoreISelPasses), so final assembly should be
; byte-identical between the two -- a more principled oracle than comparing
; to SelectionDAG's own (potentially differently-scheduled) output.

define i32 @add32(i32 %a, i32 %b) {
  %r = add i32 %a, %b
  ret i32 %r
}

define i64 @add64(i64 %a, i64 %b) {
  %r = add i64 %a, %b
  ret i64 %r
}

define i32 @sub(i32 %a, i32 %b) {
  %r = sub i32 %a, %b
  ret i32 %r
}

define i32 @mul(i32 %a, i32 %b) {
  %r = mul i32 %a, %b
  ret i32 %r
}

define i32 @sdiv(i32 %a, i32 %b) {
  %r = sdiv i32 %a, %b
  ret i32 %r
}

define i32 @and(i32 %a, i32 %b) {
  %r = and i32 %a, %b
  ret i32 %r
}

define i32 @or(i32 %a, i32 %b) {
  %r = or i32 %a, %b
  ret i32 %r
}

define i32 @xor(i32 %a, i32 %b) {
  %r = xor i32 %a, %b
  ret i32 %r
}

define i32 @constant(i32 %a) {
  %r = add i32 %a, 42
  ret i32 %r
}

define i32 @multi_op(i32 %a, i32 %b, i32 %c) {
  %t0 = add i32 %a, %b
  %t1 = mul i32 %t0, %c
  %t2 = xor i32 %t1, 7
  %r = sub i32 %t2, %a
  ret i32 %r
}
