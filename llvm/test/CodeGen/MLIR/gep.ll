; REQUIRES: llvm_enable_mlir_isel
; RUN: llc -mtriple=x86_64-unknown-linux-gnu -global-isel %s -o %t.gisel.s
; RUN: llc -mtriple=x86_64-unknown-linux-gnu -enable-mlir-isel %s -o %t.mlir.s
; RUN: diff %t.gisel.s %t.mlir.s
;
; See gep-aarch64.ll for the AArch64 variant -- see scalar-arith.ll for the
; rationale on why REQUIRES/the -global-isel oracle need a separate file
; per target rather than extra RUN lines in this one. See gep-aarch64.ll's
; header for why nested_gep (struct-field-then-variable-index, testing
; coalescing order) isn't included there even though it's included here.
;
; Verifies GEP -> G_PTR_ADD translation (llvm/lib/CodeGen/MLIR/GMIRImporter.cpp's
; importGEP), which ports IRTranslator::translateGetElementPtr's
; constant-index-coalescing algorithm for -global-isel byte parity: a
; struct-field-only GEP (one final constant-offset flush, no variable
; index), a variable-index array GEP (gmir.mul + gmir.ptr_add), a
; struct-field-then-variable-index GEP (tests coalescing order: the
; constant struct offset is flushed before the variable term), and an
; all-zero-index GEP (pure value alias, no ops emitted at all).

define i64 @struct_gep(ptr %p) {
entry:
  %g = getelementptr {i32, i64}, ptr %p, i64 0, i32 1
  %v = load i64, ptr %g
  ret i64 %v
}

define i32 @array_gep(ptr %p, i64 %i) {
entry:
  %g = getelementptr [10 x i32], ptr %p, i64 0, i64 %i
  %v = load i32, ptr %g
  ret i32 %v
}

define i64 @nested_gep(ptr %p, i64 %i) {
entry:
  %g = getelementptr {i32, [4 x i64]}, ptr %p, i64 0, i32 1, i64 %i
  %v = load i64, ptr %g
  ret i64 %v
}

define ptr @zero_index_gep(ptr %p) {
entry:
  %g = getelementptr {i32, i32}, ptr %p, i64 0, i32 0
  ret ptr %g
}
