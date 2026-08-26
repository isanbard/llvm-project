; REQUIRES: llvm_enable_mlir_isel
; REQUIRES: aarch64-registered-target
; RUN: llc -mtriple=aarch64-unknown-linux-gnu -global-isel %s -o %t.gisel.s
; RUN: llc -mtriple=aarch64-unknown-linux-gnu -enable-mlir-isel %s -o %t.mlir.s
; RUN: diff %t.gisel.s %t.mlir.s
;
; AArch64 variant of memory-ops.ll -- see that file for the full rationale
; (same content, different mtriple; kept as a separate file rather than
; extra RUN lines in the same file since lit's REQUIRES is file-global, not
; scoped to individual RUN lines).

define i32 @store_load(i32 %a) {
entry:
  %p = alloca i32, align 4
  store i32 %a, ptr %p
  %v = load i32, ptr %p
  ret i32 %v
}

define void @store_through_ptr(ptr %p, i32 %v) {
entry:
  store i32 %v, ptr %p
  ret void
}

define i32 @load_through_ptr(ptr %p) {
entry:
  %v = load i32, ptr %p
  ret i32 %v
}

define ptr @load_ptr_through_ptr(ptr %pp) {
entry:
  %v = load ptr, ptr %pp
  ret ptr %v
}

define i64 @store_load_i64(i64 %a) {
entry:
  %p = alloca i64, align 8
  store i64 %a, ptr %p
  %v = load i64, ptr %p
  ret i64 %v
}

define i32 @volatile_load(ptr %p) {
entry:
  %v = load volatile i32, ptr %p
  ret i32 %v
}

define void @volatile_store(ptr %p, i32 %v) {
entry:
  store volatile i32 %v, ptr %p
  ret void
}

define i32 @atomic_load(ptr %p) {
entry:
  %v = load atomic i32, ptr %p seq_cst, align 4
  ret i32 %v
}

define void @atomic_store(ptr %p, i32 %v) {
entry:
  store atomic i32 %v, ptr %p seq_cst, align 4
  ret void
}

define i32 @invariant_load(ptr %p) {
entry:
  %v = load i32, ptr %p, !invariant.load !0
  ret i32 %v
}

define i32 @nontemporal_load(ptr %p) {
entry:
  %v = load i32, ptr %p, !nontemporal !1
  ret i32 %v
}

define void @nontemporal_store(ptr %p, i32 %v) {
entry:
  store i32 %v, ptr %p, !nontemporal !1
  ret void
}

!0 = !{}
!1 = !{i32 1}
