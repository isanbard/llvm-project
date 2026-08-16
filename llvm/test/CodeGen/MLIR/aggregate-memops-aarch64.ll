; REQUIRES: llvm_enable_mlir_isel
; REQUIRES: aarch64-registered-target
; RUN: llc -mtriple=aarch64-unknown-linux-gnu -global-isel %s -o %t.gisel.s
; RUN: llc -mtriple=aarch64-unknown-linux-gnu -enable-mlir-isel %s -o %t.mlir.s
; RUN: diff %t.gisel.s %t.mlir.s
;
; AArch64 variant of aggregate-memops.ll -- see that file for the full
; rationale (same content, different mtriple; kept as a separate file
; since lit's REQUIRES is file-global, not scoped to individual RUN
; lines).

define void @load_store_struct(ptr %src, ptr %dst) {
entry:
  %s = load {i32, i64}, ptr %src
  store {i32, i64} %s, ptr %dst
  ret void
}

define void @load_store_nested_struct(ptr %src, ptr %dst) {
entry:
  %s = load {i32, {i16, i64}}, ptr %src
  store {i32, {i16, i64}} %s, ptr %dst
  ret void
}

define void @load_store_array(ptr %src, ptr %dst) {
entry:
  %a = load [4 x i32], ptr %src
  store [4 x i32] %a, ptr %dst
  ret void
}
