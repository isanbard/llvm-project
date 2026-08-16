; REQUIRES: llvm_enable_mlir_isel
; RUN: llc -mtriple=x86_64-unknown-linux-gnu -global-isel %s -o %t.gisel.s
; RUN: llc -mtriple=x86_64-unknown-linux-gnu -enable-mlir-isel %s -o %t.mlir.s
; RUN: diff %t.gisel.s %t.mlir.s
;
; See aggregate-memops-aarch64.ll for the AArch64 variant -- see
; scalar-arith.ll for the rationale on why REQUIRES/the -global-isel
; oracle need a separate file per target rather than extra RUN lines in
; this one.
;
; Verifies aggregate-typed load/store, wiring GMIRImporter.cpp's existing
; computeGMIRLeafTypes flattening helper into importLoad/importStore
; (previously only exercised the trivial single-leaf case -- see
; memory-ops.ll): a two-field struct (leaf alignment differs from the
; struct's own alignment), a struct containing a struct (recursive
; flattening), and a fixed-size array. Every function's own signature
; stays pointer-only (loads/stores the aggregate via memory, never takes
; or returns one directly) -- see the design doc's M3 scope note: a
; function's own formal-parameter/return-type conversion is unrelated to
; this slice and still bails on anything non-scalar/pointer, so an
; aggregate-typed signature would make the whole function fall back
; before ever reaching importLoad/importStore. Storing back to %src (not
; just %dst) also exercises the multi-leaf value flowing from an
; aggregate load's ValueMap entry into an aggregate store's getOperands
; lookup, not just a fresh load immediately followed by a store.

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
