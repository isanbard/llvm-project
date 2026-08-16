; REQUIRES: llvm_enable_mlir_isel
; REQUIRES: aarch64-registered-target
; RUN: llc -mtriple=aarch64-unknown-linux-gnu -global-isel %s -o %t.gisel.s
; RUN: llc -mtriple=aarch64-unknown-linux-gnu -enable-mlir-isel %s -o %t.mlir.s
; RUN: diff %t.gisel.s %t.mlir.s
;
; AArch64 variant of call.ll -- see that file for the full rationale (same
; content, different mtriple, plus call_struct_arg -- see call.ll's header
; for why that case is AArch64-only: X86's own -global-isel can't
; translate a struct-by-value call argument at all, so there's no working
; oracle to diff against there). call_struct_arg exercises argLeafCounts
; regrouping end-to-end (built via a load from a pointer argument to keep
; the caller's own signature scalar/pointer-only, per the established M3
; scope boundary) and is especially worth diffing on this target: AAPCS64
; packs small aggregates into consecutive argument registers, a real
; target-specific ABI path X86 testing can't exercise even where it does
; work.

declare i32 @callee(i32, i32)
declare void @callee_void(i32)
declare i32 @callee_struct({i32, i32})

define i32 @call_scalar_args(i32 %a, i32 %b) {
entry:
  %r = call i32 @callee(i32 %a, i32 %b)
  ret i32 %r
}

define void @call_void(i32 %a) {
entry:
  call void @callee_void(i32 %a)
  ret void
}

define i32 @call_struct_arg(ptr %p) {
entry:
  %s = load {i32, i32}, ptr %p
  %r = call i32 @callee_struct({i32, i32} %s)
  ret i32 %r
}
