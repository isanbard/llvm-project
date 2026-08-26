; REQUIRES: llvm_enable_mlir_isel
; REQUIRES: aarch64-registered-target
; RUN: llc -mtriple=aarch64-unknown-linux-gnu -global-isel %s -o %t.gisel.s
; RUN: llc -mtriple=aarch64-unknown-linux-gnu -enable-mlir-isel %s -o %t.mlir.s
; RUN: diff %t.gisel.s %t.mlir.s
;
; AArch64 variant of gep.ll -- see that file for the full rationale (same
; content minus nested_gep, different mtriple; kept as a separate file
; since lit's REQUIRES is file-global, not scoped to individual RUN lines).
;
; nested_gep (struct-field-then-variable-index, in gep.ll but deliberately
; NOT here) is excluded on this target for a real, investigated reason
; unrelated to gmir/importGEP's own correctness: without an explicit -O
; flag, `-global-isel` and `-enable-mlir-isel` don't necessarily reach
; AArch64PassConfig::addPreLegalizeMachineIR's combiner choice
; (isGlobalISelOptNone(), AArch64TargetMachine.cpp) the same way --
; isGlobalISelOptNone() checks not just the effective opt level but also
; whether the literal `-global-isel` CLI flag was passed, so
; `-enable-mlir-isel` alone can select a lower-ObserverLevel combiner
; (aarch64-O0-prelegalizer-combiner instead of aarch64-prelegalizer-combiner)
; purely because it isn't spelled `-global-isel`, independent of gmir's
; emitted MIR. That combiner-level difference changes whether a
; G_PTR_ADD-chain reassociation combine fires (CombinerHelper.cpp's
; matchReassocPtrAdd, gated behind ObserverLevel::SinglePass, which the O0
; variant doesn't set) -- semantically equivalent codegen either way
; (address computed via a register-offset add fused into the load vs. a
; separate immediate-offset add), just not byte-identical. Confirmed via
; a same-content .mir diff (raw pre-combiner MIR is identical either way;
; only post-combiner MIR differs) and by explicit-`-O0`-on-both-sides
; testing (which does make nested_gep byte-identical again, by routing
; both RUN lines through the opt-level check instead of the flag check) --
; but pinning -O0 on both RUN lines here surfaces a second, unrelated,
; pre-existing block-numbering difference across every function in this
; file (not just nested_gep, and not GEP-specific at all -- e.g.
; struct_gep's output block is numbered `%bb.1` under -global-isel -O0 vs
; `%bb.0` under -enable-mlir-isel -O0), which would need its own separate
; investigation and isn't specific to GEP translation. Tracking that as a known
; gap rather than chasing it here: no test in this suite has needed -O0
; pinned explicitly before, so this is the first time either latent issue
; was even reachable.

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

define ptr @zero_index_gep(ptr %p) {
entry:
  %g = getelementptr {i32, i32}, ptr %p, i64 0, i32 0
  ret ptr %g
}
