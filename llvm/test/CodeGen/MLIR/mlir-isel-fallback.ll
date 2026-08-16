; RUN: llc -mtriple=x86_64-unknown-linux-gnu %s -o %t.normal.s
; RUN: llc -mtriple=x86_64-unknown-linux-gnu -enable-mlir-isel %s -o %t.mlir.s
; RUN: diff %t.normal.s %t.mlir.s
;
; Verifies the fallback path for functions genuinely outside the supported
; subset (a switch, and -- since M3 -- a dynamic-size alloca; see
; llvm/lib/CodeGen/MLIR/GMIRImporter.cpp, whose per-instruction dispatch
; has no case for switches, and whose importAlloca bails on
; !AllocaInst::isStaticAlloca()). -enable-mlir-isel must produce
; byte-identical output to a normal llc invocation here in every build
; configuration:
;  - builds without -DLLVM_ENABLE_MLIR_ISEL=ON degrade immediately
;    (createMLIRInstructionSelectPass() returns nullptr);
;  - builds with it on run the real pass, whose importer declines (none of
;    these functions are in the supported subset), so it marks the
;    MachineFunction's ISel as failed and falls back via the same
;    ResetMachineFunctionPass + SelectionDAG path GlobalISel uses.
;
; (Straight-line scalar arithmetic, structured if/else/loop control flow,
; static-alloca/load/store, GEP, aggregate load/store, and direct calls
; (of the shapes importCall accepts) are now genuinely translated instead
; of always falling back; see scalar-arith.ll, control-flow.ll,
; memory-ops.ll, gep.ll, aggregate-memops.ll, and call.ll. A prior version
; of this file used a plain scalar-arg call (@calls_something) as its
; call-related fallback case, from back when M1/M2 had no call support at
; all -- that's no longer a valid fallback case now that such calls are
; genuinely translated (see call.ll), so it moved there and was replaced
; below by cases that are still genuinely out of scope.)
;
; @vector_gep and @narrow_index_gep additionally cover importGEP's two
; bail-out conditions (see gep.ll for the genuinely-translated GEP cases):
; a vector-typed GEP, and a variable index whose bit width doesn't match
; the pointer-index type's width (gmir has no sext/trunc op yet to fix
; that up).
;
; @load_struct_with_float covers computeGMIRLeafTypes failing on a leaf
; type convertType can't represent (float) -- confirms the whole
; load/store bails cleanly rather than partially emitting ops for the
; leaves it could handle (see aggregate-memops.ll for the genuinely-
; translated aggregate load/store cases).
;
; @indirect_call and @vararg_call cover two of importCall's bail-out
; conditions (see call.ll for the genuinely-translated call cases): an
; indirect callee, and a variadic call.

define i32 @switcher(i32 %x) {
entry:
  switch i32 %x, label %default [
    i32 0, label %case0
    i32 1, label %case1
  ]
case0:
  ret i32 10
case1:
  ret i32 20
default:
  ret i32 30
}

define i32 @dynamic_alloca(i32 %n) {
entry:
  %p = alloca i32, i32 %n
  store i32 0, ptr %p
  %v = load i32, ptr %p
  ret i32 %v
}

define <4 x ptr> @vector_gep(<4 x ptr> %p, i64 %i) {
entry:
  %g = getelementptr i32, <4 x ptr> %p, i64 %i
  ret <4 x ptr> %g
}

define i32 @narrow_index_gep(ptr %p, i32 %i) {
entry:
  %g = getelementptr i32, ptr %p, i32 %i
  %v = load i32, ptr %g
  ret i32 %v
}

define void @load_struct_with_float(ptr %src, ptr %dst) {
entry:
  %s = load {i32, float}, ptr %src
  store {i32, float} %s, ptr %dst
  ret void
}

define i32 @indirect_call(ptr %fp, i32 %a) {
entry:
  %r = call i32 %fp(i32 %a)
  ret i32 %r
}

declare i32 @vararg_callee(i32, ...)

define i32 @vararg_call(i32 %a, i32 %b) {
entry:
  %r = call i32 (i32, ...) @vararg_callee(i32 %a, i32 %b)
  ret i32 %r
}
