; RUN: llc -mtriple=x86_64-unknown-linux-gnu %s -o %t.normal.s
; RUN: llc -mtriple=x86_64-unknown-linux-gnu -enable-mlir-isel %s -o %t.mlir.s
; RUN: diff %t.normal.s %t.mlir.s
;
; Verifies the fallback path for functions genuinely outside the currently-
; supported subset (a call, a switch, and a dynamic-size alloca; see
; llvm/lib/CodeGen/MLIR/GMIRImporter.cpp, whose per-instruction dispatch has
; no case for calls/switches, and whose importAlloca bails on
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
; and static-alloca/load/store are genuinely translated instead of always
; falling back; see scalar-arith.ll, control-flow.ll, and memory-ops.ll.)
;
; @huge_alloca covers importAlloca's int64 overflow guard: an allocation
; size at or above 2^63 doesn't overflow the uint64_t byte-count
; computation, but does overflow the int64_t attribute the gmir op stores
; it in, so it must be rejected explicitly rather than silently
; reinterpreted as negative.
;
; @vector_gep and @narrow_index_gep additionally cover importGEP's two
; bail-out conditions (see gep.ll for the genuinely-translated GEP cases):
; a vector-typed GEP, and a variable index whose bit width doesn't match
; the pointer-index type's width (gmir has no sext/trunc op yet to fix
; that up).

declare i32 @callee(i32)

define i32 @calls_something(i32 %a) {
  %r = call i32 @callee(i32 %a)
  ret i32 %r
}

; Regression test: convertType (GMIRImporter.cpp) caps integer bit width at
; 64 -- gmir.constant always stores its value in a 64-bit attribute, so a
; wider !gmir.llt would make the translator's truncate-back-down step
; truncate *up*, which APInt::trunc() forbids and asserts on. i128 (and
; wider) values must fall back rather than hit that.
define i128 @add_i128(i128 %a) {
entry:
  %r = add i128 %a, 5
  ret i128 %r
}

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

define i8 @huge_alloca() {
entry:
  %p = alloca i8, i64 9223372036854775808
  store i8 0, ptr %p
  %v = load i8, ptr %p
  ret i8 %v
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
