; REQUIRES: llvm_enable_mlir_isel
; RUN: llc -mtriple=x86_64-unknown-linux-gnu -global-isel %s -o %t.gisel.s
; RUN: llc -mtriple=x86_64-unknown-linux-gnu -enable-mlir-isel %s -o %t.mlir.s
; RUN: diff %t.gisel.s %t.mlir.s
;
; Verifies real translation for M2's supported subset: branches,
; conditional branches (gmir.brcond), integer comparisons (gmir.icmp), and
; PHI nodes -- represented as MLIR block arguments fed by gmir.br/brcond's
; successor operands, not a separate gmir.phi op (see
; llvm/lib/CodeGen/MLIR/GMIRImporter.cpp and MLIRToGMIRTranslator.cpp).
;
; Same oracle as scalar-arith.ll and for the same reason: -enable-mlir-isel
; and -global-isel should produce byte-identical output, since both emit
; equivalent generic MIR (including edge probabilities -- see
; MLIRToGMIRTranslator.cpp's BranchProbabilityInfo wiring, needed for
; MachineBlockPlacement's layout/alignment decisions, e.g. loop-header
; alignment, to match) through the same downstream Legalize/RegBankSelect/
; InstructionSelect/RegAlloc/BranchFolding pipeline.

define i32 @if_else(i32 %a, i32 %b) {
entry:
  %cmp = icmp sgt i32 %a, %b
  br i1 %cmp, label %then, label %else

then:
  %t = add i32 %a, 1
  br label %merge

else:
  %e = sub i32 %b, 1
  br label %merge

merge:
  %r = phi i32 [ %t, %then ], [ %e, %else ]
  ret i32 %r
}

define i32 @count_loop(i32 %n) {
entry:
  br label %loop

loop:
  %i = phi i32 [ 0, %entry ], [ %i.next, %loop ]
  %acc = phi i32 [ 0, %entry ], [ %acc.next, %loop ]
  %acc.next = add i32 %acc, %i
  %i.next = add i32 %i, 1
  %cond = icmp slt i32 %i.next, %n
  br i1 %cond, label %loop, label %exit

exit:
  ret i32 %acc.next
}

define i32 @multi_predecessor_phi(i32 %a, i32 %b, i32 %c) {
entry:
  %cmp1 = icmp eq i32 %a, 0
  br i1 %cmp1, label %case0, label %check1

check1:
  %cmp2 = icmp eq i32 %a, 1
  br i1 %cmp2, label %case1, label %default

case0:
  br label %join

case1:
  br label %join

default:
  br label %join

join:
  %r = phi i32 [ %b, %case0 ], [ %c, %case1 ], [ %a, %default ]
  ret i32 %r
}

; Regression test: both non-dominating arms reference the literal constant
; 42. LLVM interns/uniques ConstantInts, so `then` and `else` share the
; exact same llvm::Value* for it -- GMIRImporter.cpp's getOperands used to
; memoize the gmir.constant it materializes for a ConstantInt keyed purely
; by that shared llvm::Value*, at whichever block's insertion point
; happened to be current the first time it was needed. Since `then` and
; `else` here are siblings (neither dominates the other), reusing the
; memoized value from one in the other produced an SSA value that didn't
; dominate its use -- not caught until a downstream MachineFunction pass
; (LiveVariables: "Can't find reaching def for virtreg"), well past
; anything gmir/MLIR itself verifies. Fixed by always materializing
; memoized constants at a fixed, dominates-everything cursor at the front
; of the entry block, mirroring IRTranslator's own dedicated EntryBuilder.
define i32 @shared_constant_across_blocks(i32 %c, i32 %x) {
entry:
  %t = icmp eq i32 %c, 0
  br i1 %t, label %then, label %else

then:
  %a = add i32 %x, 42
  br label %join

else:
  %b = mul i32 %x, 42
  br label %join

join:
  %r = phi i32 [ %a, %then ], [ %b, %else ]
  ret i32 %r
}
