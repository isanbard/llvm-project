; REQUIRES: llvm_enable_mlir_isel
; REQUIRES: aarch64-registered-target
; RUN: llc -mtriple=aarch64-unknown-linux-gnu -global-isel %s -o %t.gisel.s
; RUN: llc -mtriple=aarch64-unknown-linux-gnu -enable-mlir-isel %s -o %t.mlir.s
; RUN: diff %t.gisel.s %t.mlir.s
;
; AArch64 variant of control-flow.ll -- see that file for the full
; rationale (same content, different mtriple; kept as a separate file
; rather than extra RUN lines in the same file since lit's REQUIRES is
; file-global, not scoped to individual RUN lines).

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
