; REQUIRES: llvm_enable_mlir_isel
; RUN: llc -mtriple=x86_64-unknown-linux-gnu -enable-mlir-isel \
; RUN:     -print-gmir-after-combine %s -o /dev/null 2>&1 | FileCheck --check-prefix=GMIR %s
; RUN: llc -mtriple=x86_64-unknown-linux-gnu -enable-mlir-isel %s -o - | FileCheck --check-prefix=ASM %s
;
; See combine-identities-aarch64.ll for the AArch64 variant. Unlike
; every prior M1-M4 test in this directory, this one does NOT use the
; usual -global-isel byte-diff oracle -- confirmed empirically (not
; assumed) that it's the wrong invariant here: X86PassConfig's real
; GICombiner passes (addPreLegalizeMachineIR/addPreRegBankSelect) are
; both gated on getOptLevel() != CodeGenOptLevel::None, so plain
; -global-isel at this opt level genuinely does NOT simplify e.g.
; `mul x, 1` or `and x, -1` -- it emits a real imull/andl. gmir's own
; new fold()-based combiner (see IR/GMIRDialect.cpp, GMIRCombiner.cpp)
; DOES simplify these, so -enable-mlir-isel's output correctly and
; intentionally diverges from plain -global-isel's for exactly the
; functions below that exercise a real simplification -- a genuine new
; optimization, not something M4's "reproduce what the safety net would
; do anyway" byte-diff discipline applies to. Verified two ways instead:
; the GMIR-prefixed FileCheck below (gmir-level proof the new fold()s
; ran, same -print-gmir-after-X discipline every M4 slice used) and the
; ASM-prefixed FileCheck (proof the simplification survives all the way
; to correct final codegen) -- mirroring vector-scalarize.ll's precedent
; of FileChecking final assembly directly when the usual oracle doesn't
; apply.

define i32 @add_zero(i32 %x) {
  %r = add i32 %x, 0
  ret i32 %r
}
; GMIR-LABEL: func.func @add_zero
; GMIR-NOT: gmir.add
; GMIR: return %arg0
; ASM-LABEL: add_zero:
; ASM-NOT: addl
; ASM: movl %edi, %eax

define i32 @add_zero_lhs(i32 %x) {
  %r = add i32 0, %x
  ret i32 %r
}
; GMIR-LABEL: func.func @add_zero_lhs
; GMIR-NOT: gmir.add
; GMIR: return %arg0
; ASM-LABEL: add_zero_lhs:
; ASM-NOT: addl
; ASM: movl %edi, %eax

define i32 @sub_self(i32 %x) {
  %r = sub i32 %x, %x
  ret i32 %r
}
; GMIR-LABEL: func.func @sub_self
; GMIR-NOT: gmir.sub
; GMIR: gmir.constant 0
; GMIR: return
; ASM-LABEL: sub_self:
; ASM-NOT: subl
; ASM: xorl %eax, %eax

define i32 @mul_zero(i32 %x) {
  %r = mul i32 %x, 0
  ret i32 %r
}
; GMIR-LABEL: func.func @mul_zero
; GMIR-NOT: gmir.mul
; GMIR: gmir.constant 0
; GMIR: return
; ASM-LABEL: mul_zero:
; ASM-NOT: imull
; ASM: xorl %eax, %eax

define i32 @mul_one(i32 %x) {
  %r = mul i32 1, %x
  ret i32 %r
}
; GMIR-LABEL: func.func @mul_one
; GMIR-NOT: gmir.mul
; GMIR: return %arg0
; ASM-LABEL: mul_one:
; ASM-NOT: imull
; ASM: movl %edi, %eax

define i32 @and_self(i32 %x) {
  %r = and i32 %x, %x
  ret i32 %r
}
; GMIR-LABEL: func.func @and_self
; GMIR-NOT: gmir.and
; GMIR: return %arg0
; ASM-LABEL: and_self:
; ASM-NOT: andl
; ASM: movl %edi, %eax

define i32 @and_allones(i32 %x) {
  %r = and i32 %x, -1
  ret i32 %r
}
; GMIR-LABEL: func.func @and_allones
; GMIR-NOT: gmir.and
; GMIR: return %arg0
; ASM-LABEL: and_allones:
; ASM-NOT: andl
; ASM: movl %edi, %eax

define i32 @and_zero(i32 %x) {
  %r = and i32 %x, 0
  ret i32 %r
}
; GMIR-LABEL: func.func @and_zero
; GMIR-NOT: gmir.and
; GMIR: gmir.constant 0
; GMIR: return
; ASM-LABEL: and_zero:
; ASM-NOT: andl
; ASM: xorl %eax, %eax

define i32 @or_self(i32 %x) {
  %r = or i32 %x, %x
  ret i32 %r
}
; GMIR-LABEL: func.func @or_self
; GMIR-NOT: gmir.or
; GMIR: return %arg0
; ASM-LABEL: or_self:
; ASM-NOT: orl
; ASM: movl %edi, %eax

define i32 @or_zero(i32 %x) {
  %r = or i32 %x, 0
  ret i32 %r
}
; GMIR-LABEL: func.func @or_zero
; GMIR-NOT: gmir.or
; GMIR: return %arg0
; ASM-LABEL: or_zero:
; ASM-NOT: orl
; ASM: movl %edi, %eax

define i32 @or_allones(i32 %x) {
  %r = or i32 %x, -1
  ret i32 %r
}
; GMIR-LABEL: func.func @or_allones
; GMIR-NOT: gmir.or
; GMIR: gmir.constant -1
; GMIR: return
; ASM-LABEL: or_allones:
; ASM-NOT: orl
; ASM: movl $-1, %eax

define i32 @xor_zero(i32 %x) {
  %r = xor i32 %x, 0
  ret i32 %r
}
; GMIR-LABEL: func.func @xor_zero
; GMIR-NOT: gmir.xor
; GMIR: return %arg0
; ASM-LABEL: xor_zero:
; ASM-NOT: xorl %eax, %edi
; ASM: movl %edi, %eax

define i32 @const_add(i32 %unused) {
  %r = add i32 3, 4
  ret i32 %r
}
; GMIR-LABEL: func.func @const_add
; GMIR-NOT: gmir.add
; GMIR: gmir.constant 7
; GMIR: return
; ASM-LABEL: const_add:
; ASM: movl $7, %eax

define i32 @const_add_overflow(i32 %unused) {
  %r = add i32 2147483647, 1
  ret i32 %r
}
; GMIR-LABEL: func.func @const_add_overflow
; GMIR-NOT: gmir.add
; GMIR: gmir.constant -2147483648
; GMIR: return
; ASM-LABEL: const_add_overflow:
; ASM: movl $-2147483648, %eax
