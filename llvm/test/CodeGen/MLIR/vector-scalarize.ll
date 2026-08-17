; REQUIRES: llvm_enable_mlir_isel
;
; Unlike every other M4 test, this one does NOT diff against `-global-isel`
; output: the oracle itself crashes on this exact input. `mul <16 x i8>`
; correctly resolves to FewerElements/scalarize on X86 (no packed byte-lane
; multiply instruction exists, confirmed via X86LegalizerInfo.cpp's G_MUL
; rule chain -- it never mentions an s8 element type), but the real target
; Legalizer's own G_BUILD_VECTOR (which it creates to reassemble the
; scalarized lanes) has no legality rule on X86, and the Legalizer aborts
; trying to legalize its own output:
;   LLVM ERROR: unable to legalize instruction: %2:_(<16 x s8>) =
;   G_BUILD_VECTOR %35:_(s8), ...
; Confirmed via a direct `llc -global-isel` repro with zero gmir
; involvement -- filed upstream as
; https://github.com/llvm/llvm-project/issues/216655. This pipeline's own
; GMIRLegalizer produces the identical gmir.unmerge/mul/build_vector shape
; (see the -print-gmir-after-legalize RUN line below), which
; MLIRToGMIRTranslator lowers into the exact same
; G_UNMERGE_VALUES/G_MUL/G_BUILD_VECTOR MIR shape real GlobalISel's own
; scalarize action would produce -- and the shared downstream
; Legalizer/Combiner/RegBankSelect/InstructionSelect passes (unchanged,
; identical to what -global-isel uses) turn *that* into correct code, the
; standard SSE2 byte-multiply idiom (widen each byte to a word, multiply,
; mask, pack back down). So instead of the usual asm-diff regression net,
; the RUN line below FileChecks the final assembly directly for that
; idiom's key instructions.
;
; RUN: llc -mtriple=x86_64-unknown-linux-gnu -enable-mlir-isel %s -o - | FileCheck %s --check-prefix=ASM
;
; RUN: llc -mtriple=x86_64-unknown-linux-gnu -enable-mlir-isel \
; RUN:     -print-gmir-after-legalize %s -o /dev/null 2>&1 | FileCheck %s
;
; This is the actual proof that GMIRLegalizer's own code ran (same
; reasoning as every other M4 test): FewerElementsScalarizePattern
; (GMIRLegalizer.cpp) implements LegalizerHelper::
; fewerElementsVectorMultiEltType's pure-scalarize algorithm --
; gmir.unmerge (x2, splitting each <16 x i8> operand into 16 scalar
; lanes) -> gmir.mul (x16, one per lane) -> gmir.build_vector
; (reassembling the lanes) -- these three op *shapes* (unmerge with a
; vector source, scalar mul, build_vector) only ever come from this
; pattern, never from GMIRImporter directly, so their presence here is
; unambiguous evidence the new legalizer code path ran.

define <16 x i8> @vecmul16(<16 x i8> %a, <16 x i8> %b) {
  %r = mul <16 x i8> %a, %b
  ret <16 x i8> %r
}
; CHECK-LABEL: func.func @vecmul16
; CHECK: "gmir.unmerge"
; CHECK: "gmir.unmerge"
; CHECK: gmir.mul
; CHECK: "gmir.build_vector"

; ASM-LABEL: vecmul16:
; ASM: pmullw
; ASM: pand
; ASM: pmullw
; ASM: pand
; ASM: packuswb
; ASM: retq
