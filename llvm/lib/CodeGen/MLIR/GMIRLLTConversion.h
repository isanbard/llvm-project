//===- GMIRLLTConversion.h - !gmir.llt <-> LLT conversion -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Shared !gmir.llt <-> llvm::LLT conversion, used by both
// MLIRToGMIRTranslator.cpp (gmir -> MIR, needs LLT for MachineIRBuilder)
// and GMIRLegalizer.cpp (needs LLT to query MF's target LegalizerInfo, an
// LLT-based API, from gmir-level legalization patterns, then needs the
// inverse to build new gmir ops from the answer). Kept in one place so the
// mapping -- in particular the pointer-vs-integer LLT::Kind distinction
// documented on convertLLT below -- never drifts between the two
// directions.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_MLIR_GMIRLLTCONVERSION_H
#define LLVM_CODEGEN_MLIR_GMIRLLTCONVERSION_H

#include "IR/GMIRDialect.h"
#include "llvm/CodeGenTypes/LowLevelType.h"
#include "llvm/IR/DataLayout.h"

namespace llvm {
namespace gmir {

// scalarSizeInBits == 0 means "pointer in addressSpace" (see GMIRDialect.td's
// !gmir.llt doc comment) -- pointer size itself needs a DataLayout, hence
// the extra parameter.
//
// This checkout's LLT (LowLevelType.h) is a real fork-specific extension
// beyond upstream LLVM: it distinguishes Kind::INTEGER (LLT::integer(N),
// prints "iN") from a generic untyped Kind::ANY_SCALAR (LLT::scalar(N),
// prints "sN") -- upstream LLT has no such split. Using LLT::scalar()
// silently produces ANY_SCALAR operands that AArch64's InstructionSelect
// tablegen patterns don't match (they require isInteger() specifically)
// while X86's happened to be permissive enough not to care -- every gmir
// integer value must use LLT::integer(), not LLT::scalar().
inline LLT convertLLT(gmir::LLTType Ty, const llvm::DataLayout &DL) {
  if (Ty.getScalarSizeInBits() == 0)
    return LLT::pointer(Ty.getAddressSpace(),
                        DL.getPointerSizeInBits(Ty.getAddressSpace()));
  return LLT::integer(Ty.getScalarSizeInBits());
}

/// Inverse of convertLLT: builds the !gmir.llt a real LLT would round-trip
/// to. Used by GMIRLegalizer.cpp to turn a target LegalizerInfo's LLT-typed
/// answer (e.g. NarrowScalar's LegalizeActionStep::NewType) back into a
/// !gmir.llt for building new gmir ops. Only scalar/pointer LLTs occur here
/// (mirrors convertLLT's own scope) -- vectors are out of scope for the
/// whole gmir dialect so far.
inline gmir::LLTType convertToGMIRType(mlir::MLIRContext &Context, LLT Ty) {
  if (Ty.isPointer())
    return gmir::LLTType::get(&Context, /*scalarSizeInBits=*/0,
                              /*numElements=*/0, Ty.getAddressSpace(),
                              /*isScalable=*/false);
  return gmir::LLTType::get(&Context, Ty.getScalarSizeInBits(),
                            /*numElements=*/0, /*addressSpace=*/0,
                            /*isScalable=*/false);
}

} // namespace gmir
} // namespace llvm

#endif // LLVM_CODEGEN_MLIR_GMIRLLTCONVERSION_H
