//===- llvm/CodeGen/MLIRISel.h - MLIR ISel seam ----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Declares the entry points for the experimental MLIR-based instruction
// selection pipeline (see llvm/lib/CodeGen/MLIR/). This header must stay
// free of any MLIR dependency: libLLVMCodeGen always compiles and links
// against it, whether or not MLIR ISel support was built in.
//
// The real implementation lives in a separate LLVMCodeGenMLIR component
// whose CMake entry point is invoked from llvm/tools/CMakeLists.txt (after
// MLIR's own CMake machinery has been loaded), not from llvm/lib/, because
// MLIR's TableGen macros and library targets are not available while
// llvm/lib/ itself is being configured. This split means libLLVMCodeGen
// cannot call directly into the real pass, so createMLIRInstructionSelectPass
// instead dispatches through a runtime-installed factory pointer, which
// InitializeMLIRISel() installs when the optional component is linked in.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_MLIRISEL_H
#define LLVM_CODEGEN_MLIRISEL_H

#include "llvm/Support/Compiler.h"

namespace llvm {

class MachineFunctionPass;

/// Factory signature for the real MLIR ISel pass, installed at runtime by
/// InitializeMLIRISel() when MLIR ISel support was built in.
using MLIRInstructionSelectFactory = MachineFunctionPass *(*)();

/// Installs \p Factory as the source of createMLIRInstructionSelectPass()'s
/// result. Called once by InitializeMLIRISel(); left unset (so
/// createMLIRInstructionSelectPass() keeps returning nullptr) in builds
/// without MLIR ISel support.
LLVM_ABI void setMLIRInstructionSelectFactory(MLIRInstructionSelectFactory Factory);

/// Returns a freshly constructed MLIR ISel pass, or nullptr if MLIR ISel
/// support wasn't built in (or InitializeMLIRISel() hasn't run yet).
LLVM_ABI MachineFunctionPass *createMLIRInstructionSelectPass();

/// Registers the real MLIR ISel pass factory. Defined only by the optional
/// LLVMCodeGenMLIR component, which is built and linked only when
/// LLVM_ENABLE_MLIR_ISEL is on -- callers must guard calls to this function
/// with `#if LLVM_ENABLE_MLIR_ISEL` (see llvm/Config/llvm-config.h) so that
/// builds without MLIR ISel never reference an undefined symbol.
LLVM_ABI void InitializeMLIRISel();

} // namespace llvm

#endif // LLVM_CODEGEN_MLIRISEL_H
