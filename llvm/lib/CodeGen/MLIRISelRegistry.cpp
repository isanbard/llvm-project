//===- MLIRISelRegistry.cpp - MLIR ISel factory-pointer seam -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file is always compiled into libLLVMCodeGen, independent of
// LLVM_ENABLE_MLIR_ISEL, and never includes any MLIR header. It exists so
// that TargetPassConfig.cpp can call createMLIRInstructionSelectPass()
// unconditionally: the real pass lives in a separate, optionally-built
// component (llvm/lib/CodeGen/MLIR/) that libLLVMCodeGen cannot link
// directly (see llvm/include/llvm/CodeGen/MLIRISel.h for why), so instead
// that component installs itself here via a runtime factory pointer.
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/MLIRISel.h"

using namespace llvm;

static MLIRInstructionSelectFactory MLIRISelFactory = nullptr;

void llvm::setMLIRInstructionSelectFactory(
    MLIRInstructionSelectFactory Factory) {
  MLIRISelFactory = Factory;
}

MachineFunctionPass *llvm::createMLIRInstructionSelectPass() {
  return MLIRISelFactory ? MLIRISelFactory() : nullptr;
}
