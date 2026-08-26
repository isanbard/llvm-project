//===- GMIRImporter.h - LLVM IR -> gmir importer ---------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Imports an llvm::Function directly into `gmir` ops, without going
// through MLIR's `llvm` dialect: mlir::translateLLVMIRToModule takes
// ownership of a whole llvm::Module for a one-shot translation, which
// doesn't fit a pass that runs per-function on a Module shared with the
// rest of the compilation.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_MLIR_GMIRIMPORTER_H
#define LLVM_CODEGEN_MLIR_GMIRIMPORTER_H

#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace llvm {
class Function;

namespace gmir {

/// Imports F into a new mlir::func::FuncOp appended to Module, as `gmir`
/// ops. Returns a null FuncOp -- without building anything further -- the
/// moment F contains anything outside the currently-supported subset (see
/// GMIRImporter.cpp's per-instruction dispatch for the exact list).
mlir::func::FuncOp importFunction(mlir::ModuleOp Module, llvm::Function &F);

} // namespace gmir
} // namespace llvm

#endif // LLVM_CODEGEN_MLIR_GMIRIMPORTER_H
