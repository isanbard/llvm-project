//===- GMIRImporter.h - LLVM IR -> gmir importer ---------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Imports a single-basic-block, scalar-integer-arithmetic-only
// llvm::Function directly into `gmir` ops, without going through MLIR's
// `llvm` dialect: mlir::translateLLVMIRToModule takes ownership of a whole
// llvm::Module for a one-shot translation, which doesn't fit a pass that
// runs per-function on a Module shared with the rest of the compilation.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_MLIR_GMIRIMPORTER_H
#define LLVM_CODEGEN_MLIR_GMIRIMPORTER_H

#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace llvm {
class Function;

namespace gmir {

/// Imports F's single basic block into a new mlir::func::FuncOp appended to
/// Module, as `gmir` ops. Returns a null FuncOp -- without building
/// anything further -- the moment F contains anything outside the
/// currently-supported subset: more than one basic block, a non-integer
/// type (or an integer wider than 64 bits), or an instruction opcode other
/// than add/sub/mul/sdiv/and/or/xor/ret.
mlir::func::FuncOp importFunction(mlir::ModuleOp Module, llvm::Function &F);

} // namespace gmir
} // namespace llvm

#endif // LLVM_CODEGEN_MLIR_GMIRIMPORTER_H
