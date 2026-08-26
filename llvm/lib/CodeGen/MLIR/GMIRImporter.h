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
#include "llvm/ADT/DenseMap.h"

namespace llvm {
class CallInst;
class Function;

namespace gmir {

/// Maps each imported gmir.call Operation back to the original
/// llvm::CallInst it came from. MLIRToGMIRTranslator needs the original
/// CallInst to reuse CallLowering::lowerCall's CallBase-taking overload
/// (see gmir.call's doc comment in GMIRDialect.td) -- kept in a plain
/// caller-owned map rather than on the op itself (e.g. via
/// mlir::OpaqueLoc) so this per-call-site bookkeeping doesn't permanently
/// intern anything into the MLIRContext: the map lives and dies with the
/// caller's own per-MachineFunction state, while anything interned into
/// the Context lives for the whole pass instance's lifetime (see
/// MLIRInstructionSelect.cpp's Context doc comment).
using CallInstMap = llvm::DenseMap<mlir::Operation *, llvm::CallInst *>;

/// Imports F into a new mlir::func::FuncOp appended to Module, as `gmir`
/// ops, recording each imported gmir.call's original llvm::CallInst into
/// CallInsts (see CallInstMap's doc comment). Returns a null FuncOp --
/// without building anything further -- the moment F contains anything
/// outside the currently-supported subset (see GMIRImporter.cpp's
/// per-instruction dispatch for the exact list).
mlir::func::FuncOp importFunction(mlir::ModuleOp Module, llvm::Function &F,
                                  CallInstMap &CallInsts);

} // namespace gmir
} // namespace llvm

#endif // LLVM_CODEGEN_MLIR_GMIRIMPORTER_H
