//===- MLIRToGMIRTranslator.h - gmir -> MIR bridge --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Translates a `gmir`-only mlir::func::FuncOp (as built by GMIRImporter)
// into real GlobalISel generic MIR in MF, delegating formal-argument and
// return lowering to the target's existing CallLowering -- mirroring
// GlobalISel::IRTranslator's structure, but reading from `gmir` ops
// instead of LLVM IR instructions.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_MLIR_MLIRTOGMIRTRANSLATOR_H
#define LLVM_CODEGEN_MLIR_MLIRTOGMIRTRANSLATOR_H

#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace llvm {
class BranchProbabilityInfo;
class Function;
class MachineFunction;
class MachineIRBuilder;

namespace gmir {

/// Translates FuncOp (built from F by importFunction) into MF's MachineIR.
/// F is needed for CallLowering::lowerFormalArguments's Function& parameter
/// and to re-fetch each `ret` instruction's operand (lowerReturn wants an
/// llvm::Value*, not an mlir::Value). BPI supplies edge probabilities for
/// gmir.br/gmir.brcond's successors (mirrors IRTranslator's
/// addSuccessorWithProb) so downstream passes like MachineBlockPlacement
/// make the same layout/alignment decisions (e.g. loop-header alignment)
/// GlobalISel's own pipeline would. MIRBuilder is caller-owned (already
/// MF-bound) rather than constructed here so the caller can choose a
/// CSE-enabled builder to match IRTranslator::translate's own choice: a
/// plain MachineIRBuilder measurably changes downstream GlobalISel
/// combiner passes' reassociation matching (reused unchanged from the
/// real pipeline), since their rules depend on structurally-clean,
/// deduplicated input the way IRTranslator always produces it -- e.g. a
/// multi-index GEP's constant-offset gmir.ptr_add chain selected a
/// different (but semantically equivalent) AArch64 addressing mode
/// without this. Returns false if the target has no CallLowering
/// implementation, or CallLowering itself declines -- the caller should
/// treat that the same as an unsupported import: fall back gracefully.
bool translate(mlir::func::FuncOp FuncOp, Function &F, MachineFunction &MF,
               const BranchProbabilityInfo &BPI, MachineIRBuilder &MIRBuilder);

} // namespace gmir
} // namespace llvm

#endif // LLVM_CODEGEN_MLIR_MLIRTOGMIRTRANSLATOR_H
