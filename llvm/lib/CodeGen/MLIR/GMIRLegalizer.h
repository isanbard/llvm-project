//===- GMIRLegalizer.h - gmir-level legalization ---------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Rewrites `gmir` IR in place so it satisfies (a subset of) the target's
// real LegalizerInfo rules, before MLIRToGMIRTranslator lowers it to real
// MIR. Queries MF's *existing* target LegalizerInfo rather than a
// hand-ported gmir-level rule table -- avoids re-deriving the ~200-300
// legality rules per target that GlobalISel's own Legalizer relies on. Ops
// this pass doesn't recognize, or whose LegalizeAction it doesn't yet
// implement, are left untouched: the real target Legalizer
// MachineFunctionPass runs unconditionally later in the pipeline (see
// TargetPassConfig.cpp's MLIRISel branch) and catches everything left
// over -- the same safety net that already exists for Custom-resolving
// ops. See ~/llvm/mlir_instruction_selection_plan.md's M4 section for the
// full design, including why this makes the usual -global-isel asm-diff
// test alone insufficient proof that this pass's own code actually ran (a
// -print-gmir-after-legalize FileCheck test is the real proof).
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_MLIR_GMIRLEGALIZER_H
#define LLVM_CODEGEN_MLIR_GMIRLEGALIZER_H

#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace llvm {
class MachineFunction;

namespace gmir {

/// Rewrites FuncOp in place. Returns false only on an unrecoverable error
/// (currently never -- ops this pass doesn't handle are simply left alone,
/// not treated as failure, since the downstream Legalizer catches them);
/// kept bool-returning to match GMIRImporter::importFunction/
/// gmir::translate's fallible-step convention used throughout this
/// pipeline's caller, MLIRInstructionSelect.cpp.
bool legalize(mlir::func::FuncOp FuncOp, MachineFunction &MF);

} // namespace gmir
} // namespace llvm

#endif // LLVM_CODEGEN_MLIR_GMIRLEGALIZER_H
