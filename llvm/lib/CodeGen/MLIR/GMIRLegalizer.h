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
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "llvm/ADT/DenseMap.h"
#include <utility>

namespace llvm {
class DataLayout;
class LegalizerInfo;
class MachineFunction;

namespace gmir {

/// Caches the FrozenRewritePatternSet gmir::legalize() builds, keyed by
/// the (LegalizerInfo, DataLayout) pair the patterns were built to query
/// -- every function sharing one target *and* one Module (the normal
/// single-target-per-invocation case) reuses the same pattern set instead
/// of rebuilding/refreezing it per function. Meant to be owned as
/// long-lived state by the caller (e.g. a MachineFunctionPass member --
/// one pass instance per compilation thread/pipeline in any
/// parallel-codegen configuration, so this class needs no locking of its
/// own). Safe to cache across functions only because the patterns'
/// captured MLIRContext*/DataLayout&/LegalizerInfo* are themselves
/// already long-lived (see MLIRInstructionSelect.cpp: one MLIRContext per
/// pass instance, not one per function; DataLayout/LegalizerInfo are
/// owned by the Module/TargetSubtargetInfo, which outlive any single
/// MachineFunctionPass invocation). Keyed on *both* pointers, not just
/// LegalizerInfo: a single long-lived pass instance processing more than
/// one Module in sequence (not possible via plain `llc`, one Module per
/// invocation, but a real risk for any embedding -- e.g. a JIT/service --
/// that recycles one codegen pipeline across Modules sharing one
/// subtarget) could otherwise hit a cache entry whose patterns still
/// capture a *different*, possibly-already-destroyed Module's
/// DataLayout.
class LegalizerPatternCache {
public:
  const mlir::FrozenRewritePatternSet &get(mlir::MLIRContext &Context,
                                           const LegalizerInfo *LI,
                                           const llvm::DataLayout &DL);

private:
  llvm::DenseMap<std::pair<const LegalizerInfo *, const llvm::DataLayout *>,
                 mlir::FrozenRewritePatternSet>
      Cache;
};

/// Rewrites FuncOp in place. Returns false only on an unrecoverable error
/// (currently never -- ops this pass doesn't handle are simply left alone,
/// not treated as failure, since the downstream Legalizer catches them);
/// kept bool-returning to match GMIRImporter::importFunction/
/// gmir::translate's fallible-step convention used throughout this
/// pipeline's caller, MLIRInstructionSelect.cpp. PatternCache is owned by
/// the caller and reused across calls (see LegalizerPatternCache's doc).
bool legalize(mlir::func::FuncOp FuncOp, MachineFunction &MF,
              LegalizerPatternCache &PatternCache);

} // namespace gmir
} // namespace llvm

#endif // LLVM_CODEGEN_MLIR_GMIRLEGALIZER_H
