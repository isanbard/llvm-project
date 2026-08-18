//===- GMIRCombiner.h - gmir-level algebraic combining ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Runs DAGCombiner-style algebraic simplification over `gmir` IR, after
// GMIRLegalizer.h and before MLIRToGMIRTranslator lowers it to real MIR.
// M5 slice 1's tier-1 algebraic identities (e.g. `x+0 -> x`) are
// implemented as the arithmetic ops' own fold() methods (see
// IR/GMIRDialect.cpp), not as patterns registered here -- fold() fires
// automatically wherever applyPatternsGreedily runs, which is all this
// pass needs to do to exercise them. This file's own job is (a)
// providing the pipeline slot those fold()s actually run in, (b) wiring
// in CSE (mlir::eliminateCommonSubExpressions), a distinct,
// complementary simplification fold() can't do on its own (deduping
// genuinely-different-looking-but-equivalent ops, not reducing a single
// op given its own operands), and (c), starting with M5 slice 2, hosting
// genuine multi-op rewrites fold() can't express -- no new ops, no
// restructuring -- as real OpRewritePattern<OpTy> classes registered
// into CombinerPatternCache, the same shape GMIRLegalizer.h uses (e.g.
// slice 2's MulNegOneToSubPattern: `mul x, -1 -> sub(0, x)`, which needs
// a brand-new gmir.sub op fold() can't emit). See
// ~/llvm/mlir_instruction_selection_plan.md's M5 section for the full
// design, including why -print-gmir-after-combine (not just the usual
// -global-isel asm-diff) is the real proof this pass's own code ran on
// AArch64 specifically (a real downstream GICombiner pass runs there
// even at -O0-equivalent, unlike X86).
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_MLIR_GMIRCOMBINER_H
#define LLVM_CODEGEN_MLIR_GMIRCOMBINER_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include <optional>

namespace llvm {
namespace gmir {

/// Caches the FrozenRewritePatternSet gmir::combine() applies, built once
/// and reused for the pass instance's lifetime. Unlike GMIRLegalizer.h's
/// LegalizerPatternCache, this needs no per-target keying: slice 1's
/// fold()-based identities have no LegalizerInfo/DataLayout dependency
/// at all, and a future slice's DAGCombiner rule that genuinely needs a
/// TargetLowering query is expected to call it directly from the
/// pattern body (the same way GMIRLegalizerInfoAdapter wraps
/// LegalizerInfo -- see design doc §1.20's decision #3), not by keying
/// the whole pattern set on it. Revisit this simplification if a future
/// slice's patterns turn out to need per-target-keyed caching after all.
class CombinerPatternCache {
  std::optional<mlir::FrozenRewritePatternSet> Cache;

public:
  const mlir::FrozenRewritePatternSet &get(mlir::MLIRContext &Context);
};

/// Rewrites FuncOp in place: applies PatternCache's patterns (plus, for
/// every op, its own fold() -- the greedy driver invokes fold()
/// unconditionally regardless of what's in the pattern set, see
/// IR/GMIRDialect.cpp) via applyPatternsGreedily, then runs CSE once.
/// Returns false only if applyPatternsGreedily itself fails (e.g.
/// non-convergence) -- mirrors gmir::legalize()'s fallible-step
/// convention used throughout this pipeline's caller,
/// MLIRInstructionSelect.cpp.
bool combine(mlir::func::FuncOp FuncOp, CombinerPatternCache &PatternCache);

} // namespace gmir
} // namespace llvm

#endif // LLVM_CODEGEN_MLIR_GMIRCOMBINER_H
