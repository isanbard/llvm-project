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
// The tier-1 algebraic identities (e.g. `x+0 -> x`) are implemented as the
// arithmetic ops' own fold() methods (see IR/GMIRDialect.cpp), not as
// patterns registered here -- fold() fires automatically wherever
// applyPatternsGreedily runs, which is all this pass needs to do to
// exercise them. This file's own job is just (a) providing the pipeline
// slot those fold()s actually run in, and (b) wiring in CSE
// (mlir::eliminateCommonSubExpressions), a distinct, complementary
// simplification fold() can't do on its own (deduping genuinely-
// different-looking-but-equivalent ops, not reducing a single op given
// its own operands). A rule needing a genuine multi-op rewrite (which
// fold() can't express -- no new ops, no restructuring) registers a real
// pattern into CombinerPatternCache instead, the same OpRewritePattern<OpTy>
// shape GMIRLegalizer.h uses.
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
/// LegalizerPatternCache, this needs no per-target keying: the tier-1
/// fold()-based identities have no LegalizerInfo/DataLayout dependency at
/// all, and a pattern that genuinely needs a TargetLowering query is
/// expected to call it directly from the pattern body (the same way
/// GMIRLegalizerInfoAdapter wraps LegalizerInfo), not by keying the whole
/// pattern set on it.
class CombinerPatternCache {
public:
  const mlir::FrozenRewritePatternSet &get(mlir::MLIRContext &Context);

private:
  std::optional<mlir::FrozenRewritePatternSet> Cache;
};

/// Rewrites FuncOp in place: applies PatternCache's patterns via
/// applyPatternsGreedily (this is what exercises the arithmetic ops' own
/// fold()s, see IR/GMIRDialect.cpp, regardless of whether any pattern is
/// actually registered), then runs CSE once. Returns false only if
/// applyPatternsGreedily itself fails (e.g. non-convergence) -- mirrors
/// gmir::legalize()'s fallible-step convention used throughout this
/// pipeline's caller, MLIRInstructionSelect.cpp.
bool combine(mlir::func::FuncOp FuncOp, CombinerPatternCache &PatternCache);

} // namespace gmir
} // namespace llvm

#endif // LLVM_CODEGEN_MLIR_GMIRCOMBINER_H
