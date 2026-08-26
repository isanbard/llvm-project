//===- GMIRCombiner.cpp - gmir-level algebraic combining ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "GMIRCombiner.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/CSE.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace llvm;

const mlir::FrozenRewritePatternSet &
gmir::CombinerPatternCache::get(mlir::MLIRContext &Context) {
  if (!Cache)
    Cache.emplace(mlir::RewritePatternSet(&Context));
  return *Cache;
}

bool gmir::combine(mlir::func::FuncOp FuncOp,
                   CombinerPatternCache &PatternCache) {
  mlir::MLIRContext *Context = FuncOp.getContext();
  const mlir::FrozenRewritePatternSet &Frozen = PatternCache.get(*Context);

  // Region simplification must stay off, same reasoning and same real
  // crash as gmir::legalize() -- see GMIRLegalizer.cpp's comment on this
  // exact config option. combine() runs in the identical pipeline
  // position relative to MLIRToGMIRTranslator (before translate), so the
  // same 1:1-block-correspondence hazard applies regardless of what
  // patterns (if any) are registered -- the driver's region
  // simplification is a structural side effect independent of pattern
  // content.
  mlir::GreedyRewriteConfig Config;
  Config.setRegionSimplificationLevel(
      mlir::GreedySimplifyRegionLevel::Disabled);
  if (failed(applyPatternsGreedily(FuncOp, Frozen, Config)))
    return false;

  // CSE is a separate, complementary simplification fold() can't do on
  // its own: it dedups genuinely-different-looking-but-equivalent ops
  // (e.g. the same gmir.add computed twice from different blocks),
  // rather than reducing a single op given its own operands' values.
  // Uses the free-function API (not mlir::createCSEPass()/PassManager)
  // to match this codebase's existing plain-function-call style --
  // MLIRInstructionSelect.cpp never constructs an mlir::PassManager.
  mlir::DominanceInfo DomInfo(FuncOp);
  mlir::IRRewriter Rewriter(Context);
  eliminateCommonSubExpressions(Rewriter, DomInfo, FuncOp.getOperation());

  return true;
}
