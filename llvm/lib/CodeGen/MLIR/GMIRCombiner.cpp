//===- GMIRCombiner.cpp - gmir-level algebraic combining ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "GMIRCombiner.h"
#include "IR/GMIRDialect.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/CSE.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace llvm;
using namespace mlir;

namespace {
/// `mul x, -1 -> sub(0, x)` (DAGCombiner.cpp's visitMUL, unconditional --
/// no TLI/legality/hasOneUse() gating). The first M5 slice needing a
/// genuine OpRewritePattern rather than a fold(): the result (`-x`)
/// isn't equal to either existing operand, so this has to emit a brand
/// new gmir.sub, which fold() can't do (it can only return an existing
/// operand Value or a compile-time-constant Attribute). Checks both
/// operand orders explicitly -- gmir.mul is Commutative, but that trait
/// alone doesn't canonicalize operand order the way DAGCombiner's own
/// N1IsConst pass does, same as every M5 slice 1 fold() already has to.
class MulNegOneToSubPattern : public OpRewritePattern<gmir::MulOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(gmir::MulOp Op,
                                PatternRewriter &Rewriter) const override {
    mlir::Value Other;
    IntegerAttr ConstAttr;
    if (matchPattern(Op.getRhs(), m_Constant(&ConstAttr)))
      Other = Op.getLhs();
    else if (matchPattern(Op.getLhs(), m_Constant(&ConstAttr)))
      Other = Op.getRhs();
    else
      return failure();

    // Truncate to the op's real width before checking all-ones --
    // gmir.constant's I64Attr is always 64-bit sign-extended regardless
    // of the op's real, possibly narrower, width (same width-
    // correctness discipline as M5 slice 1's fold() bodies).
    auto DstGTy = cast<gmir::LLTType>(Op.getResult().getType());
    unsigned Width = DstGTy.getScalarSizeInBits();
    if (!ConstAttr.getValue().trunc(Width).isAllOnes())
      return failure();

    Location Loc = Op.getLoc();
    auto ZeroAttr = IntegerAttr::get(
        mlir::IntegerType::get(Rewriter.getContext(), 64), APInt::getZero(64));
    auto Zero = gmir::ConstantOp::create(Rewriter, Loc, DstGTy, ZeroAttr);
    Rewriter.replaceOpWithNewOp<gmir::SubOp>(Op, DstGTy, Zero.getResult(),
                                             Other);
    return success();
  }
};
} // namespace

const FrozenRewritePatternSet &
gmir::CombinerPatternCache::get(MLIRContext &Context) {
  if (!Cache) {
    RewritePatternSet Patterns(&Context);
    Patterns.add<MulNegOneToSubPattern>(&Context);
    Cache.emplace(std::move(Patterns));
  }

  return *Cache;
}

bool gmir::combine(func::FuncOp FuncOp, CombinerPatternCache &PatternCache) {
  MLIRContext *Context = FuncOp.getContext();
  const FrozenRewritePatternSet &Frozen = PatternCache.get(*Context);

  // Region simplification must stay off, same reasoning and same real
  // crash as gmir::legalize() -- see GMIRLegalizer.cpp's comment on this
  // exact config option. combine() runs in the identical pipeline
  // position relative to MLIRToGMIRTranslator (before translate), so the
  // same 1:1-block-correspondence hazard applies regardless of what
  // patterns (if any) are registered -- the driver's region
  // simplification is a structural side effect independent of pattern
  // content.
  GreedyRewriteConfig Config;
  Config.setRegionSimplificationLevel(GreedySimplifyRegionLevel::Disabled);
  if (failed(applyPatternsGreedily(FuncOp, Frozen, Config)))
    return false;

  // CSE is a separate, complementary simplification fold() can't do on
  // its own: it dedups genuinely-different-looking-but-equivalent ops
  // (e.g. the same gmir.add computed twice from different blocks),
  // rather than reducing a single op given its own operands' values.
  // Uses the free-function API (not mlir::createCSEPass()/PassManager)
  // to match this codebase's existing plain-function-call style --
  // MLIRInstructionSelect.cpp never constructs an mlir::PassManager.
  DominanceInfo DomInfo(FuncOp);
  IRRewriter Rewriter(Context);
  eliminateCommonSubExpressions(Rewriter, DomInfo, FuncOp.getOperation());

  return true;
}
