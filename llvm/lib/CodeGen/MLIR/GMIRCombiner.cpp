//===- GMIRCombiner.cpp - gmir-level algebraic combining ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "GMIRCombiner.h"
#include "GMIRLLTConversion.h"
#include "IR/GMIRDialect.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/CSE.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/CodeGen/ISDOpcodes.h"
#include "llvm/CodeGen/LowLevelTypeUtils.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/TargetLowering.h"
#include "llvm/CodeGen/TargetSubtargetInfo.h"
#include "llvm/IR/DataLayout.h"
#include <optional>

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

/// Wraps a target's real TargetLowering the same way GMIRLegalizer.cpp's
/// GMIRLegalizerInfoAdapter wraps LegalizerInfo, for patterns (starting
/// with DisjointAddToOrPattern below) that need a genuine cost/legality
/// query DAGCombiner-style, not just an op's own intrinsic semantics.
/// TLI (MF.getSubtarget().getTargetLowering(), fetched by combine()
/// below) is null only for a target whose TargetSubtargetInfo doesn't
/// override it -- unlike LegalizerInfo (only GlobalISel-supporting
/// targets implement it), every in-tree target overrides
/// getTargetLowering() (X86Subtarget.h, AArch64Subtarget.h,
/// RISCVSubtarget.h, etc.), but -enable-mlir-isel has no target
/// allowlist, so this still degrades gracefully to "always legal" on a
/// null TLI rather than crashing -- same graceful-fallback shape as
/// GMIRLegalizerInfoAdapter::getAction.
class GMIRTargetLoweringAdapter {
  const TargetLowering *TLI;
  const llvm::DataLayout &DL;

public:
  GMIRTargetLoweringAdapter(const TargetLowering *TLI,
                            const llvm::DataLayout &DL)
      : TLI(TLI), DL(DL) {}

  bool isOperationLegal(unsigned Opcode, gmir::LLTType Ty,
                        LLVMContext &Ctx) const {
    if (!TLI)
      return true;
    LLT RealTy = gmir::convertLLT(Ty, DL);
    return TLI->isOperationLegal(Opcode, getApproximateEVTForLLT(RealTy, Ctx));
  }
};

/// `add(and(a,C1), and(b,C2)) -> or(and(a,C1), and(b,C2))` when C1 and C2
/// are disjoint (DAGCombiner.cpp's visitADD, narrowed -- see
/// GMIRCombiner.h and design doc §1.23 for why: the real DAGCombiner
/// rule's `DAG.haveNoCommonBitsSet(N0, N1)` works for arbitrary N0/N1 via
/// a general KnownBits-style query gmir has no equivalent of; this
/// pattern only proves disjointness the purely structural way, from two
/// literal AND masks, which needs no new analysis infrastructure).
/// Genuinely needs TargetLowering, unlike every other M5 pattern so far:
/// the real rule only fires when introducing an OR is known not to
/// create an illegal op, `TLI.isOperationLegal(ISD::OR, VT)` --
/// GMIRTargetLoweringAdapter is this pattern's reason for existing.
class DisjointAddToOrPattern : public OpRewritePattern<gmir::AddOp> {
  GMIRTargetLoweringAdapter Adapter;
  LLVMContext &Ctx;

public:
  DisjointAddToOrPattern(MLIRContext *Context,
                         GMIRTargetLoweringAdapter Adapter, LLVMContext &Ctx)
      : OpRewritePattern(Context), Adapter(std::move(Adapter)), Ctx(Ctx) {}

  LogicalResult matchAndRewrite(gmir::AddOp Op,
                                PatternRewriter &Rewriter) const override {
    auto LHSAnd = Op.getLhs().getDefiningOp<gmir::AndOp>();
    auto RHSAnd = Op.getRhs().getDefiningOp<gmir::AndOp>();
    if (!LHSAnd || !RHSAnd)
      return failure();

    auto DstGTy = cast<gmir::LLTType>(Op.getResult().getType());
    unsigned Width = DstGTy.getScalarSizeInBits();

    // Extract each AND's constant mask regardless of operand order --
    // gmir.and is Commutative, but that trait alone doesn't canonicalize
    // operand order, same reasoning as MulNegOneToSubPattern above.
    auto getMask = [&](gmir::AndOp And) -> std::optional<APInt> {
      IntegerAttr C;
      if (matchPattern(And.getRhs(), m_Constant(&C)) ||
          matchPattern(And.getLhs(), m_Constant(&C)))
        return C.getValue().trunc(Width);
      return std::nullopt;
    };
    std::optional<APInt> C1 = getMask(LHSAnd);
    std::optional<APInt> C2 = getMask(RHSAnd);
    if (!C1 || !C2 || !(*C1 & *C2).isZero())
      return failure();

    if (!Adapter.isOperationLegal(ISD::OR, DstGTy, Ctx))
      return failure();

    Rewriter.replaceOpWithNewOp<gmir::OrOp>(Op, DstGTy, Op.getLhs(),
                                            Op.getRhs());
    return success();
  }
};
} // namespace

const FrozenRewritePatternSet &
gmir::CombinerPatternCache::get(MLIRContext &Context, const TargetLowering *TLI,
                                const llvm::DataLayout &DL, LLVMContext &Ctx) {
  auto Key = std::make_pair(TLI, &DL);
  auto It = Cache.find(Key);
  if (It != Cache.end())
    return It->second;

  RewritePatternSet Patterns(&Context);
  Patterns.add<MulNegOneToSubPattern>(&Context);
  Patterns.add<DisjointAddToOrPattern>(&Context,
                                       GMIRTargetLoweringAdapter(TLI, DL), Ctx);

  return Cache.try_emplace(Key, std::move(Patterns)).first->second;
}

bool gmir::combine(func::FuncOp FuncOp, MachineFunction &MF,
                   CombinerPatternCache &PatternCache) {
  MLIRContext *Context = FuncOp.getContext();
  const TargetLowering *TLI = MF.getSubtarget().getTargetLowering();
  const llvm::DataLayout &DL = MF.getDataLayout();
  LLVMContext &Ctx = MF.getFunction().getContext();
  const FrozenRewritePatternSet &Frozen =
      PatternCache.get(*Context, TLI, DL, Ctx);

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
