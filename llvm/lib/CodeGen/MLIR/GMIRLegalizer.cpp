//===- GMIRLegalizer.cpp - gmir-level legalization -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "GMIRLegalizer.h"
#include "GMIRLLTConversion.h"
#include "IR/GMIRDialect.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/CodeGen/GlobalISel/LegalizerInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/TargetSubtargetInfo.h"
#include <optional>
#include <utility>

using namespace llvm;

namespace {

/// Maps a gmir op type to the TargetOpcode::G_* it mirrors, for querying
/// the real target LegalizerInfo. Only specialized for the opcodes this
/// pass's patterns need; extend as later patterns add more.
template <typename OpTy> unsigned getGenericOpcode();
template <> unsigned getGenericOpcode<gmir::AddOp>() {
  return TargetOpcode::G_ADD;
}
template <> unsigned getGenericOpcode<gmir::SubOp>() {
  return TargetOpcode::G_SUB;
}
template <> unsigned getGenericOpcode<gmir::AndOp>() {
  return TargetOpcode::G_AND;
}
template <> unsigned getGenericOpcode<gmir::OrOp>() {
  return TargetOpcode::G_OR;
}
template <> unsigned getGenericOpcode<gmir::XorOp>() {
  return TargetOpcode::G_XOR;
}
template <> unsigned getGenericOpcode<gmir::MulOp>() {
  return TargetOpcode::G_MUL;
}

/// Thin wrapper around a target's LegalizerInfo, letting patterns below
/// ask "what does the target's *existing* legality rules say about this
/// gmir op" without gmir having to hand-port its own copy of those rules
/// (this component's whole point -- see GMIRLegalizer.h). Held by value
/// inside GMIRLegalizePatternBase (see below) so pattern instances can be
/// cached in LegalizerPatternCache across functions rather than rebuilt
/// from a short-lived MachineFunction& each time.
/// LI (MF.getSubtarget().getLegalizerInfo(), fetched by legalize() below)
/// is null for any target whose TargetSubtargetInfo doesn't override it
/// (the base class's default); every in-tree target that actually reaches
/// this pass via -enable-mlir-isel does (GMIRImporter/CallLowering already
/// assume real GlobalISel support), but -enable-mlir-isel itself has no
/// target allowlist, so treat a null LegalizerInfo the same as "nothing is
/// NarrowScalar" rather than crashing: getAction() degrades to always
/// reporting Legal, so every pattern below simply fails to match and
/// legalize() is a no-op, same graceful-fallback shape as every other
/// failure path in this pipeline.
class GMIRLegalizerInfoAdapter {
public:
  explicit GMIRLegalizerInfoAdapter(const LegalizerInfo *LI) : LI(LI) {}

  LegalizeActionStep getAction(unsigned Opcode, ArrayRef<LLT> Types) const {
    if (!LI)
      return LegalizeActionStep(LegalizeActions::Legal, 0, LLT{});
    return LI->getAction(LegalityQuery(Opcode, Types));
  }

private:
  const LegalizerInfo *LI;
};

} // namespace

/// Returns {NarrowTy, NumParts} if Step (the op's single, already-computed
/// LegalizeActionStep -- see each pattern's matchAndRewrite below, which
/// queries GMIRLegalizerInfoAdapter::getAction exactly once and dispatches
/// on the result rather than having each candidate action re-query it) is
/// an exact, leftover-free, purely-scalar NarrowScalar split; std::nullopt
/// otherwise (not NarrowScalar, the split isn't exact -- general
/// non-exact-multiple leftover handling is out of scope, see
/// GMIRLegalizer.h -- or either type is a vector). Shared by every
/// pattern's NarrowScalar dispatch case -- all need the identical "is this
/// an exact NarrowScalar split, and if so what to split into" check before
/// their per-op-family rewrite logic (carry-chained vs. independent-chunk
/// vs. schoolbook mul) diverges. Declared in GMIRLegalizer.h (not static)
/// so GMIRLegalizerTest.cpp can unit-test it directly: no in-tree target's
/// real LegalizerInfo currently produces a NarrowScalar step for a
/// vector-typed gmir op (every vector case in this pipeline goes through
/// FewerElements down to scalar elements first), so the vector-shape guard
/// below can't be exercised via a real `llc -enable-mlir-isel` invocation.
std::optional<std::pair<LLT, unsigned>>
gmir::getExactNarrowScalarSplit(LegalizeActionStep Step, LLT DstTy) {
  if (Step.Action != LegalizeActions::NarrowScalar)
    return std::nullopt;

  // NarrowScalar here means "split one scalar integer into N narrower
  // scalar limbs" (the schoolbook-style algorithms below) -- it does NOT
  // cover per-element width narrowing of a vector (e.g. <4 x s32> ->
  // <4 x s16>), which a target's LegalizerInfo can also report as
  // NarrowScalar. Bail on either side being a vector rather than silently
  // misreading a same-element-count, narrower-per-element step as an
  // N-way scalar limb split -- that would build a gmir.unmerge whose
  // piece count doesn't match the source's real element count and abort
  // in UnmergeOp::verify() instead of gracefully falling back.
  LLT NarrowTy = Step.NewType;
  if (DstTy.isVector() || NarrowTy.isVector())
    return std::nullopt;

  unsigned DstBits = DstTy.getSizeInBits();
  unsigned NarrowBits = NarrowTy.getSizeInBits();
  if (NarrowBits == 0 || DstBits % NarrowBits != 0)
    return std::nullopt;
  unsigned NumParts = DstBits / NarrowBits;
  if (NumParts < 2)
    return std::nullopt;
  return std::make_pair(NarrowTy, NumParts);
}

/// Unmerges Op's lhs/rhs operands into NumParts NarrowGTy-typed pieces
/// each. The common first step of every NarrowScalar pattern below --
/// they diverge only afterward, on how the per-chunk op sequence composes
/// (carry-chained vs. independent). Factored out so a future fix to this
/// step (e.g. general non-exact-multiple leftover handling, currently out
/// of scope -- see getExactNarrowScalarSplit) only needs to change one
/// place, not one per pattern. A plain (non-explicitly-specialized)
/// template, so `static` works directly here -- unlike
/// getGenericOpcode<OpTy>() above, whose *explicit* per-type
/// specializations can't independently take a storage-class specifier.
template <typename OpTy>
static std::pair<gmir::UnmergeOp, gmir::UnmergeOp>
unmergeNarrowOperands(mlir::PatternRewriter &Rewriter, OpTy Op,
                      mlir::Location Loc, gmir::LLTType NarrowGTy,
                      unsigned NumParts) {
  SmallVector<mlir::Type, 4> NarrowResultTypes(NumParts, NarrowGTy);
  auto LhsParts =
      gmir::UnmergeOp::create(Rewriter, Loc, NarrowResultTypes, Op.getLhs());
  auto RhsParts =
      gmir::UnmergeOp::create(Rewriter, Loc, NarrowResultTypes, Op.getRhs());
  return {LhsParts, RhsParts};
}

namespace {

/// Implements the single WidenScalar action shared verbatim by
/// `G_ADD`/`G_AND`/`G_MUL`/`G_OR`/`G_SUB`/`G_XOR` (LegalizerHelper.cpp's
/// widenScalar switch): any-extend both operands up to the wider legal
/// type, re-run Op there, then truncate the result back down. `G_ANYEXT`,
/// not `ZEXT`/`SEXT`, is correct for all six -- each one's low-N result
/// bits depend only on the low-N input bits, so the extended high bits'
/// actual value never affects the truncated result. Unlike the
/// NarrowScalar split above, there's no multi-piece bookkeeping at all:
/// exactly one any-extend per operand, one op, one truncate. Shared by
/// every pattern's WidenScalar dispatch case below (AddSubLegalizePattern,
/// BitwiseLegalizePattern, MulLegalizePattern) so the sequence is written
/// once rather than duplicated per op family.
template <typename OpTy>
void rewriteWidenScalar(OpTy Op, mlir::PatternRewriter &Rewriter, LLT WideTy) {
  auto DstGTy = cast<gmir::LLTType>(Op.getResult().getType());
  mlir::MLIRContext *Context = Rewriter.getContext();
  gmir::LLTType WideGTy = gmir::convertToGMIRType(*Context, WideTy);
  mlir::Location Loc = Op.getLoc();

  auto LhsWide = gmir::AnyExtOp::create(Rewriter, Loc, WideGTy, Op.getLhs());
  auto RhsWide = gmir::AnyExtOp::create(Rewriter, Loc, WideGTy, Op.getRhs());
  auto WideRes = OpTy::create(Rewriter, Loc, WideGTy, LhsWide.getResult(),
                              RhsWide.getResult());
  Rewriter.replaceOpWithNewOp<gmir::TruncOp>(Op, DstGTy, WideRes.getResult());
}

/// Shared constructor/member boilerplate for every gmir-level
/// legalization pattern below: each needs a GMIRLegalizerInfoAdapter and
/// a DataLayout reference, and an identical 3-line constructor
/// forwarding to OpRewritePattern<OpTy>. Factored into one base (rather
/// than duplicated across every pattern class) so a future change to what
/// shared state these patterns carry only needs one edit.
template <typename OpTy>
class GMIRLegalizePatternBase : public mlir::OpRewritePattern<OpTy> {
public:
  GMIRLegalizePatternBase(mlir::MLIRContext *Context,
                          GMIRLegalizerInfoAdapter Adapter,
                          const llvm::DataLayout &DL)
      : mlir::OpRewritePattern<OpTy>(Context), Adapter(Adapter), DL(DL) {}

protected:
  GMIRLegalizerInfoAdapter Adapter;
  const llvm::DataLayout &DL;
};

/// Dispatches on GMIRLegalizerInfoAdapter::getAction's single result for
/// `G_ADD`/`G_SUB`, querying the rule table exactly once per visit rather
/// than registering one independently-matching pattern per candidate
/// LegalizeAction (which would mean a redundant rule-table walk per
/// candidate every time the driver visits a gmir.add/gmir.sub). NarrowScalar
/// and WidenScalar are implemented; more actions are added to this same
/// dispatch as this pass grows.
///
/// NarrowScalar case implements LegalizerHelper::narrowScalarAddSub's
/// exact hi/lo+carry split algorithm, ported to build gmir ops instead of
/// real MIR: unmerge each operand into NumParts narrower pieces, chain
/// gmir.uaddo/gmir.uadde (or usubo/usube) across the pieces threading the
/// carry, then merge the per-piece results back into the original width.
/// No artifact-cleanup patterns are needed: the freshly emitted ops' own
/// further legalization (e.g. the i1 carry type) and any merge/unmerge
/// cancellation happen automatically via the same permanent downstream
/// target Legalizer pass, exactly as they would for real GlobalISel's own
/// narrowScalarAddSub output. WidenScalar case: see rewriteWidenScalar.
template <typename OpTy, typename CarryOOp, typename CarryEOp>
class AddSubLegalizePattern : public GMIRLegalizePatternBase<OpTy> {
  using Base = GMIRLegalizePatternBase<OpTy>;
  using Base::Adapter;
  using Base::DL;

public:
  using Base::Base;

  LogicalResult
  matchAndRewrite(OpTy Op, mlir::PatternRewriter &Rewriter) const override {
    auto DstGTy = cast<gmir::LLTType>(Op.getResult().getType());
    LLT DstTy = gmir::convertLLT(DstGTy, DL);
    LegalizeActionStep Step =
        Adapter.getAction(getGenericOpcode<OpTy>(), {DstTy});

    if (auto Split = gmir::getExactNarrowScalarSplit(Step, DstTy)) {
      auto [NarrowTy, NumParts] = *Split;
      mlir::MLIRContext *Context = Rewriter.getContext();
      gmir::LLTType NarrowGTy = gmir::convertToGMIRType(*Context, NarrowTy);
      gmir::LLTType CarryGTy =
          gmir::convertToGMIRType(*Context, LLT::integer(1));

      mlir::Location Loc = Op.getLoc();
      auto [LhsParts, RhsParts] =
          unmergeNarrowOperands(Rewriter, Op, Loc, NarrowGTy, NumParts);

      SmallVector<mlir::Value, 4> DstParts;
      mlir::Value CarryIn;
      for (unsigned I = 0; I != NumParts; ++I) {
        mlir::Value L = LhsParts.getDsts()[I];
        mlir::Value R = RhsParts.getDsts()[I];
        if (I == 0) {
          auto C = CarryOOp::create(Rewriter, Loc, NarrowGTy, CarryGTy, L, R);
          DstParts.push_back(C.getDst());
          CarryIn = C.getCarryOut();
        } else {
          auto C = CarryEOp::create(Rewriter, Loc, NarrowGTy, CarryGTy, L, R,
                                    CarryIn);
          DstParts.push_back(C.getDst());
          CarryIn = C.getCarryOut();
        }
      }
      Rewriter.replaceOpWithNewOp<gmir::MergeOp>(Op, DstGTy, DstParts);
      return success();
    }

    if (Step.Action == LegalizeActions::WidenScalar) {
      rewriteWidenScalar(Op, Rewriter, Step.NewType);
      return success();
    }

    return failure();
  }
};

/// Dispatches on GMIRLegalizerInfoAdapter::getAction's single result for
/// `G_AND`/`G_OR`/`G_XOR`, same one-getAction()-call-per-visit rationale
/// as AddSubLegalizePattern above. NarrowScalar and WidenScalar are
/// implemented.
///
/// NarrowScalar case implements LegalizerHelper::narrowScalarBasic's
/// algorithm: unlike add/sub there's no carry to thread between chunks --
/// each narrow chunk pair is independent, so OpTy is just reapplied to
/// every pair of unmerged pieces directly. WidenScalar case: see
/// rewriteWidenScalar.
template <typename OpTy>
class BitwiseLegalizePattern : public GMIRLegalizePatternBase<OpTy> {
  using Base = GMIRLegalizePatternBase<OpTy>;
  using Base::Adapter;
  using Base::DL;

public:
  using Base::Base;

  LogicalResult
  matchAndRewrite(OpTy Op, mlir::PatternRewriter &Rewriter) const override {
    auto DstGTy = cast<gmir::LLTType>(Op.getResult().getType());
    LLT DstTy = gmir::convertLLT(DstGTy, DL);
    LegalizeActionStep Step =
        Adapter.getAction(getGenericOpcode<OpTy>(), {DstTy});

    if (auto Split = gmir::getExactNarrowScalarSplit(Step, DstTy)) {
      auto [NarrowTy, NumParts] = *Split;
      mlir::MLIRContext *Context = Rewriter.getContext();
      gmir::LLTType NarrowGTy = gmir::convertToGMIRType(*Context, NarrowTy);

      mlir::Location Loc = Op.getLoc();
      auto [LhsParts, RhsParts] =
          unmergeNarrowOperands(Rewriter, Op, Loc, NarrowGTy, NumParts);

      SmallVector<mlir::Value, 4> DstParts;
      for (unsigned I = 0; I != NumParts; ++I) {
        auto Chunk = OpTy::create(Rewriter, Loc, NarrowGTy,
                                  LhsParts.getDsts()[I], RhsParts.getDsts()[I]);
        DstParts.push_back(Chunk.getResult());
      }
      Rewriter.replaceOpWithNewOp<gmir::MergeOp>(Op, DstGTy, DstParts);
      return success();
    }

    if (Step.Action == LegalizeActions::WidenScalar) {
      rewriteWidenScalar(Op, Rewriter, Step.NewType);
      return success();
    }

    return failure();
  }
};

/// Dispatches on GMIRLegalizerInfoAdapter::getAction's single result for
/// `G_MUL`, same one-getAction()-call-per-visit rationale as
/// AddSubLegalizePattern/BitwiseLegalizePattern above. Only the
/// WidenScalar case is implemented so far: schoolbook multiplication (the
/// NarrowScalar case) is a distinct algorithm from either add/sub's carry
/// chain or bitwise's independent chunks, and is added separately.
class MulLegalizePattern : public GMIRLegalizePatternBase<gmir::MulOp> {
  using Base = GMIRLegalizePatternBase<gmir::MulOp>;
  using Base::Adapter;
  using Base::DL;

public:
  using Base::Base;

  LogicalResult
  matchAndRewrite(gmir::MulOp Op,
                  mlir::PatternRewriter &Rewriter) const override {
    auto DstGTy = cast<gmir::LLTType>(Op.getResult().getType());
    LLT DstTy = gmir::convertLLT(DstGTy, DL);
    LegalizeActionStep Step =
        Adapter.getAction(getGenericOpcode<gmir::MulOp>(), {DstTy});

    if (Step.Action == LegalizeActions::WidenScalar) {
      rewriteWidenScalar(Op, Rewriter, Step.NewType);
      return success();
    }

    return failure();
  }
};

} // namespace

const mlir::FrozenRewritePatternSet &
gmir::LegalizerPatternCache::get(mlir::MLIRContext &Context,
                                 const LegalizerInfo *LI,
                                 const llvm::DataLayout &DL) {
  auto Key = std::make_pair(LI, &DL);
  auto It = Cache.find(Key);
  if (It != Cache.end())
    return It->second;

  // One dispatcher pattern per op (see AddSubLegalizePattern/
  // BitwiseLegalizePattern's doc comments): each queries
  // GMIRLegalizerInfoAdapter::getAction exactly once per visit and
  // dispatches internally on the result, instead of registering one
  // independently-matching pattern per candidate LegalizeAction.
  mlir::RewritePatternSet Patterns(&Context);
  Patterns
      .add<AddSubLegalizePattern<gmir::AddOp, gmir::UAddOOp, gmir::UAddEOp>>(
          &Context, GMIRLegalizerInfoAdapter(LI), DL);
  Patterns
      .add<AddSubLegalizePattern<gmir::SubOp, gmir::USubOOp, gmir::USubEOp>>(
          &Context, GMIRLegalizerInfoAdapter(LI), DL);
  Patterns.add<BitwiseLegalizePattern<gmir::AndOp>>(
      &Context, GMIRLegalizerInfoAdapter(LI), DL);
  Patterns.add<BitwiseLegalizePattern<gmir::OrOp>>(
      &Context, GMIRLegalizerInfoAdapter(LI), DL);
  Patterns.add<BitwiseLegalizePattern<gmir::XorOp>>(
      &Context, GMIRLegalizerInfoAdapter(LI), DL);
  Patterns.add<MulLegalizePattern>(&Context, GMIRLegalizerInfoAdapter(LI), DL);

  return Cache.try_emplace(Key, std::move(Patterns)).first->second;
}

bool gmir::legalize(mlir::func::FuncOp FuncOp, MachineFunction &MF,
                    LegalizerPatternCache &PatternCache) {
  mlir::MLIRContext *Context = FuncOp.getContext();
  const LegalizerInfo *LI = MF.getSubtarget().getLegalizerInfo();
  const llvm::DataLayout &DL = MF.getDataLayout();
  const mlir::FrozenRewritePatternSet &Frozen =
      PatternCache.get(*Context, LI, DL);

  // Region simplification must stay off: applyPatternsGreedily's default
  // config (GreedySimplifyRegionLevel::Aggressive) merges structurally-
  // identical sibling blocks and erases unreachable ones as a side effect
  // independent of whether any pattern above actually matched anything.
  // MLIRToGMIRTranslator.cpp's GMIRToMIRWalker::run() parallel-walks
  // FuncOp's blocks alongside the original llvm::Function's, assuming
  // exact 1:1 lockstep correspondence (guaranteed by GMIRImporter, but not
  // preserved by a block-merging pass run afterward) -- silently losing
  // that correspondence desyncs the walk and crashes lowerReturn's
  // block-terminator cast on perfectly valid input (e.g. two sibling
  // blocks computing the same value before a shared successor). This pass
  // only ever rewrites individual ops in place and never touches block
  // structure, so there's nothing for region simplification to legitimately
  // do here anyway.
  mlir::GreedyRewriteConfig Config;
  Config.setRegionSimplificationLevel(
      mlir::GreedySimplifyRegionLevel::Disabled);

  return succeeded(applyPatternsGreedily(FuncOp, Frozen, Config));
}
