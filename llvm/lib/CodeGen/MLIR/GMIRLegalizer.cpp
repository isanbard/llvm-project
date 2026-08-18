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
using namespace mlir;

namespace {

/// Maps a gmir op type to the TargetOpcode::G_* it mirrors, for querying
/// the real target LegalizerInfo. Only specialized for the opcodes this
/// slice's patterns need; extend as later slices add more.
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
/// (this milestone's whole point -- see GMIRLegalizer.h). Held by value
/// inside GMIRLegalizePatternBase (see below) so pattern instances can
/// be cached in LegalizerPatternCache across functions rather than
/// rebuilt from a short-lived MachineFunction& each time.
/// LI (MF.getSubtarget().getLegalizerInfo(), fetched by legalize() below)
/// is null for any target whose TargetSubtargetInfo doesn't override it
/// (the base class's default); every in-tree target that actually reaches
/// this pass via -enable-mlir-isel does (GMIRImporter/CallLowering
/// already assume real GlobalISel support), but -enable-mlir-isel itself
/// has no target allowlist, so treat a null LegalizerInfo the same as
/// "nothing is NarrowScalar" rather than crashing: getAction() degrades
/// to always reporting Legal, so every pattern below simply fails to
/// match and legalize() is a no-op, same graceful-fallback shape as every
/// other failure path in this pipeline.
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

/// Returns {NarrowTy, NumParts} if Step (the op's single, already-computed
/// LegalizeActionStep -- see each merged pattern's matchAndRewrite below,
/// which queries GMIRLegalizerInfoAdapter::getAction exactly once and
/// dispatches on the result rather than having each candidate action
/// re-query it) is an exact, leftover-free NarrowScalar split;
/// std::nullopt otherwise (not NarrowScalar, or the split isn't exact --
/// general non-exact-multiple leftover handling is out of scope, see
/// GMIRLegalizer.h). Shared by every merged pattern's NarrowScalar
/// dispatch case -- all need the identical "is this an exact NarrowScalar
/// split, and if so what to split into" check before their per-op-family
/// rewrite logic (carry-chained vs. independent-chunk vs. schoolbook mul)
/// diverges.
std::optional<std::pair<LLT, unsigned>>
getExactNarrowScalarSplit(LegalizeActionStep Step, LLT DstTy) {
  if (Step.Action != LegalizeActions::NarrowScalar)
    return std::nullopt;

  LLT NarrowTy = Step.NewType;
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
/// each. The common first step of AddSubLegalizePattern's,
/// BitwiseLegalizePattern's, and MulLegalizePattern's NarrowScalar/
/// FewerElements dispatch cases -- they diverge only afterward, on how
/// the per-piece op sequence composes (carry-chained, independent-chunk,
/// schoolbook mul, or independent-lane) and on what NarrowGTy/NumParts
/// mean (narrower bit-width chunks for the scalar cases, vector lanes for
/// the FewerElements case -- gmir.unmerge's verifier distinguishes the
/// two by the source operand's own shape, see GMIRDialect.cpp). Factored
/// out so a future fix to this step (e.g. general non-exact-multiple
/// leftover handling, currently out of scope -- see
/// getExactNarrowScalarSplit) only needs to change one place, not one per
/// pattern.
template <typename OpTy>
std::pair<gmir::UnmergeOp, gmir::UnmergeOp>
unmergeNarrowOperands(PatternRewriter &Rewriter, OpTy Op, Location Loc,
                      gmir::LLTType NarrowGTy, unsigned NumParts) {
  SmallVector<mlir::Type, 4> NarrowResultTypes(NumParts, NarrowGTy);
  auto LhsParts =
      gmir::UnmergeOp::create(Rewriter, Loc, NarrowResultTypes, Op.getLhs());
  auto RhsParts =
      gmir::UnmergeOp::create(Rewriter, Loc, NarrowResultTypes, Op.getRhs());
  return {LhsParts, RhsParts};
}

/// Shared constructor/member boilerplate for every gmir-level
/// legalization pattern below: each needs a GMIRLegalizerInfoAdapter and
/// a DataLayout reference, and an identical 3-line constructor
/// forwarding to OpRewritePattern<OpTy>. Factored into one base (rather
/// than left duplicated across all 3 merged pattern classes) so a future
/// change to what shared state these patterns carry only needs one edit.
template <typename OpTy>
class GMIRLegalizePatternBase : public OpRewritePattern<OpTy> {
public:
  GMIRLegalizePatternBase(MLIRContext *Context,
                          GMIRLegalizerInfoAdapter Adapter,
                          const llvm::DataLayout &DL)
      : OpRewritePattern<OpTy>(Context), Adapter(Adapter), DL(DL) {}

protected:
  GMIRLegalizerInfoAdapter Adapter;
  const llvm::DataLayout &DL;
};

/// Shared WidenScalar rewrite body for `G_ADD`/`G_AND`/`G_MUL`/`G_OR`/
/// `G_XOR`/`G_SUB` (LegalizerHelper.cpp's widenScalar switch: any-extend
/// both operands to WideTy, perform OpTy there, truncate the result back
/// down). `G_ANYEXT`, not `ZEXT`/`SEXT`, is correct for all six of these
/// ops -- each one's low-N result bits depend only on the low-N input
/// bits, so the extended high bits' actual value never affects the
/// truncated result. A plain function, not its own pattern: every op
/// family's merged pattern below (AddSubLegalizePattern,
/// BitwiseLegalizePattern, MulLegalizePattern) calls this once its single
/// getAction() query resolves to WidenScalar, instead of WidenScalar
/// being handled by a second, independently-matching pattern that would
/// re-query the same LegalizerInfo rule table a second time for the same
/// op.
template <typename OpTy>
void rewriteWidenScalar(OpTy Op, PatternRewriter &Rewriter, LLT WideTy) {
  auto DstGTy = cast<gmir::LLTType>(Op.getResult().getType());
  MLIRContext *Context = Rewriter.getContext();
  gmir::LLTType WideGTy = gmir::convertToGMIRType(*Context, WideTy);
  Location Loc = Op.getLoc();

  auto LhsWide = gmir::AnyExtOp::create(Rewriter, Loc, WideGTy, Op.getLhs());
  auto RhsWide = gmir::AnyExtOp::create(Rewriter, Loc, WideGTy, Op.getRhs());
  auto WideRes = OpTy::create(Rewriter, Loc, WideGTy, LhsWide.getResult(),
                              RhsWide.getResult());
  Rewriter.replaceOpWithNewOp<gmir::TruncOp>(Op, DstGTy, WideRes.getResult());
}

/// Shared FewerElements rewrite body -- originally MulLegalizePattern-only
/// (M4 slice 4's mul-scalarize scenario was the only one with a concrete
/// motivating case), generalized here to every op family's merged pattern
/// once it became clear the scalarize algorithm itself has nothing
/// mul-specific about it: unmerge each vector operand into its scalar
/// lanes, reapply OpTy per lane, reassemble via gmir.build_vector. Returns
/// false (leaving Op untouched) when Step isn't a pure-scalarize
/// FewerElements action -- see MulLegalizePattern's doc comment for why a
/// still-vector Step.NewType means this is FewerElements's other flavor
/// (sub-vector splitting), out of scope here same as before.
template <typename OpTy>
bool rewriteFewerElements(OpTy Op, PatternRewriter &Rewriter,
                          LegalizeActionStep Step, LLT DstTy,
                          gmir::LLTType DstGTy) {
  if (Step.Action != LegalizeActions::FewerElements || Step.NewType.isVector())
    return false;

  MLIRContext *Context = Rewriter.getContext();
  gmir::LLTType LaneGTy = gmir::convertToGMIRType(*Context, Step.NewType);
  unsigned NumLanes = DstTy.getNumElements();
  Location Loc = Op.getLoc();
  auto [LhsParts, RhsParts] =
      unmergeNarrowOperands(Rewriter, Op, Loc, LaneGTy, NumLanes);

  SmallVector<mlir::Value, 4> DstParts;
  for (unsigned I = 0; I != NumLanes; ++I) {
    auto Lane = OpTy::create(Rewriter, Loc, LaneGTy, LhsParts.getDsts()[I],
                             RhsParts.getDsts()[I]);
    DstParts.push_back(Lane.getResult());
  }

  Rewriter.replaceOpWithNewOp<gmir::BuildVectorOp>(Op, DstGTy, DstParts);
  return true;
}

/// Merges what used to be two separately-registered, separately-matching
/// patterns -- NarrowScalar (LegalizerHelper::narrowScalarAddSub's exact
/// hi/lo+carry split algorithm, ported to build gmir ops instead of real
/// MIR) and WidenScalar (see rewriteWidenScalar) -- for `G_ADD`/`G_SUB`
/// into one pattern per op. Queries GMIRLegalizerInfoAdapter::getAction
/// exactly once per visit and dispatches on the result, instead of paying
/// for two redundant rule-table walks (one per formerly-separate pattern)
/// every time the driver visits a gmir.add/gmir.sub.
///
/// NarrowScalar case: unmerge each operand into NumParts narrower pieces,
/// chain gmir.uaddo/gmir.uadde (or usubo/usube) across the pieces
/// threading the carry, then merge the per-piece results back into the
/// original width. Bails on any split that isn't an exact multiple --
/// confirmed unreachable for i686's i64-add/sub scenario (64/32 == 2
/// exactly); general leftover handling is explicitly out of scope (see
/// GMIRLegalizer.h). No artifact-cleanup patterns are needed: the freshly
/// emitted ops' own further legalization (e.g. the i1 carry type) and any
/// merge/unmerge cancellation happen automatically via the same permanent
/// downstream target Legalizer pass, exactly as they would for real
/// GlobalISel's own narrowScalarAddSub output.
template <typename OpTy, typename CarryOOp, typename CarryEOp>
class AddSubLegalizePattern : public GMIRLegalizePatternBase<OpTy> {
  using Base = GMIRLegalizePatternBase<OpTy>;
  using Base::Adapter;
  using Base::DL;

public:
  using Base::Base;

  LogicalResult matchAndRewrite(OpTy Op,
                                PatternRewriter &Rewriter) const override {
    auto DstGTy = cast<gmir::LLTType>(Op.getResult().getType());
    LLT DstTy = gmir::convertLLT(DstGTy, DL);
    LegalizeActionStep Step =
        Adapter.getAction(getGenericOpcode<OpTy>(), {DstTy});

    if (auto Split = getExactNarrowScalarSplit(Step, DstTy)) {
      auto [NarrowTy, NumParts] = *Split;
      MLIRContext *Context = Rewriter.getContext();
      gmir::LLTType NarrowGTy = gmir::convertToGMIRType(*Context, NarrowTy);
      gmir::LLTType CarryGTy =
          gmir::convertToGMIRType(*Context, LLT::integer(1));

      Location Loc = Op.getLoc();
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

    if (rewriteFewerElements(Op, Rewriter, Step, DstTy, DstGTy))
      return success();

    return failure();
  }
};

/// Merges NarrowScalar (LegalizerHelper::narrowScalarBasic's algorithm for
/// `G_AND`/`G_OR`/`G_XOR`: unlike add/sub there's no carry to thread
/// between chunks -- each narrow chunk pair is independent, so OpTy is
/// just reapplied to every pair of unmerged pieces directly) and
/// WidenScalar (see rewriteWidenScalar) into one pattern per op, same
/// one-getAction()-call-per-visit rationale as AddSubLegalizePattern
/// above.
template <typename OpTy>
class BitwiseLegalizePattern : public GMIRLegalizePatternBase<OpTy> {
  using Base = GMIRLegalizePatternBase<OpTy>;
  using Base::Adapter;
  using Base::DL;

public:
  using Base::Base;

  LogicalResult matchAndRewrite(OpTy Op,
                                PatternRewriter &Rewriter) const override {
    auto DstGTy = cast<gmir::LLTType>(Op.getResult().getType());
    LLT DstTy = gmir::convertLLT(DstGTy, DL);
    LegalizeActionStep Step =
        Adapter.getAction(getGenericOpcode<OpTy>(), {DstTy});

    if (auto Split = getExactNarrowScalarSplit(Step, DstTy)) {
      auto [NarrowTy, NumParts] = *Split;
      MLIRContext *Context = Rewriter.getContext();
      gmir::LLTType NarrowGTy = gmir::convertToGMIRType(*Context, NarrowTy);

      Location Loc = Op.getLoc();
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

    if (rewriteFewerElements(Op, Rewriter, Step, DstTy, DstGTy))
      return success();

    return failure();
  }
};

/// Merges three formerly-separate patterns -- NarrowScalar, WidenScalar
/// (see rewriteWidenScalar), and FewerElements (see rewriteFewerElements)
/// -- for `gmir.mul` into one pattern, same one-getAction()-call-per-visit
/// rationale as AddSubLegalizePattern/BitwiseLegalizePattern above, but
/// for mul this collapses three redundant rule-table walks into one
/// instead of two.
///
/// NarrowScalar case implements LegalizerHelper::narrowScalarMul/
/// multiplyRegisters's algorithm, but only the NumParts == 2 case: at
/// exactly 2 limbs, multiplyRegisters's loop only ever executes its
/// last-limb branch once, needing no carry-propagation op at all -- just
/// the schoolbook 3-multiply/2-add shape: Lo = ALo*BLo (kept as-is, mod
/// 2^NarrowBits); Hi = umulh(ALo,BLo) + ALo*BHi + AHi*BLo (each cross
/// term's own overflow beyond NarrowBits is discarded, matching plain
/// integer multiplication's mod-2^64 semantics for the full result).
/// Every NumParts==2 case is empirically confirmed to be the only one
/// gmir.mul's NarrowScalar action ever produces today (i686's G_MUL
/// clamps s64 straight to s32, a single 2-limb split) -- but that's a
/// fact about the one target rule this has been checked against, not a
/// structural guarantee from GMIRImporter's 64-bit integer cap (which
/// bounds the *source* type's width, not what step size a target's
/// LegalizerInfo::getAction chooses to narrow by). matchAndRewrite below
/// bails explicitly on any other NumParts rather than assuming this can't
/// happen.
///
/// FewerElements case: see rewriteFewerElements, shared with
/// AddSubLegalizePattern/BitwiseLegalizePattern above -- originally
/// mul-only (mirroring LegalizerHelper::fewerElementsVectorMultiEltType's
/// pure-scalarize case, LegalizerHelper.cpp:5243-5310, reached via the
/// G_MUL case block at 5691-5816), generalized once it became clear the
/// scalarize algorithm has nothing mul-specific about it.
class MulLegalizePattern : public GMIRLegalizePatternBase<gmir::MulOp> {
  using Base = GMIRLegalizePatternBase<gmir::MulOp>;
  using Base::Adapter;
  using Base::DL;

public:
  using Base::Base;

  LogicalResult matchAndRewrite(gmir::MulOp Op,
                                PatternRewriter &Rewriter) const override {
    auto DstGTy = cast<gmir::LLTType>(Op.getResult().getType());
    LLT DstTy = gmir::convertLLT(DstGTy, DL);
    LegalizeActionStep Step =
        Adapter.getAction(getGenericOpcode<gmir::MulOp>(), {DstTy});

    if (auto Split = getExactNarrowScalarSplit(Step, DstTy)) {
      auto [NarrowTy, NumParts] = *Split;

      // Only the 2-limb case is implemented (see the class doc comment
      // for why NumParts other than 2 is a real, if so-far-unobserved,
      // possibility rather than something the importer's type cap rules
      // out) -- bail explicitly rather than mishandle it, same
      // fail-closed-to-the-real-downstream-Legalizer discipline as every
      // other unhandled case in this file.
      if (NumParts != 2)
        return failure();

      MLIRContext *Context = Rewriter.getContext();
      gmir::LLTType NarrowGTy = gmir::convertToGMIRType(*Context, NarrowTy);
      Location Loc = Op.getLoc();
      auto [LhsParts, RhsParts] =
          unmergeNarrowOperands(Rewriter, Op, Loc, NarrowGTy, NumParts);
      mlir::Value ALo = LhsParts.getDsts()[0];
      mlir::Value AHi = LhsParts.getDsts()[1];
      mlir::Value BLo = RhsParts.getDsts()[0];
      mlir::Value BHi = RhsParts.getDsts()[1];

      auto Lo = gmir::MulOp::create(Rewriter, Loc, NarrowGTy, ALo, BLo);
      auto HiHi = gmir::UMulHOp::create(Rewriter, Loc, NarrowGTy, ALo, BLo);
      auto LoHi = gmir::MulOp::create(Rewriter, Loc, NarrowGTy, ALo, BHi);
      auto HiLo = gmir::MulOp::create(Rewriter, Loc, NarrowGTy, AHi, BLo);
      auto Hi0 = gmir::AddOp::create(Rewriter, Loc, NarrowGTy, HiHi.getResult(),
                                     LoHi.getResult());
      auto Hi = gmir::AddOp::create(Rewriter, Loc, NarrowGTy, Hi0.getResult(),
                                    HiLo.getResult());

      SmallVector<mlir::Value, 2> DstParts{Lo.getResult(), Hi.getResult()};
      Rewriter.replaceOpWithNewOp<gmir::MergeOp>(Op, DstGTy, DstParts);
      return success();
    }

    if (Step.Action == LegalizeActions::WidenScalar) {
      rewriteWidenScalar(Op, Rewriter, Step.NewType);
      return success();
    }

    if (rewriteFewerElements(Op, Rewriter, Step, DstTy, DstGTy))
      return success();

    return failure();
  }
};

} // namespace

const FrozenRewritePatternSet &
gmir::LegalizerPatternCache::get(MLIRContext &Context, const LegalizerInfo *LI,
                                 const llvm::DataLayout &DL) {
  auto Key = std::make_pair(LI, &DL);
  auto It = Cache.find(Key);
  if (It != Cache.end())
    return It->second;

  // One merged dispatcher pattern per op (see AddSubLegalizePattern/
  // BitwiseLegalizePattern/MulLegalizePattern's doc comments): each
  // queries GMIRLegalizerInfoAdapter::getAction exactly once per visit
  // and dispatches internally on the result, instead of registering one
  // independently-matching pattern per candidate LegalizeAction (which
  // used to mean 2-3 redundant rule-table walks per op visit).
  RewritePatternSet Patterns(&Context);
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

bool gmir::legalize(func::FuncOp FuncOp, MachineFunction &MF,
                    LegalizerPatternCache &PatternCache) {
  MLIRContext *Context = FuncOp.getContext();
  const LegalizerInfo *LI = MF.getSubtarget().getLegalizerInfo();
  const llvm::DataLayout &DL = MF.getDataLayout();
  const FrozenRewritePatternSet &Frozen = PatternCache.get(*Context, LI, DL);

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
  GreedyRewriteConfig Config;
  Config.setRegionSimplificationLevel(GreedySimplifyRegionLevel::Disabled);

  return succeeded(applyPatternsGreedily(FuncOp, Frozen, Config));
}
