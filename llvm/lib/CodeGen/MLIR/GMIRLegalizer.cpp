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

/// Thin wrapper around a target's LegalizerInfo, letting patterns below
/// ask "what does the target's *existing* legality rules say about this
/// gmir op" without gmir having to hand-port its own copy of those rules
/// (this milestone's whole point -- see GMIRLegalizer.h). Held by value
/// inside NarrowScalarAddSubPattern (see below) so pattern instances can
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

/// Implements the single top-level `G_ADD`/`G_SUB` NarrowScalar action
/// (LegalizerHelper::narrowScalarAddSub's exact hi/lo+carry split
/// algorithm, ported to build gmir ops instead of real MIR): unmerge each
/// operand into NumParts narrower pieces, chain
/// gmir.uaddo/gmir.uadde (or usubo/usube) across the pieces threading the
/// carry, then merge the per-piece results back into the original width.
/// Bails on any split that isn't an exact multiple -- confirmed
/// unreachable for this slice's i686 i64-add/sub scenario (64/32 == 2
/// exactly); general leftover handling is explicitly out of scope (see
/// GMIRLegalizer.h). No artifact-cleanup patterns are needed: the freshly
/// emitted ops' own further legalization (e.g. the i1 carry type) and any
/// merge/unmerge cancellation happen automatically via the same permanent
/// downstream target Legalizer pass, exactly as they would for real
/// GlobalISel's own narrowScalarAddSub output.
template <typename OpTy, typename CarryOOp, typename CarryEOp>
class NarrowScalarAddSubPattern : public OpRewritePattern<OpTy> {
public:
  NarrowScalarAddSubPattern(MLIRContext *Context,
                            GMIRLegalizerInfoAdapter Adapter,
                            const llvm::DataLayout &DL)
      : OpRewritePattern<OpTy>(Context), Adapter(Adapter), DL(DL) {}

  LogicalResult matchAndRewrite(OpTy Op,
                                PatternRewriter &Rewriter) const override {
    auto DstGTy = cast<gmir::LLTType>(Op.getResult().getType());
    LLT DstTy = gmir::convertLLT(DstGTy, DL);
    LegalizeActionStep Step =
        Adapter.getAction(getGenericOpcode<OpTy>(), {DstTy});
    if (Step.Action != LegalizeActions::NarrowScalar)
      return failure();

    LLT NarrowTy = Step.NewType;
    unsigned DstBits = DstTy.getSizeInBits();
    unsigned NarrowBits = NarrowTy.getSizeInBits();
    if (NarrowBits == 0 || DstBits % NarrowBits != 0)
      return failure();
    unsigned NumParts = DstBits / NarrowBits;
    if (NumParts < 2)
      return failure();

    MLIRContext *Context = Rewriter.getContext();
    gmir::LLTType NarrowGTy = gmir::convertToGMIRType(*Context, NarrowTy);
    gmir::LLTType CarryGTy = gmir::convertToGMIRType(*Context, LLT::integer(1));

    Location Loc = Op.getLoc();
    SmallVector<mlir::Type, 4> NarrowResultTypes(NumParts, NarrowGTy);
    auto LhsParts =
        gmir::UnmergeOp::create(Rewriter, Loc, NarrowResultTypes, Op.getLhs());
    auto RhsParts =
        gmir::UnmergeOp::create(Rewriter, Loc, NarrowResultTypes, Op.getRhs());

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
        auto C =
            CarryEOp::create(Rewriter, Loc, NarrowGTy, CarryGTy, L, R, CarryIn);
        DstParts.push_back(C.getDst());
        CarryIn = C.getCarryOut();
      }
    }
    Rewriter.replaceOpWithNewOp<gmir::MergeOp>(Op, DstGTy, DstParts);
    return success();
  }

private:
  GMIRLegalizerInfoAdapter Adapter;
  const llvm::DataLayout &DL;
};

} // namespace

const FrozenRewritePatternSet &
gmir::LegalizerPatternCache::get(MLIRContext &Context, const LegalizerInfo *LI,
                                 const llvm::DataLayout &DL) {
  auto It = Cache.find(LI);
  if (It != Cache.end())
    return It->second;

  RewritePatternSet Patterns(&Context);
  Patterns.add<
      NarrowScalarAddSubPattern<gmir::AddOp, gmir::UAddOOp, gmir::UAddEOp>>(
      &Context, GMIRLegalizerInfoAdapter(LI), DL);
  Patterns.add<
      NarrowScalarAddSubPattern<gmir::SubOp, gmir::USubOOp, gmir::USubEOp>>(
      &Context, GMIRLegalizerInfoAdapter(LI), DL);

  return Cache.try_emplace(LI, std::move(Patterns)).first->second;
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
