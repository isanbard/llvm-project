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

/// Thin wrapper around MF's target LegalizerInfo, letting patterns below
/// ask "what does the target's *existing* legality rules say about this
/// gmir op" without gmir having to hand-port its own copy of those rules
/// (this milestone's whole point -- see GMIRLegalizer.h).
/// MF.getSubtarget().getLegalizerInfo() is dereferenced unchecked: every
/// in-tree GlobalISel target defines one (a null return only happens for
/// targets that never enable GlobalISel at all), matching this codebase's
/// existing `*MF.getSubtarget().getCallLowering()` idiom
/// (MLIRToGMIRTranslator.cpp).
class GMIRLegalizerInfoAdapter {
public:
  explicit GMIRLegalizerInfoAdapter(MachineFunction &MF)
      : LI(*MF.getSubtarget().getLegalizerInfo()) {}

  LegalizeActionStep getAction(unsigned Opcode, ArrayRef<LLT> Types) const {
    return LI.getAction(LegalityQuery(Opcode, Types));
  }

private:
  const LegalizerInfo &LI;
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
                            const GMIRLegalizerInfoAdapter &Adapter,
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
  const GMIRLegalizerInfoAdapter &Adapter;
  const llvm::DataLayout &DL;
};

} // namespace

bool gmir::legalize(func::FuncOp FuncOp, MachineFunction &MF) {
  MLIRContext *Context = FuncOp.getContext();
  GMIRLegalizerInfoAdapter Adapter(MF);
  const llvm::DataLayout &DL = MF.getDataLayout();

  RewritePatternSet Patterns(Context);
  Patterns.add<
      NarrowScalarAddSubPattern<gmir::AddOp, gmir::UAddOOp, gmir::UAddEOp>>(
      Context, Adapter, DL);
  Patterns.add<
      NarrowScalarAddSubPattern<gmir::SubOp, gmir::USubOOp, gmir::USubEOp>>(
      Context, Adapter, DL);

  FrozenRewritePatternSet Frozen(std::move(Patterns));
  return succeeded(applyPatternsGreedily(FuncOp, Frozen));
}
