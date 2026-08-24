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
#include "llvm/IR/InstrTypes.h"
#include <optional>
#include <type_traits>

using namespace llvm;
using namespace mlir;

// Packages Value (already computed at the op's real width) as the
// 64-bit-sign-extended I64Attr gmir.constant expects -- mirrors
// GMIRDialect.cpp's TU-local helper of the same name exactly (see its
// doc comment there for the full width-correctness rationale). Not
// promoted to a shared header: same "not worth it for one file's use"
// call M5 slice 2 already made for this exact helper, now needed a
// second time within this file (ReassociateConstOpPattern below)
// rather than duplicated inline per call site the way slice 2/3 wrote
// the sign-extend dance each time. Declared static, not wrapped in an
// anonymous namespace, per Bill's Clang/LLVM style preference: `static`
// is the right tool for a free function; anonymous namespaces are for
// classes, which can't be `static` directly (see every class below).
static IntegerAttr makeGMIRConstAttr(MLIRContext *Context, APInt Value) {
  return IntegerAttr::get(mlir::IntegerType::get(Context, 64), Value.sext(64));
}

// Given Op (of type OpTy), returns (X, C) if exactly one operand is a
// constant C and the other is a non-constant X, both already truncated
// to Width -- checking both operand orders, since gmir has no
// automatic canonicalization (unlike DAGCombiner's own N1IsConst
// canonicalization). Shared by every M5 slice-2+ pattern needing this
// "peel a constant off a binop, either side" shape
// (MulNegOneToSubPattern, DisjointAddToOrPattern,
// ReassociateConstOpPattern below) -- static, not anonymous-namespace-
// wrapped, per Bill's Clang/LLVM style preference for free functions.
template <typename OpTy>
static std::optional<std::pair<mlir::Value, APInt>>
matchConstOperand(OpTy Op, unsigned Width) {
  IntegerAttr C;
  if (matchPattern(Op.getRhs(), m_Constant(&C)))
    return std::make_pair(Op.getLhs(), C.getValue().trunc(Width));

  if (matchPattern(Op.getLhs(), m_Constant(&C)))
    return std::make_pair(Op.getRhs(), C.getValue().trunc(Width));

  return std::nullopt;
}

// Shared search skeleton for RepeatedOperandIdempotentPattern/
// XorSelfCancelPattern below: for each of Op's two operands, checks
// whether it's defined by OpTy and whether Op's *other* operand
// structurally equals one of that inner op's own two operands. On the
// first match, calls OnMatch(Inner, MatchedLhs) -- true if Other
// equaled Inner.getLhs(), false if it equaled Inner.getRhs() -- and
// returns its result; std::nullopt if no candidate matches. The two
// patterns need different replacement values on a match (AND/OR want
// Inner's whole result, XOR wants one of Inner's own sub-operands), so
// that choice is left to the caller's callback rather than baked in
// here.
template <typename OpTy, typename CallbackTy>
static std::optional<mlir::Value> matchRepeatedOperand(OpTy Op,
                                                       CallbackTy OnMatch) {
  for (mlir::Value Cand : {Op.getLhs(), Op.getRhs()}) {
    auto Inner = Cand.getDefiningOp<OpTy>();
    if (!Inner)
      continue;

    mlir::Value Other = (Cand == Op.getLhs()) ? Op.getRhs() : Op.getLhs();
    if (Other == Inner.getLhs())
      return OnMatch(Inner, /*MatchedLhs=*/true);

    if (Other == Inner.getRhs())
      return OnMatch(Inner, /*MatchedLhs=*/false);
  }

  return std::nullopt;
}

// Shared by every icmp pattern in this slice -- TargetLowering.cpp:5722's
// SETEQ/SETNE gate (re-verified to also gate candidates 2-7 individually
// at their own call sites, TargetLowering.cpp:5724-5773). Checks against
// CmpInst::ICMP_EQ/ICMP_NE's raw integer values directly, matching
// GMIR_ICmpOp's own "raw predicate integer, not an enum attr" convention.
static bool isEqualityPredicate(int64_t Pred) {
  return Pred == CmpInst::ICMP_EQ || Pred == CmpInst::ICMP_NE;
}

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
    auto DstGTy = cast<gmir::LLTType>(Op.getResult().getType());
    unsigned Width = DstGTy.getScalarSizeInBits();

    auto Match = matchConstOperand(Op, Width);
    if (!Match || !Match->second.isAllOnes())
      return failure();
    mlir::Value Other = Match->first;

    Location Loc = Op.getLoc();
    auto ZeroAttr = IntegerAttr::get(
        mlir::IntegerType::get(Rewriter.getContext(), 64), APInt::getZero(64));
    auto Zero = gmir::ConstantOp::create(Rewriter, Loc, DstGTy, ZeroAttr);
    Rewriter.replaceOpWithNewOp<gmir::SubOp>(Op, DstGTy, Zero.getResult(),
                                             Other);
    return success();
  }
};

/// `sub(-1, x) -> xor(x, -1)` (DAGCombiner.cpp:4398-4400, "Canonicalize
/// (sub -1, x) -> ~x" -- unconditional, no TLI/legality/hasOneUse()
/// gating, same shape as MulNegOneToSubPattern above). Needs a genuine
/// OpRewritePattern, not fold(): the result (~x) isn't equal to either
/// existing operand of the sub, so it has to emit a brand-new gmir.xor.
/// gmir.sub is deliberately NOT Commutative (subtraction isn't), so
/// unlike matchConstOperand's dual-order shape, only the LHS position is
/// meaningful here -- sub(x, -1) is a completely different value (x+1),
/// not this identity -- so this checks getLhs() only via matchPattern/
/// m_Constant directly, rather than matchConstOperand (which would
/// wrongly also match the sub(x,-1) shape).
class SubMinusOneToXorPattern : public OpRewritePattern<gmir::SubOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(gmir::SubOp Op,
                                PatternRewriter &Rewriter) const override {
    auto DstGTy = cast<gmir::LLTType>(Op.getResult().getType());
    unsigned Width = DstGTy.getScalarSizeInBits();

    IntegerAttr C;
    if (!matchPattern(Op.getLhs(), m_Constant(&C)) ||
        !C.getValue().trunc(Width).isAllOnes())
      return failure();

    Location Loc = Op.getLoc();
    IntegerAttr NegOneAttr =
        makeGMIRConstAttr(Rewriter.getContext(), APInt::getAllOnes(Width));
    auto NegOne = gmir::ConstantOp::create(Rewriter, Loc, DstGTy, NegOneAttr);
    Rewriter.replaceOpWithNewOp<gmir::XorOp>(Op, DstGTy, Op.getRhs(),
                                             NegOne.getResult());
    return success();
  }
};

/// `xor(and(x,y), y) -> and(xor(x,-1), y)` (DAGCombiner.cpp:10638-10644,
/// N0.hasOneUse()-gated -- the first hasOneUse()-gated pattern in this
/// file, ported directly from DAGCombiner's own gate rather than
/// invented; confirmed no existing precedent for this gate elsewhere in
/// GMIRCombiner.cpp/GMIRLegalizer.cpp). Introducing a second gmir.and
/// only pays off if the matched and's result isn't needed elsewhere, so
/// this is a structural profitability check, not a TLI/legality query.
///
/// DAGCombiner's own call site only checks ONE of 4 structurally
/// equivalent shapes (AND must be N0, and N0's *second* operand must
/// equal N1) -- visitXOR only ever canonicalizes a *constant* operand to
/// the RHS, never reorders two non-constant operands. gmir has no
/// automatic canonicalization either, same reasoning as every other
/// dual-order check in this pipeline, so this pattern deliberately
/// generalizes to all 4 combinations (both gmir.xor operand orders x
/// both gmir.and operand orders) rather than literally porting the
/// single shape DAGCombiner happens to check. Doesn't reuse
/// matchRepeatedOperand<OpTy> (unlike XorSelfCancelPattern above): that
/// helper's Inner and Outer op are the same OpTy by construction, but
/// here Outer is XorOp and Inner is AndOp -- a genuinely different,
/// two-op-type shape, not worth generalizing the helper for in this
/// slice.
class XorAndDeMorganPattern : public OpRewritePattern<gmir::XorOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(gmir::XorOp Op,
                                PatternRewriter &Rewriter) const override {
    for (mlir::Value Cand : {Op.getLhs(), Op.getRhs()}) {
      auto And = Cand.getDefiningOp<gmir::AndOp>();
      if (!And || !And.getResult().hasOneUse())
        continue;

      mlir::Value Other = (Cand == Op.getLhs()) ? Op.getRhs() : Op.getLhs();
      mlir::Value X;
      if (Other == And.getRhs())
        X = And.getLhs();
      else if (Other == And.getLhs())
        X = And.getRhs();
      else
        continue;

      auto DstGTy = cast<gmir::LLTType>(Op.getResult().getType());
      unsigned Width = DstGTy.getScalarSizeInBits();
      Location Loc = Op.getLoc();
      IntegerAttr NegOneAttr =
          makeGMIRConstAttr(Rewriter.getContext(), APInt::getAllOnes(Width));
      auto NegOne = gmir::ConstantOp::create(Rewriter, Loc, DstGTy, NegOneAttr);
      auto NotX =
          gmir::XorOp::create(Rewriter, Loc, DstGTy, X, NegOne.getResult());
      Rewriter.replaceOpWithNewOp<gmir::AndOp>(Op, DstGTy, NotX.getResult(),
                                               Other);
      return success();
    }

    return failure();
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
      auto Match = matchConstOperand(And, Width);
      return Match ? std::optional(Match->second) : std::nullopt;
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

/// Function-pointer type for ReassociateConstOpPattern's constant
/// combinator (add/mul/and/or/xor's own APInt operator, injected via the
/// constructor rather than a per-op template function specialization --
/// see GMIRCombiner.h and design doc §1.24 for why: the latter shape is
/// exactly what caused friction on GMIRLegalizer.cpp's
/// getGenericOpcode<OpTy>() when Bill tried converting it to `static`
/// (explicit specializations can't independently take a storage-class-
/// specifier), so this pattern deliberately avoids reintroducing it).
using ConstCombinator = APInt (*)(const APInt &, const APInt &);

/// `(op (op x, c1), c2) -> (op x, (op c1, c2))` (DAGCombiner.cpp's
/// reassociateOpsCommutative, lines 1252-1259 -- the purely structural,
/// unconditional half of a helper shared identically by visitADD/MUL/
/// AND/OR/XOR; the third sub-fold there is TLI-gated via
/// isReassocProfitable(SelectionDAG&, ...)/isReassocProfitable(
/// MachineRegisterInfo&, ...), neither of which exists yet at
/// gmir-combine time -- see design doc §1.24, not ported). fold() always
/// runs before patterns (M5 slice 2's established precedent), so Op
/// never has both operands constant here -- no coordination conflict
/// with the arithmetic-op fold()s slice 1 already registered.
template <typename OpTy>
class ReassociateConstOpPattern : public OpRewritePattern<OpTy> {
  ConstCombinator Combine;

public:
  ReassociateConstOpPattern(MLIRContext *Context, ConstCombinator Combine)
      : OpRewritePattern<OpTy>(Context), Combine(Combine) {}

  LogicalResult matchAndRewrite(OpTy Op,
                                PatternRewriter &Rewriter) const override {
    auto DstGTy = cast<gmir::LLTType>(Op.getResult().getType());
    unsigned Width = DstGTy.getScalarSizeInBits();

    auto Outer = matchConstOperand(Op, Width);
    if (!Outer)
      return failure();
    auto [OuterOther, C2] = *Outer;

    // OuterOther must itself be OpTy(x, c1) for some non-constant x and
    // constant c1 -- gmir has no automatic canonicalization, same
    // reasoning as every prior slice's dual-order checks.
    auto InnerOp = OuterOther.template getDefiningOp<OpTy>();
    if (!InnerOp)
      return failure();

    auto Inner = matchConstOperand(InnerOp, Width);
    if (!Inner)
      return failure();
    auto [X, C1] = *Inner;

    IntegerAttr NewConst =
        makeGMIRConstAttr(Rewriter.getContext(), Combine(C1, C2));
    auto NewC =
        gmir::ConstantOp::create(Rewriter, Op.getLoc(), DstGTy, NewConst);
    Rewriter.replaceOpWithNewOp<OpTy>(Op, DstGTy, X, NewC.getResult());
    return success();
  }
};

/// AND/OR idempotence (DAGCombiner.cpp's reassociateOpsCommutative,
/// lines 1270-1277): `(a op b) op a -> a op b`, `(a op b) op b -> a op
/// b`. Both ops share identical behavior (replace with the inner op's
/// whole result) -- unlike XOR below, which returns a *sub*-operand of
/// the inner op instead, so it can't share this template.
template <typename OpTy>
class RepeatedOperandIdempotentPattern : public OpRewritePattern<OpTy> {
public:
  using OpRewritePattern<OpTy>::OpRewritePattern;

  LogicalResult matchAndRewrite(OpTy Op,
                                PatternRewriter &Rewriter) const override {
    auto Result = matchRepeatedOperand(
        Op, [](OpTy Inner, bool) { return Inner.getResult(); });
    if (!Result)
      return failure();

    Rewriter.replaceOp(Op, *Result);
    return success();
  }
};

/// XOR self-cancellation (DAGCombiner.cpp's reassociateOpsCommutative,
/// lines 1278-1285): `(a ^ b) ^ a -> b`, `(a ^ b) ^ b -> a`. Kept
/// standalone rather than templated: unlike AND/OR's idempotence above,
/// the result here is one of the *inner* op's own sub-operands, not the
/// inner op's whole result, so it doesn't fit
/// RepeatedOperandIdempotentPattern's shape.
class XorSelfCancelPattern : public OpRewritePattern<gmir::XorOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(gmir::XorOp Op,
                                PatternRewriter &Rewriter) const override {
    auto Result =
        matchRepeatedOperand(Op, [](gmir::XorOp Inner, bool MatchedLhs) {
          return MatchedLhs ? Inner.getRhs() : Inner.getLhs();
        });
    if (!Result)
      return failure();

    Rewriter.replaceOp(Op, *Result);
    return success();
  }
};

/// `(X op Y) == (X op Z) -> Y == Z` for op in {add, sub, xor}
/// (TargetLowering.cpp:5722-5741). The two "aligned" pairings (operand(0)
/// matches operand(0), or operand(1) matches operand(1)) are checked for
/// all three op types unconditionally; the two "swapped" pairings are
/// checked only when IsCommutative (true for add/xor, false for sub --
/// gmir.sub isn't Commutative, matching TargetLowering.cpp's own
/// isCommutativeBinOp(N0.getOpcode()) gate on the swapped checks).
template <typename OpTy>
class ICmpSameBinOpPattern : public OpRewritePattern<gmir::ICmpOp> {
  bool IsCommutative;

public:
  ICmpSameBinOpPattern(MLIRContext *Context, bool IsCommutative)
      : OpRewritePattern(Context), IsCommutative(IsCommutative) {}

  LogicalResult matchAndRewrite(gmir::ICmpOp Op,
                                PatternRewriter &Rewriter) const override {
    if (!isEqualityPredicate(Op.getPredicateAttr().getValue().getSExtValue()))
      return failure();

    auto LhsOp = Op.getLhs().getDefiningOp<OpTy>();
    auto RhsOp = Op.getRhs().getDefiningOp<OpTy>();
    if (!LhsOp || !RhsOp)
      return failure();

    mlir::Value NewLhs, NewRhs;
    if (LhsOp.getLhs() == RhsOp.getLhs()) {
      NewLhs = LhsOp.getRhs();
      NewRhs = RhsOp.getRhs();
    } else if (LhsOp.getRhs() == RhsOp.getRhs()) {
      NewLhs = LhsOp.getLhs();
      NewRhs = RhsOp.getLhs();
    } else if (IsCommutative && LhsOp.getLhs() == RhsOp.getRhs()) {
      NewLhs = LhsOp.getRhs();
      NewRhs = RhsOp.getLhs();
    } else if (IsCommutative && LhsOp.getRhs() == RhsOp.getLhs()) {
      NewLhs = LhsOp.getLhs();
      NewRhs = RhsOp.getRhs();
    } else {
      return failure();
    }

    // gmir.icmp's result is never SameOperandsAndResultType with its
    // operands (res is always 1-bit) -- the replacement must reuse Op's
    // own result type, not derive one from NewLhs/NewRhs.
    Rewriter.replaceOpWithNewOp<gmir::ICmpOp>(
        Op, Op.getResult().getType(), Op.getPredicateAttr(), NewLhs, NewRhs);
    return success();
  }
};

/// `(X op Y) cmp Z -> Y cmp 0` when Z structurally equals X, and (for
/// add/xor only) `X cmp 0` when Z equals Y (foldSetCCWithBinOp,
/// TargetLowering.cpp:4596-4632 -- the X==N1 check applies to add/sub/xor
/// alike, 4611-4612; the Y==N1 -> X==0 check is add/xor-only, 4617-4620 --
/// sub's Y==N1 sibling needs a shift op gmir doesn't have, deliberately
/// excluded here, same exclusion as DAGCombiner.cpp's own SUB-specific
/// X==Y<<1 fallthrough at 4626-4631).
///
/// Checks both gmir.icmp operand positions for being the binop (the "Z" in
/// the identity above): TargetLowering.cpp reaches this fold from two call
/// sites, one of which is additionally gated by a register-pressure/
/// induction-variable profitability heuristic (a pure profitability
/// judgment, not a correctness requirement) that gmir has no equivalent
/// induction-variable-chain analysis to replicate. This pattern
/// deliberately ports only the sound core identity, applied symmetrically
/// to both icmp operand positions, matching this slice's Tier-1/
/// unconditional scope.
template <typename OpTy>
class ICmpBinOpEqOtherPattern : public OpRewritePattern<gmir::ICmpOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(gmir::ICmpOp Op,
                                PatternRewriter &Rewriter) const override {
    if (!isEqualityPredicate(Op.getPredicateAttr().getValue().getSExtValue()))
      return failure();

    for (mlir::Value Cand : {Op.getLhs(), Op.getRhs()}) {
      auto BinOp = Cand.getDefiningOp<OpTy>();
      if (!BinOp)
        continue;

      mlir::Value Other = (Cand == Op.getLhs()) ? Op.getRhs() : Op.getLhs();
      mlir::Value Remaining;
      if (BinOp.getLhs() == Other)
        Remaining = BinOp.getRhs();
      else if constexpr (!std::is_same_v<OpTy, gmir::SubOp>)
        if (BinOp.getRhs() == Other)
          Remaining = BinOp.getLhs();

      if (!Remaining)
        continue;

      auto DstGTy = cast<gmir::LLTType>(BinOp.getResult().getType());
      IntegerAttr ZeroAttr = makeGMIRConstAttr(
          Rewriter.getContext(), APInt::getZero(DstGTy.getScalarSizeInBits()));
      auto Zero =
          gmir::ConstantOp::create(Rewriter, Op.getLoc(), DstGTy, ZeroAttr);
      Rewriter.replaceOpWithNewOp<gmir::ICmpOp>(Op, Op.getResult().getType(),
                                                Op.getPredicateAttr(),
                                                Remaining, Zero.getResult());
      return success();
    }

    return failure();
  }
};

/// `(X op C1) == C2 -> X == combine(C1, C2)` for op in {add, xor}
/// (TargetLowering.cpp:5750-5755/5758-5763), hasOneUse()-gated on the
/// inner op, no TLI query -- same gate class as XorAndDeMorganPattern.
/// Outer op (ICmpOp) and inner op (OpTy) are different types, same shape
/// as XorAndDeMorganPattern's outer-XorOp/inner-AndOp combo, not
/// ReassociateConstOpPattern's same-op-type shape.
template <typename OpTy>
class ICmpConstAdjustPattern : public OpRewritePattern<gmir::ICmpOp> {
  ConstCombinator Combine;

public:
  ICmpConstAdjustPattern(MLIRContext *Context, ConstCombinator Combine)
      : OpRewritePattern(Context), Combine(Combine) {}

  LogicalResult matchAndRewrite(gmir::ICmpOp Op,
                                PatternRewriter &Rewriter) const override {
    if (!isEqualityPredicate(Op.getPredicateAttr().getValue().getSExtValue()))
      return failure();

    // Width comes from the compared *operands*, never Op's own 1-bit
    // result -- the width-truncation nuance specific to icmp among every
    // pattern in this file so far (every prior arithmetic pattern's
    // Op.getResult().getType() coincides with its operand width; icmp's
    // does not).
    auto OperandGTy = cast<gmir::LLTType>(Op.getLhs().getType());
    unsigned Width = OperandGTy.getScalarSizeInBits();

    auto Outer = matchConstOperand(Op, Width);
    if (!Outer)
      return failure();
    auto [OuterOther, C2] = *Outer;

    auto InnerOp = OuterOther.template getDefiningOp<OpTy>();
    if (!InnerOp || !InnerOp.getResult().hasOneUse())
      return failure();

    auto Inner = matchConstOperand(InnerOp, Width);
    if (!Inner)
      return failure();
    auto [X, C1] = *Inner;

    IntegerAttr NewConst =
        makeGMIRConstAttr(Rewriter.getContext(), Combine(C1, C2));
    auto NewC =
        gmir::ConstantOp::create(Rewriter, Op.getLoc(), OperandGTy, NewConst);
    Rewriter.replaceOpWithNewOp<gmir::ICmpOp>(Op, Op.getResult().getType(),
                                              Op.getPredicateAttr(), X,
                                              NewC.getResult());
    return success();
  }
};

/// `(C1 - X) == C2 -> X == (C1 - C2)` (TargetLowering.cpp:5767-5773),
/// hasOneUse()-gated. gmir.sub isn't Commutative, so C1 must specifically
/// be the sub's LHS -- same non-commutative-specific treatment as
/// SubMinusOneToXorPattern above (matchConstOperand's dual-order peel
/// would wrongly also match sub(X, C1), a completely different value).
class ICmpSubConstPattern : public OpRewritePattern<gmir::ICmpOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(gmir::ICmpOp Op,
                                PatternRewriter &Rewriter) const override {
    if (!isEqualityPredicate(Op.getPredicateAttr().getValue().getSExtValue()))
      return failure();

    auto OperandGTy = cast<gmir::LLTType>(Op.getLhs().getType());
    unsigned Width = OperandGTy.getScalarSizeInBits();

    auto Outer = matchConstOperand(Op, Width);
    if (!Outer)
      return failure();
    auto [OuterOther, C2] = *Outer;

    auto InnerSub = OuterOther.getDefiningOp<gmir::SubOp>();
    if (!InnerSub || !InnerSub.getResult().hasOneUse())
      return failure();

    IntegerAttr C;
    if (!matchPattern(InnerSub.getLhs(), m_Constant(&C)))
      return failure();
    APInt C1 = C.getValue().trunc(Width);
    mlir::Value X = InnerSub.getRhs();

    IntegerAttr NewConst = makeGMIRConstAttr(Rewriter.getContext(), C1 - C2);
    auto NewC =
        gmir::ConstantOp::create(Rewriter, Op.getLoc(), OperandGTy, NewConst);
    Rewriter.replaceOpWithNewOp<gmir::ICmpOp>(Op, Op.getResult().getType(),
                                              Op.getPredicateAttr(), X,
                                              NewC.getResult());
    return success();
  }
};
} // namespace

const FrozenRewritePatternSet &
gmir::CombinerPatternCache::get(MLIRContext &Context, const TargetLowering *TLI,
                                const llvm::DataLayout &DL, LLVMContext &Ctx) {
  auto Key = std::make_tuple(TLI, &DL, &Ctx);
  auto It = Cache.find(Key);
  if (It != Cache.end())
    return It->second;

  RewritePatternSet Patterns(&Context);
  Patterns.add<MulNegOneToSubPattern>(&Context);
  Patterns.add<SubMinusOneToXorPattern>(&Context);
  Patterns.add<XorAndDeMorganPattern>(&Context);
  Patterns.add<DisjointAddToOrPattern>(&Context,
                                       GMIRTargetLoweringAdapter(TLI, DL), Ctx);
  Patterns.add<ReassociateConstOpPattern<gmir::AddOp>>(
      &Context, +[](const APInt &A, const APInt &B) { return A + B; });
  Patterns.add<ReassociateConstOpPattern<gmir::MulOp>>(
      &Context, +[](const APInt &A, const APInt &B) { return A * B; });
  Patterns.add<ReassociateConstOpPattern<gmir::AndOp>>(
      &Context, +[](const APInt &A, const APInt &B) { return A & B; });
  Patterns.add<ReassociateConstOpPattern<gmir::OrOp>>(
      &Context, +[](const APInt &A, const APInt &B) { return A | B; });
  Patterns.add<ReassociateConstOpPattern<gmir::XorOp>>(
      &Context, +[](const APInt &A, const APInt &B) { return A ^ B; });
  Patterns.add<RepeatedOperandIdempotentPattern<gmir::AndOp>>(&Context);
  Patterns.add<RepeatedOperandIdempotentPattern<gmir::OrOp>>(&Context);
  Patterns.add<XorSelfCancelPattern>(&Context);

  // M5 slice 6: gmir.icmp's first combiner coverage.
  Patterns.add<ICmpSameBinOpPattern<gmir::AddOp>>(&Context,
                                                  /*IsCommutative=*/true);
  Patterns.add<ICmpSameBinOpPattern<gmir::SubOp>>(&Context,
                                                  /*IsCommutative=*/false);
  Patterns.add<ICmpSameBinOpPattern<gmir::XorOp>>(&Context,
                                                  /*IsCommutative=*/true);
  Patterns.add<ICmpBinOpEqOtherPattern<gmir::AddOp>>(&Context);
  Patterns.add<ICmpBinOpEqOtherPattern<gmir::SubOp>>(&Context);
  Patterns.add<ICmpBinOpEqOtherPattern<gmir::XorOp>>(&Context);
  Patterns.add<ICmpConstAdjustPattern<gmir::AddOp>>(
      &Context, +[](const APInt &C1, const APInt &C2) { return C2 - C1; });
  Patterns.add<ICmpConstAdjustPattern<gmir::XorOp>>(
      &Context, +[](const APInt &C1, const APInt &C2) { return C1 ^ C2; });
  Patterns.add<ICmpSubConstPattern>(&Context);

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
