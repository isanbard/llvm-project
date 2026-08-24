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
// a brand-new gmir.sub op fold() can't emit), and (d), starting with M5
// slice 3, hosting patterns gated on a genuine TargetLowering query (via
// GMIRTargetLoweringAdapter in GMIRCombiner.cpp, the same shape
// GMIRLegalizerInfoAdapter wraps LegalizerInfo in GMIRLegalizer.cpp) --
// e.g. slice 3's DisjointAddToOrPattern: `add(and(a,C1), and(b,C2)) ->
// or(...)` when the masks are disjoint and TLI reports OR legal. M5
// slice 4 adds more patterns in category (c) -- constant reassociation
// and repeated-operand identities (`(x+c1)+c2 -> x+(c1+c2)`, `(a&b)&a
// -> a&b`, `(a^b)^a -> b`) -- ported from a single DAGCombiner.cpp
// helper (reassociateOpsCommutative) shared by visitADD/MUL/AND/OR/XOR.
// M5 slice 5 gives gmir.sub its first real combiner coverage beyond
// x-x->0: three new SubOp::fold() arms (`A-(A-B) -> B`, `(A+B)-A -> B`,
// `(A+B)-B -> A`, all SSA-equality-based like x-x->0), plus two new
// patterns, `sub(-1,x) -> xor(x,-1)` (SubMinusOneToXorPattern) and
// `xor(and(x,y),y) -> and(xor(x,-1),y)` (XorAndDeMorganPattern, the
// first hasOneUse()-gated pattern in this file -- a structural
// profitability check, not a TargetLowering query). M5 slice 6 gives
// gmir.icmp its first combiner coverage: a reflexive `icmp cc X, X ->
// true/false` fold() (IR/GMIRDialect.cpp), the unconditional
// `(X op Y) cmp (X op Z) -> Y cmp Z` / `(X op Y) cmp X/Y -> Y/X cmp 0`
// cancellation family for op in {add, sub, xor} (ICmpSameBinOpPattern/
// ICmpBinOpEqOtherPattern), and the hasOneUse()-gated constant-adjustment
// folds `(X op C1) cmp C2 -> X cmp combine(C1,C2)` for op in
// {add, xor, sub} (ICmpConstAdjustPattern/ICmpSubConstPattern).
// See ~/llvm/mlir_instruction_selection_plan.md's M5 section for the full
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
#include "llvm/ADT/DenseMap.h"
#include <tuple>
#include <utility>

namespace llvm {
class DataLayout;
class LLVMContext;
class MachineFunction;
class TargetLowering;

namespace gmir {

/// Caches the FrozenRewritePatternSet gmir::combine() applies, keyed by
/// (TargetLowering, DataLayout, LLVMContext) -- same shape and same
/// rationale as GMIRLegalizer.h's LegalizerPatternCache (see its doc
/// comment for the full argument for why more than just TargetLowering
/// needs to be part of the key), extended with LLVMContext specifically
/// because DisjointAddToOrPattern (GMIRCombiner.cpp) captures an
/// `LLVMContext &` at cache-population time and reuses it on every later
/// matchAndRewrite call -- unlike LegalizerInfo/DataLayout, whose
/// lifetimes track the TargetSubtargetInfo, an LLVMContext's lifetime
/// tracks the Module being compiled. Omitting it from the key would let
/// a stale cache hit (same TLI/DL pointers, reused for a different
/// Module/LLVMContext in a long-lived embedding that recompiles multiple
/// Modules) hand back a pattern set holding a dangling context
/// reference. Meant to be owned as long-lived state by the caller (e.g.
/// a MachineFunctionPass member), safe to cache across functions for the
/// same reason LegalizerPatternCache is: the patterns' captured
/// MLIRContext*/DataLayout&/TargetLowering*/LLVMContext& are all owned
/// by the Module/TargetSubtargetInfo, which outlive any single
/// MachineFunctionPass invocation -- just not across a Module boundary,
/// which the key now accounts for.
class CombinerPatternCache {
  llvm::DenseMap<std::tuple<const TargetLowering *, const llvm::DataLayout *,
                            const llvm::LLVMContext *>,
                 mlir::FrozenRewritePatternSet>
      Cache;

public:
  // Ctx (the LLVM IR LLVMContext, distinct from the mlir::MLIRContext
  // Context parameter) is only used to build the patterns on a cache
  // miss -- like Context itself, it's a per-call parameter, not part of
  // the cache key.
  const mlir::FrozenRewritePatternSet &get(mlir::MLIRContext &Context,
                                           const TargetLowering *TLI,
                                           const llvm::DataLayout &DL,
                                           llvm::LLVMContext &Ctx);
};

/// Rewrites FuncOp in place: applies PatternCache's patterns (plus, for
/// every op, its own fold() -- the greedy driver invokes fold()
/// unconditionally regardless of what's in the pattern set, see
/// IR/GMIRDialect.cpp) via applyPatternsGreedily, then runs CSE once.
/// Returns false only if applyPatternsGreedily itself fails (e.g.
/// non-convergence) -- mirrors gmir::legalize()'s fallible-step
/// convention used throughout this pipeline's caller,
/// MLIRInstructionSelect.cpp. MF supplies the real TargetLowering/
/// DataLayout/LLVMContext some patterns need to query (starting with M5
/// slice 3), the same way gmir::legalize() uses it for LegalizerInfo.
bool combine(mlir::func::FuncOp FuncOp, MachineFunction &MF,
             CombinerPatternCache &PatternCache);

} // namespace gmir
} // namespace llvm

#endif // LLVM_CODEGEN_MLIR_GMIRCOMBINER_H
