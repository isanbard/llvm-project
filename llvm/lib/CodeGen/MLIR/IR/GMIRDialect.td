//===- GMIRDialect.td - GMIR dialect ---------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the `gmir` dialect: an MLIR mirror of GlobalISel's
// generic MIR, used as the target of the experimental MLIR-based
// instruction-selection pipeline. Ops grow incrementally, milestone by
// milestone; see ~/llvm/mlir_instruction_selection_plan.md for the roadmap.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_MLIR_IR_GMIRDIALECT_TD
#define LLVM_CODEGEN_MLIR_IR_GMIRDIALECT_TD

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def GMIR_Dialect : Dialect {
  let name = "gmir";
  let summary = "Mirrors GlobalISel's generic MIR (gMIR)";
  let cppNamespace = "::llvm::gmir";
  // Wires up the generatedTypePrinter/generatedTypeParser functions
  // -gen-typedef-defs already emits (previously unused: nothing textually
  // printed or parsed `gmir` IR before M4 slice 1's -print-gmir-after-
  // legalize debug flag). Without this, MLIRContext's generic asm printer
  // hits "dialect has no registered type printing hook" and aborts the
  // moment it needs to print a bare !gmir.llt (e.g. a FuncOp's argument
  // list) -- gmir ops themselves only ever used custom assemblyFormat or
  // were never printed at all until now.
  let useDefaultTypePrinterParser = 1;
  // Lets the greedy pattern rewrite driver turn an Attribute-typed
  // OpFoldResult (e.g. SubOp::fold's `x-x -> 0`, which needs a *new*
  // zero value rather than an existing operand) back into a real
  // gmir.constant op -- see GMIRDialect.cpp's materializeConstant.
  let hasConstantMaterializer = 1;
}

class GMIR_Type<string name, string typeMnemonic, list<Trait> traits = [],
                string baseCppClass = "::mlir::Type">
    : TypeDef<GMIR_Dialect, name, traits, baseCppClass> {
  let mnemonic = typeMnemonic;
}

def GMIR_LLTType : GMIR_Type<"LLT", "llt"> {
  let summary = "Mirrors GlobalISel's Low Level Type (LLT)";
  let description = [{
    A structural mirror of `llvm::LLT`: either a scalar of a given bit
    width, a vector of `numElements` such scalars, or a pointer in
    `addressSpace` (when `scalarSizeInBits` is 0). Deliberately uses the
    default derived assembly format for this scaffolding slice rather than
    LLT's compact `s32`/`v4s32`/`p0`/`nxv4s32` textual syntax -- the compact
    custom parser/printer is deferred until real `gmir` ops exist to
    exercise it.
  }];
  let parameters = (ins
    "unsigned":$scalarSizeInBits,
    "unsigned":$numElements,
    "unsigned":$addressSpace,
    "bool":$isScalable
  );
  let assemblyFormat = "`<` $scalarSizeInBits `,` $numElements `,` "
                        "$addressSpace `,` $isScalable `>`";
}

class GMIR_Op<string mnemonic, list<Trait> traits = []>
    : Op<GMIR_Dialect, mnemonic, traits>;

class GMIR_BinOp<string mnemonic, list<Trait> traits = []>
    : GMIR_Op<mnemonic, !listconcat(traits, [Pure, SameOperandsAndResultType])> {
  let arguments = (ins GMIR_LLTType:$lhs, GMIR_LLTType:$rhs);
  let results = (outs GMIR_LLTType:$res);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($res)";
}

// `hasFolder = 1` on the six ops below (not GMIR_UMulHOp/GMIR_SDivOp,
// out of scope for M5 slice 1): implements DAGCombiner-style tier-1
// algebraic identities (e.g. `x+0 -> x`) directly as MLIR op folders,
// the idiomatic mechanism for "reduce to one of my own operands, or to
// a constant, given my operands' current values" -- fires automatically
// wherever applyPatternsGreedily already runs (GMIRLegalizer.cpp,
// GMIRCombiner.cpp), no separate pattern registration needed. See
// GMIRDialect.cpp for the fold() bodies and ~/llvm/
// mlir_instruction_selection_plan.md's M5 section for the DAGCombiner
// provenance of each rule.
def GMIR_AddOp  : GMIR_BinOp<"add",  [Commutative]> {
  let hasFolder = 1;
}
def GMIR_UMulHOp : GMIR_BinOp<"umulh", [Commutative]>;
def GMIR_SubOp  : GMIR_BinOp<"sub"> {
  let hasFolder = 1;
}
def GMIR_MulOp  : GMIR_BinOp<"mul",  [Commutative]> {
  let hasFolder = 1;
}
def GMIR_SDivOp : GMIR_BinOp<"sdiv">;
def GMIR_AndOp  : GMIR_BinOp<"and",  [Commutative]> {
  let hasFolder = 1;
}
def GMIR_OrOp   : GMIR_BinOp<"or",   [Commutative]> {
  let hasFolder = 1;
}
def GMIR_XorOp  : GMIR_BinOp<"xor",  [Commutative]> {
  let hasFolder = 1;
}

def GMIR_ConstantOp : GMIR_Op<"constant", [Pure, ConstantLike]> {
  let summary = "A constant of `!gmir.llt` type";
  let description = [{
    Holds a 64-bit value (signless: the bit pattern, not a signedness
    claim -- signedness is a property of the consuming op, e.g. `gmir.sdiv`).
    Constants that don't fit in 64 bits are unsupported for now (M1 is
    scoped to i32/i64 scalar arithmetic).

    `hasFolder = 1`: the `ConstantLike` trait's contract requires a working
    `fold()` (returning the op's own value attribute) -- e.g. the greedy
    pattern rewrite driver's constant-CSE step (GreedyRewriteConfig's
    `cseConstants`, on by default; see GMIRLegalizer.cpp) walks every
    ConstantLike op expecting this, and asserts if it's missing. Unused
    until M4 slice 1 introduced the first pass (GMIRLegalizer) that runs
    applyPatternsGreedily over gmir IR at all.
  }];
  let arguments = (ins I64Attr:$value);
  let results = (outs GMIR_LLTType:$res);
  let assemblyFormat = "$value attr-dict `:` type($res)";
  let hasFolder = 1;
}

def GMIR_ICmpOp : GMIR_Op<"icmp", [Pure]> {
  let summary = "Integer comparison, mirroring GlobalISel's G_ICMP";
  let description = [{
    `predicate` holds the raw `llvm::CmpInst::Predicate` integer value
    (e.g. `ICMP_EQ`, `ICMP_SLT`) -- a plain attribute rather than a new
    MLIR enum-attr, matching `gmir.constant`'s value attribute (M1): kept
    simple until real need for parser/printer-level enum names arises.
  }];
  let arguments = (ins I64Attr:$predicate, GMIR_LLTType:$lhs, GMIR_LLTType:$rhs);
  let results = (outs GMIR_LLTType:$res);
}

def GMIR_BrOp : GMIR_Op<"br", [Pure, Terminator,
    DeclareOpInterfaceMethods<BranchOpInterface, ["getSuccessorForOperands"]>]> {
  let summary = "Unconditional branch, mirroring GlobalISel's G_BR";
  let description = [{
    `destOperands` feed `dest`'s block arguments -- MLIR's idiom for SSA
    values live across block boundaries (i.e. what LLVM IR represents as
    PHI nodes in the destination block; there is no separate `gmir.phi`
    op). Modeled directly on `cf.br` (`mlir::cf::BranchOp`).
  }];
  let arguments = (ins Variadic<GMIR_LLTType>:$destOperands);
  let successors = (successor AnySuccessor:$dest);
}

def GMIR_CondBrOp : GMIR_Op<"brcond", [Pure, Terminator, AttrSizedOperandSegments,
    DeclareOpInterfaceMethods<BranchOpInterface, ["getSuccessorForOperands"]>]> {
  let summary = "Conditional branch, mirroring GlobalISel's G_BRCOND + G_BR";
  let description = [{
    Modeled directly on `cf.cond_br` (`mlir::cf::CondBranchOp`); see
    `gmir.br` for why successor operands (here two groups, one per
    destination) stand in for PHI nodes. `AttrSizedOperandSegments` is
    needed because there are two variadic operand groups plus the scalar
    `condition`.
  }];
  let arguments = (ins GMIR_LLTType:$condition,
                       Variadic<GMIR_LLTType>:$trueDestOperands,
                       Variadic<GMIR_LLTType>:$falseDestOperands);
  let successors = (successor AnySuccessor:$trueDest, AnySuccessor:$falseDest);
}

def GMIR_AllocaOp : GMIR_Op<"alloca"> {
  let summary = "Static stack allocation, mirroring G_FRAME_INDEX";
  let description = [{
    Static-only: constant size, entry-block placement in the original IR
    (`llvm::AllocaInst::isStaticAlloca()`). Dynamic-sized or non-entry-block
    allocas fall back to the legacy selector rather than being modeled here.
    Deliberately named without a "static_" prefix so a future
    `gmir.dyn_alloca` (mirroring `G_DYN_STACKALLOC`) reads as a natural
    sibling op rather than a retrofit -- matches how GlobalISel itself keeps
    these as two separate opcodes rather than one polymorphic one.
  }];
  let arguments = (ins I64Attr:$size, I64Attr:$align);
  let results = (outs GMIR_LLTType:$res); // always a pointer LLT
}

def GMIR_LoadOp : GMIR_Op<"load",
    [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "Load from memory, mirroring GlobalISel's G_LOAD";
  let description = [{
    `ordering`/`syncscope` hold the raw `llvm::AtomicOrdering`/
    `SyncScope::ID` integer values (0 = NotAtomic / System) -- a plain
    attribute rather than a new MLIR enum-attr, matching `gmir.icmp`'s
    predicate. Carried from day one rather than added only once atomics are
    needed: `MachineMemOperand`'s constructor threads ordering/syncscope in
    alongside size/alignment, so adding them to this op later would be a
    breaking shape change, not an additive one. For a struct/array-typed
    load, the importer emits one `gmir.load` per flattened scalar/pointer
    leaf (each with its own already-resolved offset pointer and alignment)
    rather than a single multi-result op -- mirrors IRTranslator's
    multi-register loop, not a new gmir-level aggregate concept.

    `isInvariant`/`isNonTemporal` mirror `TargetLoweringBase::
    getLoadMemOperandFlags`'s `MOInvariant`/`MONonTemporal` derivation from
    the source `!invariant.load`/`!nontemporal` metadata -- a plain
    presence check, no analysis dependency, unlike `MODereferenceable`
    (deliberately not modeled here: the real derivation needs
    `isDereferenceableAndAlignedPointer`, which needs `AssumptionCache`/
    `TargetLibraryInfo` that the importer doesn't have threaded in, and
    the cheaper `!dereferenceable`-metadata-only fallback is narrow enough
    not to be worth the op-shape churn until something needs it).
  }];
  let arguments = (ins GMIR_LLTType:$ptr, I64Attr:$align, I64Attr:$ordering,
                       I64Attr:$syncscope, UnitAttr:$isVolatile,
                       UnitAttr:$isInvariant, UnitAttr:$isNonTemporal);
  let results = (outs GMIR_LLTType:$res);
}

def GMIR_StoreOp : GMIR_Op<"store",
    [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "Store to memory, mirroring GlobalISel's G_STORE";
  let description = [{
    See `gmir.load` for the `ordering`/`syncscope` attribute convention and
    the "one op per flattened leaf" aggregate-decomposition approach.
    `isNonTemporal` mirrors `TargetLoweringBase::getStoreMemOperandFlags`
    exactly (which, unlike the load side, has no invariant or
    dereferenceable concept for stores at all).
  }];
  let arguments = (ins GMIR_LLTType:$value, GMIR_LLTType:$ptr, I64Attr:$align,
                       I64Attr:$ordering, I64Attr:$syncscope,
                       UnitAttr:$isVolatile, UnitAttr:$isNonTemporal);
}

def GMIR_PtrAddOp : GMIR_Op<"ptr_add", [Pure]> {
  let summary = "Pointer + integer offset, mirroring GlobalISel's G_PTR_ADD";
  let description = [{
    `noUWrap`/`noUSWrap`/`inBounds` mirror MachineInstr::MIFlag::NoUWrap/
    NoUSWrap/InBounds (see MachineInstr::copyFlagsFromInstruction's
    GetElementPtrInst case) -- UnitAttrs, matching `gmir.load`/
    `gmir.store`'s `isVolatile` convention for boolean flags rather than a
    packed bitmask. The importer emits one `gmir.ptr_add` per coalesced
    constant-offset flush point and one per variable GEP index, mirroring
    IRTranslator::translateGetElementPtr's coalescing algorithm exactly
    (ported for -global-isel byte-parity rather than a naive
    one-ptr_add-per-index scheme -- see design doc §1.5's GEP bullet).
    Also reused (starting in a later milestone slice) for materializing a
    sub-object byte offset within an already-flattened aggregate
    load/store, unrelated to any user-written GEP.
  }];
  let arguments = (ins GMIR_LLTType:$ptr, GMIR_LLTType:$offset,
                       UnitAttr:$noUWrap, UnitAttr:$noUSWrap,
                       UnitAttr:$inBounds);
  let results = (outs GMIR_LLTType:$res); // always a pointer LLT
}

def GMIR_AnyExtOp : GMIR_Op<"anyext", [Pure]> {
  let summary = "Extends a scalar to a wider type with unspecified new bits, mirroring GlobalISel's G_ANYEXT";
  let description = [{
    Emitted only by GMIRLegalizer's shared rewriteWidenScalar helper (see
    GMIRLegalizer.cpp, called from AddSubLegalizePattern's/
    BitwiseLegalizePattern's/MulLegalizePattern's WidenScalar dispatch
    case), to widen a too-narrow operand up to a legal width before
    re-running the original op there -- correct (rather than `zext`/
    `sext`) because for every op rewriteWidenScalar handles, the low-N
    result bits depend only on the low-N input bits, so the extended
    high bits' actual value never matters. `hasVerifier = 1`: no stock ODS
    trait expresses "result must be strictly wider than the operand"
    (`SameOperandsAndResultType` doesn't fit -- the whole point of this op
    is that they differ), so `AnyExtOp::verify()` in GMIRDialect.cpp checks
    it by hand.
  }];
  let arguments = (ins GMIR_LLTType:$src);
  let results = (outs GMIR_LLTType:$res);
  let hasVerifier = 1;
}

def GMIR_TruncOp : GMIR_Op<"trunc", [Pure]> {
  let summary = "Truncates a scalar to a narrower type, mirroring GlobalISel's G_TRUNC";
  let description = [{
    Inverse of `gmir.anyext`: emitted only by GMIRLegalizer's shared
    rewriteWidenScalar helper, to truncate a widened op's result back
    down to its original width. See `gmir.anyext` for why
    `hasVerifier = 1` is needed here too (checking `res` is strictly
    narrower than `src`).
  }];
  let arguments = (ins GMIR_LLTType:$src);
  let results = (outs GMIR_LLTType:$res);
  let hasVerifier = 1;
}

class GMIR_AddSubCarryOp<string mnemonic, list<Trait> traits = []>
    : GMIR_Op<mnemonic, !listconcat(traits, [Pure])> {
  let summary = "Add/sub with carry, mirroring GlobalISel's G_U{ADD,SUB}{O,E}";
  let description = [{
    `dst`/`carryOut` mirror the two results of `G_UADDO`/`G_UADDE`/
    `G_USUBO`/`G_USUBE` -- `carryOut` is a boolean-width (`!gmir.llt<1, ...>`)
    result, not the same type as `dst`, which is why this doesn't reuse
    `GMIR_BinOp` (its `SameOperandsAndResultType` doesn't fit). Emitted only
    by GMIRLegalizer's AddSubLegalizePattern (see GMIRLegalizer.cpp) --
    GMIRImporter never produces these directly, since LLVM IR has no
    surface-level add-with-carry construct that survives to this pipeline's
    supported subset.
  }];
  let results = (outs GMIR_LLTType:$dst, GMIR_LLTType:$carryOut);
}

def GMIR_UAddOOp : GMIR_AddSubCarryOp<"uaddo", [Commutative]> {
  let arguments = (ins GMIR_LLTType:$lhs, GMIR_LLTType:$rhs);
}
def GMIR_UAddEOp : GMIR_AddSubCarryOp<"uadde"> {
  let arguments = (ins GMIR_LLTType:$lhs, GMIR_LLTType:$rhs, GMIR_LLTType:$carryIn);
}
def GMIR_USubOOp : GMIR_AddSubCarryOp<"usubo"> {
  let arguments = (ins GMIR_LLTType:$lhs, GMIR_LLTType:$rhs);
}
def GMIR_USubEOp : GMIR_AddSubCarryOp<"usube"> {
  let arguments = (ins GMIR_LLTType:$lhs, GMIR_LLTType:$rhs, GMIR_LLTType:$carryIn);
}

def GMIR_UnmergeOp : GMIR_Op<"unmerge", [Pure]> {
  let summary = "Splits a value into narrower pieces, mirroring G_UNMERGE_VALUES";
  let description = [{
    Two distinct uses, both emitted only by GMIRLegalizer, mirroring
    G_UNMERGE_VALUES's own dual real-GlobalISel role: (1) AddSubLegalizePattern's/
    BitwiseLegalizePattern's/MulLegalizePattern's NarrowScalar dispatch
    cases split a too-wide *scalar* operand into `dsts.size()` contiguous
    narrower scalar pieces; (2) the shared rewriteFewerElements helper
    (called from all three merged patterns' FewerElements dispatch case)
    splits a *vector* operand into its `dsts.size()` scalar lanes (one
    dst per element, not a bit-width split) before scalarizing a vector
    op. `UnmergeOp::verify()`
    in GMIRDialect.cpp distinguishes the two by `src`'s shape and checks
    each accordingly -- no stock ODS trait fits either case ("all
    *results* share one type" isn't a built-in trait, and `src`'s type
    legitimately differs from `dsts`'s, ruling out
    `SameOperandsAndResultType`).
  }];
  let arguments = (ins GMIR_LLTType:$src);
  let results = (outs Variadic<GMIR_LLTType>:$dsts);
  let hasVerifier = 1;
}

def GMIR_MergeOp : GMIR_Op<"merge", [Pure]> {
  let summary = "Concatenates narrow pieces into a scalar, mirroring G_MERGE_VALUES";
  let description = [{
    Inverse of `gmir.unmerge`: reassembles a NarrowScalar dispatch case's
    per-piece results (`gmir.uaddo`/`gmir.uadde`/`usubo`/`usube` for
    AddSubLegalizePattern, or a narrowed `gmir.and`/`or`/`xor` for
    BitwiseLegalizePattern) back into the original wider `!gmir.llt`.
    See `gmir.unmerge` for why `hasVerifier = 1` is needed here too
    (checking `srcs` all share one type).
  }];
  let arguments = (ins Variadic<GMIR_LLTType>:$srcs);
  let results = (outs GMIR_LLTType:$dst);
  let hasVerifier = 1;
}

def GMIR_BuildVectorOp : GMIR_Op<"build_vector", [Pure]> {
  let summary = "Builds a vector from scalar elements, mirroring GlobalISel's G_BUILD_VECTOR";
  let description = [{
    Emitted only by GMIRLegalizer's shared rewriteFewerElements helper
    (called from AddSubLegalizePattern's/BitwiseLegalizePattern's/
    MulLegalizePattern's FewerElements dispatch case), to reassemble a
    scalarized vector op's per-lane results into a single vector.
    Distinct from `gmir.merge` (concatenates scalars into a wider
    *scalar*, mirroring G_MERGE_VALUES): real GlobalISel itself uses a
    different opcode here (MachineIRBuilder::getOpcodeForMerge picks
    G_BUILD_VECTOR specifically for a vector destination with scalar
    sources) -- mirrored 1:1 here rather than overloading `gmir.merge`'s
    existing scalar-bit-concat verifier semantics onto a structurally
    different operation. `hasVerifier = 1`: checks `srcs` all share one
    non-pointer, non-vector type, and that `dst` is a fixed vector whose
    element count matches `srcs.size()` and element width matches --
    no stock ODS trait expresses this.
  }];
  let arguments = (ins Variadic<GMIR_LLTType>:$srcs);
  let results = (outs GMIR_LLTType:$dst);
  let hasVerifier = 1;
}

def GMIR_CallOp : GMIR_Op<"call"> {
  let summary = "Direct call, bridging to CallLowering::lowerCall";
  let description = [{
    Direct calls only (no indirect callee, no varargs, no tail/musttail, no
    byval/sret/inalloca/preallocated/byref/swifterror/swiftself/swiftasync,
    no operand bundles, not an intrinsic/inline-asm/GC statepoint call --
    see GMIRImporter.cpp's importCall for the exact bail-out list).
    `callee`/`callingConv` are descriptive attributes reconstructed from the
    original llvm::CallBase at import time for self-description -- they are
    NOT this op's translation source of truth. MLIRToGMIRTranslator instead
    recovers the original llvm::CallInst* via this op's mlir::OpaqueLoc
    location (set at import time; see importCall) and calls
    CallLowering::lowerCall's CallBase-taking overload directly on it,
    reusing its argument-flag derivation, tail-call-eligibility
    recomputation, and sret-demotion logic wholesale rather than
    re-deriving llvm::ISD::ArgFlagsTy bits by hand from IR Attributes -- see
    MLIRToGMIRTranslator.cpp's translateCall for the full rationale.
    `argLeafCounts[i]` gives the number of leaf `args` operands contributed
    by original IR argument `i` (parallel to computeGMIRLeafTypes'
    flattening of that argument's type), letting the translator regroup the
    flat `args` operand list back into the ArrayRef<ArrayRef<Register>>
    shape CallBase-based lowerCall's ArgRegs parameter expects.
    Deliberately does NOT implement CallOpInterface/SymbolUserOpInterface:
    `callee` generally does not name a sibling func.FuncOp in this module
    (GMIRImporter only ever imports the one function currently being
    compiled, never its callees), so treating it as a verifier-resolved
    module symbol would be incorrect, unlike mlir::func::CallOp's usual
    contract.
  }];
  let arguments = (ins FlatSymbolRefAttr:$callee, I64Attr:$callingConv,
                       Variadic<GMIR_LLTType>:$args,
                       DenseI32ArrayAttr:$argLeafCounts);
  let results = (outs Optional<GMIR_LLTType>:$result);
}

#endif // LLVM_CODEGEN_MLIR_IR_GMIRDIALECT_TD
