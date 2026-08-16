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

def GMIR_AddOp  : GMIR_BinOp<"add",  [Commutative]>;
def GMIR_SubOp  : GMIR_BinOp<"sub">;
def GMIR_MulOp  : GMIR_BinOp<"mul",  [Commutative]>;
def GMIR_SDivOp : GMIR_BinOp<"sdiv">;
def GMIR_AndOp  : GMIR_BinOp<"and",  [Commutative]>;
def GMIR_OrOp   : GMIR_BinOp<"or",   [Commutative]>;
def GMIR_XorOp  : GMIR_BinOp<"xor",  [Commutative]>;

def GMIR_ConstantOp : GMIR_Op<"constant", [Pure, ConstantLike]> {
  let summary = "A constant of `!gmir.llt` type";
  let description = [{
    Holds a 64-bit value (signless: the bit pattern, not a signedness
    claim -- signedness is a property of the consuming op, e.g. `gmir.sdiv`).
    Constants that don't fit in 64 bits are unsupported for now (M1 is
    scoped to i32/i64 scalar arithmetic).
  }];
  let arguments = (ins I64Attr:$value);
  let results = (outs GMIR_LLTType:$res);
  let assemblyFormat = "$value attr-dict `:` type($res)";
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
  }];
  let arguments = (ins GMIR_LLTType:$ptr, I64Attr:$align, I64Attr:$ordering,
                       I64Attr:$syncscope, UnitAttr:$isVolatile);
  let results = (outs GMIR_LLTType:$res);
}

def GMIR_StoreOp : GMIR_Op<"store",
    [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "Store to memory, mirroring GlobalISel's G_STORE";
  let description = [{
    See `gmir.load` for the `ordering`/`syncscope` attribute convention and
    the "one op per flattened leaf" aggregate-decomposition approach.
  }];
  let arguments = (ins GMIR_LLTType:$value, GMIR_LLTType:$ptr, I64Attr:$align,
                       I64Attr:$ordering, I64Attr:$syncscope,
                       UnitAttr:$isVolatile);
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

#endif // LLVM_CODEGEN_MLIR_IR_GMIRDIALECT_TD
