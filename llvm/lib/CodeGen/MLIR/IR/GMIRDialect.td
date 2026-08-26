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
// instruction-selection pipeline. Operations are added incrementally as
// the pipeline gains real translation coverage.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_MLIR_IR_GMIRDIALECT_TD
#define LLVM_CODEGEN_MLIR_IR_GMIRDIALECT_TD

include "mlir/IR/OpBase.td"
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
    default derived assembly format for now rather than LLT's compact
    `s32`/`v4s32`/`p0`/`nxv4s32` textual syntax -- the compact custom
    parser/printer is deferred until real `gmir` ops exist to exercise it.
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
    Constants that don't fit in 64 bits are unsupported for now (this
    pipeline is currently scoped to i32/i64 scalar arithmetic).
  }];
  let arguments = (ins I64Attr:$value);
  let results = (outs GMIR_LLTType:$res);
  let assemblyFormat = "$value attr-dict `:` type($res)";
}

#endif // LLVM_CODEGEN_MLIR_IR_GMIRDIALECT_TD
