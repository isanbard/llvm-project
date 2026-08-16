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
// instruction-selection pipeline. This defines the dialect and its
// `!gmir.llt` type (mirroring GlobalISel's LLT) only -- operations are
// added incrementally as the pipeline gains real translation coverage.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_MLIR_IR_GMIRDIALECT_TD
#define LLVM_CODEGEN_MLIR_IR_GMIRDIALECT_TD

include "mlir/IR/OpBase.td"

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

#endif // LLVM_CODEGEN_MLIR_IR_GMIRDIALECT_TD
