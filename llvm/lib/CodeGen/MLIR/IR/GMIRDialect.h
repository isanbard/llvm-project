//===- GMIRDialect.h - GMIR dialect --------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the `gmir` dialect. See GMIRDialect.td for the
// rationale; this header is only compiled as part of the optional
// LLVMCodeGenMLIR component (see llvm/lib/CodeGen/MLIR/CMakeLists.txt).
//
// Deliberately lives under llvm/lib/ rather than llvm/include/: it has no
// consumers outside this component, and llvm/include/ is processed before
// mlir/ is registered (see llvm/lib/CodeGen/MLIR/CMakeLists.txt), so a
// TableGen'd header living there would hit the same mlir_tablegen-ordering
// problem this component's placement under llvm/tools/ already works around.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_MLIR_IR_GMIRDIALECT_H
#define LLVM_CODEGEN_MLIR_IR_GMIRDIALECT_H

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "IR/GMIRDialect.h.inc"

#define GET_TYPEDEF_CLASSES
#include "IR/GMIRDialectTypes.h.inc"

#define GET_OP_CLASSES
#include "IR/GMIRDialectOps.h.inc"

#endif // LLVM_CODEGEN_MLIR_IR_GMIRDIALECT_H
