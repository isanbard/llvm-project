//===- GMIRDialect.cpp - GMIR dialect implementation ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Registers the `gmir` dialect and its `!gmir.llt` type.
//
//===----------------------------------------------------------------------===//

#include "GMIRDialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace llvm::gmir;

#include "IR/GMIRDialect.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "IR/GMIRDialectTypes.cpp.inc"

#define GET_OP_CLASSES
#include "IR/GMIRDialectOps.cpp.inc"

// BranchOpInterface methods for GMIR_BrOp/GMIR_CondBrOp -- ODS only
// declares these (DeclareOpInterfaceMethods), it doesn't define them.
// Mirrors mlir::cf::BranchOp/CondBranchOp exactly
// (mlir/lib/Dialect/ControlFlow/IR/ControlFlowOps.cpp:290-297,573-584).
// Note: llvm::Attribute (IR function/param attributes) and mlir::Attribute
// are distinct types that collide under this file's `using namespace mlir;`
// plus the ambient `using namespace llvm;` pulled in via the CodeGen PCH --
// explicitly qualified below for the same reason M1's Value/Type/DenseMap
// ambiguities were.
mlir::SuccessorOperands BrOp::getSuccessorOperands(unsigned index) {
  assert(index == 0 && "invalid successor index");
  return mlir::SuccessorOperands(getDestOperandsMutable());
}

Block *BrOp::getSuccessorForOperands(ArrayRef<mlir::Attribute>) {
  return getDest();
}

mlir::SuccessorOperands CondBrOp::getSuccessorOperands(unsigned index) {
  assert(index < 2 && "invalid successor index");
  return mlir::SuccessorOperands(index == 0 ? getTrueDestOperandsMutable()
                                             : getFalseDestOperandsMutable());
}

Block *CondBrOp::getSuccessorForOperands(ArrayRef<mlir::Attribute> operands) {
  if (mlir::IntegerAttr condAttr =
          dyn_cast_or_null<mlir::IntegerAttr>(operands.front()))
    return condAttr.getValue().isOne() ? getTrueDest() : getFalseDest();
  return nullptr;
}

// MemoryEffectsOpInterface methods for GMIR_LoadOp/GMIR_StoreOp -- ODS only
// declares these (DeclareOpInterfaceMethods), same reason as BranchOpInterface
// above. Mirrors mlir::LLVM::LoadOp/StoreOp::getEffects exactly
// (mlir/lib/Dialect/LLVMIR/IR/LLVMDialect.cpp:839-853,918-932): a plain
// Read/Write on the pointer operand, plus a conservative extra Read+Write
// pair when the access is volatile or "stronger than unordered" atomic,
// since such accesses can have target-specific effects beyond the single
// pointed-to location.
// mlir::MemoryEffects/mlir::SideEffects collide with llvm::MemoryEffects
// (IR-level function/call memory-effect attributes) under this file's
// blanket `using namespace mlir;` + the ambient `using namespace llvm;` --
// same class of ambiguity as Value/Type/Attribute/DataLayout above,
// explicitly qualified for the same reason.
void LoadOp::getEffects(SmallVectorImpl<mlir::SideEffects::EffectInstance<
                            mlir::MemoryEffects::Effect>> &effects) {
  effects.emplace_back(mlir::MemoryEffects::Read::get(), &getPtrMutable());
  if (getIsVolatile() || getOrderingAttr().getInt() != 0) {
    effects.emplace_back(mlir::MemoryEffects::Write::get());
    effects.emplace_back(mlir::MemoryEffects::Read::get());
  }
}

void StoreOp::getEffects(SmallVectorImpl<mlir::SideEffects::EffectInstance<
                             mlir::MemoryEffects::Effect>> &effects) {
  effects.emplace_back(mlir::MemoryEffects::Write::get(), &getPtrMutable());
  if (getIsVolatile() || getOrderingAttr().getInt() != 0) {
    effects.emplace_back(mlir::MemoryEffects::Write::get());
    effects.emplace_back(mlir::MemoryEffects::Read::get());
  }
}

void GMIRDialect::initialize() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "IR/GMIRDialectTypes.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "IR/GMIRDialectOps.cpp.inc"
      >();
}
