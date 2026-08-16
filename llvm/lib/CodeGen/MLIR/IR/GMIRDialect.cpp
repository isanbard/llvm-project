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
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace llvm::gmir;

#include "IR/GMIRDialect.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "IR/GMIRDialectTypes.cpp.inc"

#define GET_OP_CLASSES
#include "IR/GMIRDialectOps.cpp.inc"

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
