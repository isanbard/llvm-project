//===- llvm/unittest/CodeGen/MLIR/GMIRImporterTest.cpp --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Unit test for GMIRImporter.cpp's importAlloca int64 overflow guard. The
// declined-alloca case is also covered end-to-end in
// llvm/test/CodeGen/MLIR/mlir-isel-fallback.ll (which checks the whole
// pipeline falls back to identical codegen), but reaching that fallback
// still requires the real backend to lay out a stack frame for the huge
// alloca. Testing importFunction directly here avoids depending on that
// being safe for arbitrarily large sizes on every target.
//
//===----------------------------------------------------------------------===//

#include "GMIRImporter.h"
#include "IR/GMIRDialect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/SourceMgr.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace {

class GMIRImporterTest : public testing::Test {
protected:
  GMIRImporterTest() {
    Context.getOrLoadDialect<gmir::GMIRDialect>();
    Context.getOrLoadDialect<mlir::func::FuncDialect>();
  }

  std::unique_ptr<Module> parseIR(StringRef Assembly) {
    SMDiagnostic Err;
    std::unique_ptr<Module> M = parseAssemblyString(Assembly, Err, LLVMCtx);
    if (!M)
      Err.print("GMIRImporterTest", errs());
    return M;
  }

  mlir::MLIRContext Context;
  LLVMContext LLVMCtx;
};

TEST_F(GMIRImporterTest, OversizedAllocaIsDeclined) {
  // 2^63 bytes: fits in uint64_t (so the existing arraySize*elementSize
  // overflow check doesn't catch it), but overflows the int64_t attribute
  // AllocaOp stores its size in.
  std::unique_ptr<Module> M = parseIR(R"(
    define i8 @huge_alloca() {
    entry:
      %p = alloca i8, i64 9223372036854775808
      store i8 0, ptr %p
      %v = load i8, ptr %p
      ret i8 %v
    }
  )");
  ASSERT_TRUE(M);

  mlir::OwningOpRef<mlir::ModuleOp> GMIRModule(
      mlir::ModuleOp::create(mlir::UnknownLoc::get(&Context)));
  gmir::CallInstMap CallInsts;
  mlir::func::FuncOp FuncOp = gmir::importFunction(
      *GMIRModule, *M->getFunction("huge_alloca"), CallInsts);
  EXPECT_FALSE(FuncOp);
}

TEST_F(GMIRImporterTest, OrdinaryAllocaIsImported) {
  std::unique_ptr<Module> M = parseIR(R"(
    define i8 @small_alloca() {
    entry:
      %p = alloca i8, i64 16
      store i8 0, ptr %p
      %v = load i8, ptr %p
      ret i8 %v
    }
  )");
  ASSERT_TRUE(M);

  mlir::OwningOpRef<mlir::ModuleOp> GMIRModule(
      mlir::ModuleOp::create(mlir::UnknownLoc::get(&Context)));
  gmir::CallInstMap CallInsts;
  mlir::func::FuncOp FuncOp = gmir::importFunction(
      *GMIRModule, *M->getFunction("small_alloca"), CallInsts);
  EXPECT_TRUE(FuncOp);
}

} // namespace
