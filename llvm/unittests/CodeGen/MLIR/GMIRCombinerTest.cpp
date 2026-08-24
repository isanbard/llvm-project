//===- llvm/unittest/CodeGen/MLIR/GMIRCombinerTest.cpp ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Unit test for CombinerPatternCache's cache key (GMIRCombiner.h/.cpp).
// DisjointAddToOrPattern captures an `LLVMContext &` at cache-population
// time; the key must include it, or a cache hit for the same
// (TargetLowering*, DataLayout*) pair reused across two different
// LLVMContext/Module instances would hand back a pattern set holding a
// dangling context reference. Not reachable via a single `llc` invocation
// (one process, one Module) -- exercised directly here instead by calling
// get() with two distinct LLVMContexts sharing the same DataLayout/null
// TargetLowering.
//
//===----------------------------------------------------------------------===//

#include "GMIRCombiner.h"
#include "IR/GMIRDialect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/MLIRContext.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/LLVMContext.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace {

TEST(GMIRCombinerTest, CacheKeyDistinguishesLLVMContexts) {
  mlir::MLIRContext MLIRCtx;
  MLIRCtx.getOrLoadDialect<gmir::GMIRDialect>();
  MLIRCtx.getOrLoadDialect<mlir::func::FuncDialect>();

  DataLayout DL("");
  LLVMContext Ctx1, Ctx2;
  gmir::CombinerPatternCache Cache;

  const mlir::FrozenRewritePatternSet &Set1 =
      Cache.get(MLIRCtx, /*TLI=*/nullptr, DL, Ctx1);
  const mlir::FrozenRewritePatternSet &Set2 =
      Cache.get(MLIRCtx, /*TLI=*/nullptr, DL, Ctx2);

  // Same (TLI, DL) but different LLVMContext -- must be distinct cache
  // entries, not a hit reusing Ctx1's captured reference for Ctx2.
  EXPECT_NE(&Set1, &Set2);

  // Same (TLI, DL, Ctx) as the first call -- must be a genuine cache hit.
  const mlir::FrozenRewritePatternSet &Set1Again =
      Cache.get(MLIRCtx, /*TLI=*/nullptr, DL, Ctx1);
  EXPECT_EQ(&Set1, &Set1Again);
}

} // namespace
