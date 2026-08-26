//===- llvm/unittest/CodeGen/MLIR/GMIRLLTConversionTest.cpp ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Unit tests for gmir::convertLLT/convertToGMIRType (GMIRLLTConversion.h),
// in particular the scalable-vector bit. No current gmir producer ever
// builds a scalable vector LLT or !gmir.llt (there's no parser entry point
// for hand-authored gmir text, and every import-side LLT is already
// fixed-width -- see convertType in GMIRImporter.cpp), so that round trip
// can't be exercised via a real `llc -enable-mlir-isel` invocation. These
// tests exercise it directly against synthetic LLT/LLTType values instead.
//
//===----------------------------------------------------------------------===//

#include "GMIRLLTConversion.h"
#include "gtest/gtest.h"

using namespace llvm;
using namespace llvm::gmir;

namespace {

class GMIRLLTConversionTest : public testing::Test {
protected:
  GMIRLLTConversionTest() { Context.getOrLoadDialect<gmir::GMIRDialect>(); }

  mlir::MLIRContext Context;
};

TEST_F(GMIRLLTConversionTest, FixedVectorRoundTrips) {
  LLT V4S32 = LLT::vector(ElementCount::getFixed(4), LLT::integer(32));
  gmir::LLTType GTy = convertToGMIRType(Context, V4S32);
  EXPECT_FALSE(GTy.getIsScalable());
  EXPECT_EQ(GTy.getNumElements(), 4u);

  DataLayout DL("");
  EXPECT_EQ(convertLLT(GTy, DL), V4S32);
}

TEST_F(GMIRLLTConversionTest, ScalableVectorRoundTrips) {
  // Regression test: convertToGMIRType used to hardcode isScalable=false,
  // and convertLLT used to always build ElementCount::getFixed(...) --
  // either bug would silently turn a scalable vector into a same-lane-count
  // fixed vector instead of preserving the distinction, which is wrong even
  // though no current caller happens to hit this path yet.
  LLT NxV4S32 = LLT::vector(ElementCount::getScalable(4), LLT::integer(32));
  gmir::LLTType GTy = convertToGMIRType(Context, NxV4S32);
  EXPECT_TRUE(GTy.getIsScalable());
  EXPECT_EQ(GTy.getNumElements(), 4u);

  DataLayout DL("");
  EXPECT_EQ(convertLLT(GTy, DL), NxV4S32);
}

} // namespace
