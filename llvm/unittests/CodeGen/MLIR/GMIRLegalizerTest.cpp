//===- llvm/unittest/CodeGen/MLIR/GMIRLegalizerTest.cpp -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Unit tests for gmir::getExactNarrowScalarSplit (GMIRLegalizer.cpp). No
// in-tree target's real LegalizerInfo currently produces a NarrowScalar
// LegalizeActionStep for a vector-typed gmir op (every vector case in this
// pipeline goes through FewerElements down to scalar elements first, and
// NarrowScalar only ever fires afterward on an already-scalar element) --
// so the vector-shape guard this function relies on can't be exercised via
// a real `llc -enable-mlir-isel` invocation. These tests exercise it
// directly against synthetic LegalizeActionStep/LLT values instead.
//
//===----------------------------------------------------------------------===//

#include "GMIRLegalizer.h"
#include "gtest/gtest.h"

using namespace llvm;
using namespace llvm::gmir;

namespace {

TEST(GMIRLegalizerTest, NonNarrowScalarActionIsRejected) {
  LLT S64 = LLT::integer(64);
  LegalizeActionStep Step(LegalizeActions::Legal, 0, LLT::integer(32));
  EXPECT_EQ(getExactNarrowScalarSplit(Step, S64), std::nullopt);
}

TEST(GMIRLegalizerTest, ExactScalarSplitIsAccepted) {
  LLT S64 = LLT::integer(64);
  LLT S32 = LLT::integer(32);
  LegalizeActionStep Step(LegalizeActions::NarrowScalar, 0, S32);
  auto Split = getExactNarrowScalarSplit(Step, S64);
  ASSERT_TRUE(Split.has_value());
  EXPECT_EQ(Split->first, S32);
  EXPECT_EQ(Split->second, 2u);
}

TEST(GMIRLegalizerTest, NonExactLeftoverSplitIsRejected) {
  // 64 / 24 isn't an integer -- no exact, leftover-free split exists.
  LLT S64 = LLT::integer(64);
  LLT S24 = LLT::integer(24);
  LegalizeActionStep Step(LegalizeActions::NarrowScalar, 0, S24);
  EXPECT_EQ(getExactNarrowScalarSplit(Step, S64), std::nullopt);
}

TEST(GMIRLegalizerTest, VectorDstTypeIsRejected) {
  // A real target's LegalizerInfo can report NarrowScalar for a vector
  // query too (narrowing each element's width while keeping the element
  // count) -- getExactNarrowScalarSplit must not misread that as an
  // N-way scalar limb split of the whole vector's bit width. Regression
  // test for the vector-shape guard: without it, DstBits (128) / NarrowBits
  // (32) would wrongly compute NumParts=4 here.
  LLT V4S32 = LLT::vector(ElementCount::getFixed(4), LLT::integer(32));
  LLT S32 = LLT::integer(32);
  LegalizeActionStep Step(LegalizeActions::NarrowScalar, 0, S32);
  EXPECT_EQ(getExactNarrowScalarSplit(Step, V4S32), std::nullopt);
}

TEST(GMIRLegalizerTest, VectorNarrowTypeIsRejected) {
  // Same guard, exercised via Step.NewType being the vector instead of
  // DstTy -- both sides need the check, not just one. DstTy is 128 bits
  // and NarrowTy (a 2x32-bit vector) is 64 bits total, so without the
  // guard this would wrongly compute an exact NumParts=2 split via plain
  // bit-count division (128/64), the same way VectorDstTypeIsRejected's
  // repro does -- a NarrowTy.getSizeInBits() of 2 vector-lanes divides
  // evenly just like a plain scalar's would.
  LLT S128 = LLT::integer(128);
  LLT V2S32 = LLT::vector(ElementCount::getFixed(2), LLT::integer(32));
  LegalizeActionStep Step(LegalizeActions::NarrowScalar, 0, V2S32);
  EXPECT_EQ(getExactNarrowScalarSplit(Step, S128), std::nullopt);
}

} // namespace
