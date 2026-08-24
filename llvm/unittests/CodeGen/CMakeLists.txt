set(LLVM_LINK_COMPONENTS
  ${LLVM_TARGETS_TO_BUILD}
  Analysis
  AsmParser
  AsmPrinter
  CodeGen
  CodeGenTypes
  Core
  FileCheck
  IRPrinter
  MC
  MIRParser
  Passes
  ScalarOpts
  SelectionDAG
  Support
  Target
  TargetParser
  TransformUtils
  )

add_llvm_unittest(CodeGenTests
  AllocationOrderTest.cpp
  AMDGPUMetadataTest.cpp
  AsmPrinterDwarfTest.cpp
  CCStateTest.cpp
  DIEHashTest.cpp
  DIETest.cpp
  DroppedVariableStatsMIRTest.cpp
  DwarfStringPoolEntryRefTest.cpp
  GCMetadata.cpp
  InstrRefLDVTest.cpp
  LowLevelTypeTest.cpp
  LexicalScopesTest.cpp
  MachineBasicBlockTest.cpp
  MachineDomTreeUpdaterTest.cpp
  MachineInstrBundleIteratorTest.cpp
  MachineInstrTest.cpp
  MachineModuleInfoTest.cpp
  MachineOperandTest.cpp
  MIR2VecTest.cpp
  RegAllocBasicTest.cpp
  RegAllocScoreTest.cpp
  RegisterTest.cpp
  PassManagerTest.cpp
  RematerializerTest.cpp
  ScalableVectorMVTsTest.cpp
  SchedBoundary.cpp
  SelectionDAGAddressAnalysisTest.cpp
  SelectionDAGNodeConstructionTest.cpp
  SelectionDAGPatternMatchTest.cpp
  TypeTraitsTest.cpp
  TargetOptionsTest.cpp
  TestAsmPrinter.cpp
  X86MCInstLowerTest.cpp
  )

add_subdirectory(GlobalISel)
add_subdirectory(CGPluginTest)

# The CodeGenMLIR component this tests doesn't exist unless
# LLVM_ENABLE_MLIR_ISEL is on (see llvm/lib/CodeGen/MLIR/CMakeLists.txt) --
# guard the whole subdirectory the same way llvm/tools/CMakeLists.txt guards
# the library itself, rather than trying to reference a nonexistent
# component from an unconditional CMakeLists.txt.
if (LLVM_ENABLE_MLIR_ISEL)
  add_subdirectory(MLIR)
endif()

target_link_libraries(CodeGenTests PRIVATE LLVMTestingSupport)
