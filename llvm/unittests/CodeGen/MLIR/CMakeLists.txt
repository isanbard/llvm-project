# This whole directory is only add_subdirectory()'d (from
# llvm/unittests/CodeGen/CMakeLists.txt) when LLVM_ENABLE_MLIR_ISEL is ON --
# the CodeGenMLIR component it tests doesn't exist otherwise. Safe to link
# against it unconditionally here (unlike llvm/lib/CodeGen/MLIR/ itself,
# which needs the llvm/tools/CMakeLists.txt add_subdirectory ordering trick
# so mlir_tablegen is available): llvm/CMakeLists.txt's top-level
# add_subdirectory(unittests) runs after add_subdirectory(tools), so
# CodeGenMLIR is already registered by the time this file is processed.

include_directories(${LLVM_MAIN_SRC_DIR}/lib/CodeGen/MLIR)
include_directories(${LLVM_BINARY_DIR}/lib/CodeGen/MLIR)
include_directories(${LLVM_MAIN_SRC_DIR}/../mlir/include)
include_directories(${LLVM_BINARY_DIR}/tools/mlir/include)

set(LLVM_LINK_COMPONENTS
  CodeGen
  CodeGenMLIR
  CodeGenTypes
  Core
  GlobalISel
  Support
  Target
  )

add_llvm_unittest(MLIRCodeGenTests
  GMIRCombinerTest.cpp
  GMIRLegalizerTest.cpp
  )

target_link_libraries(MLIRCodeGenTests PRIVATE
  MLIRIR
  MLIRFuncDialect
  )
