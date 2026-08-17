# This component is deliberately NOT reached via llvm/lib/CodeGen/CMakeLists.txt's
# own add_subdirectory (unlike SelectionDAG/AsmPrinter/MIRParser/GlobalISel):
# MLIR's CMake machinery (mlir_tablegen, MLIR component targets) isn't
# available yet when llvm/lib/ is processed -- it's only loaded once mlir's
# own CMakeLists.txt runs, which happens later, from llvm/tools/CMakeLists.txt.
# So this directory is instead add_subdirectory()'d directly from
# llvm/tools/CMakeLists.txt, after MLIR has been registered there. See
# llvm/include/llvm/CodeGen/MLIRISel.h for how the resulting library gets
# wired back into libLLVMCodeGen without a build-time dependency cycle.

include_directories(${LLVM_MAIN_SRC_DIR}/../mlir/include)
include_directories(${LLVM_BINARY_DIR}/tools/mlir/include)

set(LLVM_TARGET_DEFINITIONS IR/GMIRDialect.td)
mlir_tablegen(IR/GMIRDialect.h.inc -gen-dialect-decls)
mlir_tablegen(IR/GMIRDialect.cpp.inc -gen-dialect-defs)
mlir_tablegen(IR/GMIRDialectTypes.h.inc -gen-typedef-decls)
mlir_tablegen(IR/GMIRDialectTypes.cpp.inc -gen-typedef-defs)
mlir_tablegen(IR/GMIRDialectOps.h.inc -gen-op-decls)
mlir_tablegen(IR/GMIRDialectOps.cpp.inc -gen-op-defs)
add_public_tablegen_target(MLIRGMIRDialectIncGen)

add_llvm_component_library(LLVMCodeGenMLIR
  GMIRImporter.cpp
  GMIRLegalizer.cpp
  IR/GMIRDialect.cpp
  MLIRInstructionSelect.cpp
  MLIRToGMIRTranslator.cpp

  DEPENDS
  MLIRGMIRDialectIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRFuncDialect
  MLIRTransformUtils

  LINK_COMPONENTS
  CodeGen
  GlobalISel
  Support
  )
