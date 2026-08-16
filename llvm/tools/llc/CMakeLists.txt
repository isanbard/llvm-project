set(LLVM_LINK_COMPONENTS
  AllTargetsAsmParsers
  AllTargetsCodeGens
  AllTargetsDescs
  AllTargetsInfos
  Analysis
  AsmPrinter
  CodeGen
  CodeGenTypes
  Core
  IPO
  IRPrinter
  IRReader
  MC
  MIRParser
  Passes
  Plugins
  Remarks
  ScalarOpts
  SelectionDAG
  Support
  Target
  TargetParser
  TransformUtils
  Vectorize
  )

# The experimental MLIR-based instruction-selection pipeline isn't a
# per-target codegen library (unlike GlobalISel/SelectionDAG, pulled in
# transitively via AllTargetsCodeGens), so it needs to be linked explicitly.
# See llvm/include/llvm/CodeGen/MLIRISel.h for why libLLVMCodeGen can't just
# link it directly.
if (LLVM_ENABLE_MLIR_ISEL)
  list(APPEND LLVM_LINK_COMPONENTS CodeGenMLIR)
endif()

add_llvm_tool(llc
  llc.cpp
  NewPMDriver.cpp

  DEPENDS
  intrinsics_gen
  SUPPORT_PLUGINS
  )

export_executable_symbols_for_plugins(llc)
