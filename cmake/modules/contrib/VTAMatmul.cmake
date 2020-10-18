if(USE_VTA_MATMUL)
  include_directories(BEFORE SYSTEM ${VTA_HW_PATH}/include)
  file(GLOB VTAMATMUL_RELAY_CONTRIB_SRC src/relay/backend/contrib/vta_matmul/codegen.cc)
  list(APPEND COMPILER_SRCS ${VTAMATMUL_RELAY_CONTRIB_SRC})
  # file(GLOB VTAMATMUL_CONTRIB_SRC src/runtime/contrib/vta_matmul/vta_matmul_runtime.cc)
  # list(APPEND RUNTIME_SRCS ${VTAMATMUL_CONTRIB_SRC})
  set(VTA_CONFIG ${PYTHON} ${VTA_HW_PATH}/config/vta_config.py)

  if(EXISTS ${CMAKE_CURRENT_BINARY_DIR}/vta_config.json)
    message(STATUS "Use VTA config " ${CMAKE_CURRENT_BINARY_DIR}/vta_config.json)
    set(VTA_CONFIG ${PYTHON} ${VTA_HW_PATH}/config/vta_config.py
      --use-cfg=${CMAKE_CURRENT_BINARY_DIR}/vta_config.json)
  endif()

  execute_process(COMMAND ${VTA_CONFIG} --target OUTPUT_VARIABLE VTA_TARGET OUTPUT_STRIP_TRAILING_WHITESPACE)

  message(STATUS "Build VTA runtime with target: " ${VTA_TARGET})

  execute_process(COMMAND ${VTA_CONFIG} --defs OUTPUT_VARIABLE __vta_defs)

  string(REGEX MATCHALL "(^| )-D[A-Za-z0-9_=.]*" VTA_DEFINITIONS "${__vta_defs}")
  # Add fsim driver sources
  file(GLOB VTA_ILA_RUNTIME_SRCS ${VTA_HW_PATH}/src/*.cc)
  file(GLOB VTA_ILA_RUNTIME_SRCS vta/runtime/*.cc)
  list(APPEND VTA_ILA_RUNTIME_SRCS ${VTA_HW_PATH}/src/sim/sim_driver.cc)
  list(APPEND VTA_ILA_RUNTIME_SRCS ${VTA_HW_PATH}/src/sim/sim_tlpp.cc)
  list(APPEND VTA_ILA_RUNTIME_SRCS ${VTA_HW_PATH}/src/vmem/virtual_memory.cc)
  list(APPEND VTA_ILA_RUNTIME_SRCS src/runtime/contrib/vta_matmul/vta_matmul_runtime.cc)
  # Target lib: vta_fsim
  add_library(vta_ila SHARED ${VTA_ILA_RUNTIME_SRCS})
  target_include_directories(vta_ila SYSTEM PUBLIC ${VTA_HW_PATH}/include)
  foreach(__def ${VTA_DEFINITIONS})
    string(SUBSTRING ${__def} 3 -1 __strip_def)
    target_compile_definitions(vta_ila PUBLIC ${__strip_def})
  endforeach()
  message(STATUS "Build with Codegen for VTA...")
endif(USE_VTA_MATMUL)