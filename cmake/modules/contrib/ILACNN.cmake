if(USE_ILACNN_CODEGEN STREQUAL "ON")
  add_definitions(-DUSE_ILACNN_RUNTIME=1)
  file(GLOB ILACNN_RELAY_CONTRIB_SRC src/relay/backend/contrib/ilacnn/*.cc)
  list(APPEND COMPILER_SRCS ${ILACNN_RELAY_CONTRIB_SRC})
  list(APPEND COMPILER_SRCS ${JSON_RELAY_CONTRIB_SRC})

  file(GLOB ILACNN_CONTRIB_SRC src/runtime/contrib/ilacnn/ilacnn_runtime.cc)
  list(APPEND RUNTIME_SRCS ${ILACNN_CONTRIB_SRC})
endif()