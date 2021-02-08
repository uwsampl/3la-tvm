if(USE_ILAVTA_CODEGEN STREQUAL "ON") 
  add_definitions(-DUSE_ILAVTA_RUNTIME=1)
  file(GLOB ILAVTA_RELAY_CONTRIB_SRC src/relay/backend/contrib/ilavta/*.cc)
  list(APPEND COMPILER_SRCS ${ILAVTA_RELAY_CONTRIB_SRC})
  list(APPEND COMPILER_SRCS ${JSON_RELAY_CONTRIB_SRC})

  file(GLOB ILAVTA_CONTRIB_SRC src/runtime/contrib/ilavta/ilavta_runtime.cc)
  list(APPEND RUNTIME_SRCS ${ILAVTA_CONTRIB_SRC})
endif()
