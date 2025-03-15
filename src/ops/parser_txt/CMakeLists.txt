set(OP_NAME parser_txt)

set(TARGET_NAME ${OP_NAME})

set(INCLUDE_DIRS 
  ${CMAKE_CURRENT_SOURCE_DIR}/..
  ${DREAMPLACE_INCLUDE_DIRS}
  ${3DPLACER_INCLUDE_DIRS}
  ${LIMBO_SOURCE_DIR}
  ${CUB_DIR}
  )

message(STATUS "LIMBO_BINARY_DIR ${LIMBO_BINARY_DIR}")

set(LINK_LIBS 
  ${3DPLACER_LIBRARIES}
  OpenMP::OpenMP_CXX
  ${ZLIB_LIBRARIES}
  ${LIMBO_BINARY_DIR}/limbo/parsers/gdsii/stream/libgdsparser.a
  ${LIMBO_BINARY_DIR}/limbo/thirdparty/gzstream/libgzstream.a
  # ${DREAMPLACE_BINARY_DIR}/dreamplace/ops/utility/libutility_cxx.a
  )

add_pytorch_extension(${TARGET_NAME}_cpp 
  src/parser_txt.cpp
  EXTRA_INCLUDE_DIRS ${INCLUDE_DIRS}
  EXTRA_LINK_LIBRARIES ${LINK_LIBS})

install(TARGETS 
  ${TARGET_NAME}_cpp
  DESTINATION src/ops/${OP_NAME})

# if (TORCH_ENABLE_CUDA)
# add_pytorch_extension(${TARGET_NAME}_cuda 
#   src/k_reorder_cuda.cpp
#   src/k_reorder_cuda_kernel.cu
#   EXTRA_INCLUDE_DIRS ${INCLUDE_DIRS}
#   EXTRA_LINK_LIBRARIES ${LINK_LIBS}
#   EXTRA_DEFINITIONS DRAWPLACE=${DRAWPLACE})
# install(TARGETS 
#   ${TARGET_NAME}_cuda 
#   DESTINATION dreamplace/ops/${OP_NAME})
# endif(TORCH_ENABLE_CUDA)

file(GLOB INSTALL_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/*.py)
install(FILES ${INSTALL_SRCS} DESTINATION src/ops/${OP_NAME})
