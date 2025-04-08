set(OP_NAME parser_txt)

set(TARGET_NAME ${OP_NAME})

set(INCLUDE_DIRS 
  ${CMAKE_CURRENT_SOURCE_DIR}/..
  ${DREAMPLACE_OPS_DIR}
  ${3DPLACER_INCLUDE_DIRS}
  ${LIMBO_SOURCE_DIR}
  ${Boost_INCLUDE_DIRS}
  ${ZLIB_INCLUDE_DIRS}
  )

set(LINK_LIBS 
  utility_cxx
  # place_io_cxx
  3d_placer_lib
  OpenMP::OpenMP_CXX
  ${ZLIB_LIBRARIES}
  )

add_pytorch_extension(${TARGET_NAME}_cpp 
  src/parser_txt.cpp
  EXTRA_INCLUDE_DIRS ${INCLUDE_DIRS}
  EXTRA_LINK_LIBRARIES ${LINK_LIBS})

install(TARGETS 
  ${TARGET_NAME}_cpp
  DESTINATION placer/ops/${OP_NAME}
  )

# if (TORCH_ENABLE_CUDA)
# add_pytorch_extension(${TARGET_NAME}_cuda 
#   src/k_reorder_cuda.cpp
#   src/k_reorder_cuda_kernel.cu
#   EXTRA_INCLUDE_DIRS ${INCLUDE_DIRS}
#   EXTRA_LINK_LIBRARIES ${LINK_LIBS}
#   EXTRA_DEFINITIONS DRAWPLACE=${DRAWPLACE})
# install(TARGETS 
#   ${TARGET_NAME}_cuda 
#   DESTINATION thirdparty/DREAMPlace/dreamplace/ops/${OP_NAME})
# endif(TORCH_ENABLE_CUDA)

file(GLOB INSTALL_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/*.py)
install(FILES ${INSTALL_SRCS} DESTINATION placer/ops/${OP_NAME})