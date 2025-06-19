set(OP_NAME hpwl_d2d)

set(TARGET_NAME ${OP_NAME})

set(INCLUDE_DIRS 
  ${CMAKE_CURRENT_SOURCE_DIR}/..
  ${DREAMPLACE_OPS_DIR}
  ${3DPLACER_INCLUDE_DIRS}
  )

set(LINK_LIBS 
  utility_cxx
  utils_3d_cxx
  OpenMP::OpenMP_CXX
  3d_placer_lib
  )

add_pytorch_extension(${TARGET_NAME}_cpp 
  src/hpwl_d2d.cpp
  EXTRA_INCLUDE_DIRS ${INCLUDE_DIRS}
  EXTRA_LINK_LIBRARIES ${LINK_LIBS})

install(TARGETS 
  ${TARGET_NAME}_cpp
  DESTINATION placer/ops/${OP_NAME}
  )

file(GLOB INSTALL_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/*.py)
install(FILES ${INSTALL_SRCS} DESTINATION placer/ops/${OP_NAME})