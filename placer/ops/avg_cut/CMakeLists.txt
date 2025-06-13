set(OP_NAME avg_cut)

set(TARGET_NAME ${OP_NAME})

set(INCLUDE_DIRS 
  ${CMAKE_CURRENT_SOURCE_DIR}/..
  ${DREAMPLACE_OPS_DIR}
  ${LIMBO_SOURCE_DIR}
  ${Boost_INCLUDE_DIRS}
  ${ZLIB_INCLUDE_DIRS}
  ${3DPLACER_INCLUDE_DIRS}
  )

set(LINK_LIBS 
  utility_cxx
  utils_3d_cxx
  OpenMP::OpenMP_CXX
  ${ZLIB_LIBRARIES}
  3d_placer_lib
  )

add_pytorch_extension(${TARGET_NAME}_cpp 
  src/avg_cut.cpp
  EXTRA_INCLUDE_DIRS ${INCLUDE_DIRS}
  EXTRA_LINK_LIBRARIES ${LINK_LIBS})

install(TARGETS 
  ${TARGET_NAME}_cpp
  DESTINATION placer/ops/${OP_NAME}
  )

file(GLOB INSTALL_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/*.py)
install(FILES ${INSTALL_SRCS} DESTINATION placer/ops/${OP_NAME})