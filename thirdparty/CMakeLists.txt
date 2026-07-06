#project(thirdparty)

set(HMETIS_RUNTIME_DIR ${CMAKE_CURRENT_SOURCE_DIR}/hmetis)
set(HMETIS_WRAPPER ${HMETIS_RUNTIME_DIR}/hmetis)
set(HMETIS_BINARY ${HMETIS_RUNTIME_DIR}/hmetis2.0pre1)

if(EXISTS ${HMETIS_WRAPPER} AND EXISTS ${HMETIS_BINARY})
  install(PROGRAMS ${HMETIS_WRAPPER} ${HMETIS_BINARY} DESTINATION bin)
else()
  message(WARNING
    "hMETIS runtime files not found under ${HMETIS_RUNTIME_DIR}; "
    "partitioner: hmetis and SpecPart hMETIS preprocessing will require "
    "bin/hmetis to be provided manually.")
endif()

add_subdirectory(3d-placer)
add_subdirectory(DREAMPlace)
add_subdirectory(Differentiable-3D-Partitioner)