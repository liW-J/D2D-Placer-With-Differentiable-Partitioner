add_subdirectory(d2d_result_analyzer)
add_subdirectory(thirdparty_api)

file(GLOB INSTALL_SRCS "${CMAKE_CURRENT_SOURCE_DIR}/*.py")

install(
    FILES ${INSTALL_SRCS} DESTINATION placer/tools
    )