add_subdirectory(ops)
add_subdirectory(tools)
add_subdirectory(scripts)

file(GLOB INSTALL_SRCS "${CMAKE_CURRENT_SOURCE_DIR}/*.py")
install(
    FILES ${INSTALL_SRCS} DESTINATION placer
    )
