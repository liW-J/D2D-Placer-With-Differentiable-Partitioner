add_subdirectory(parser_txt)  
add_subdirectory(power_map)
add_subdirectory(hmetis)
add_subdirectory(partition)

file(GLOB INSTALL_SRCS "${CMAKE_CURRENT_SOURCE_DIR}/*.py")
install(
    FILES ${INSTALL_SRCS}
    DESTINATION ${PLACER_CPP_DIR}
    )