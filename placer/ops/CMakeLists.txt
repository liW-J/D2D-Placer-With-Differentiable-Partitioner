add_subdirectory(parser_txt)  
add_subdirectory(power_map)
add_subdirectory(hmetis)
add_subdirectory(partition_aux)
add_subdirectory(avg_cut)
add_subdirectory(terminal_aux)
add_subdirectory(utils_3d)
add_subdirectory(multi_bipartition)

file(GLOB INSTALL_SRCS "${CMAKE_CURRENT_SOURCE_DIR}/*.py")
install(
    FILES ${INSTALL_SRCS}
    DESTINATION ${PLACER_CPP_DIR}
    )