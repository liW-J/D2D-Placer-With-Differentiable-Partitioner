add_subdirectory(parser_txt)  
add_subdirectory(power_map)
add_subdirectory(hmetis)
add_subdirectory(partition_aux)
add_subdirectory(avg_cut)
add_subdirectory(terminal_aux)
add_subdirectory(utils_3d)
add_subdirectory(multi_bipartition)
add_subdirectory(refinement)
add_subdirectory(hpwl_d2d)
add_subdirectory(macro_balance)
add_subdirectory(parts_reader)
add_subdirectory(bin_based_fm)

file(GLOB INSTALL_SRCS "${CMAKE_CURRENT_SOURCE_DIR}/*.py")
install(
    FILES ${INSTALL_SRCS}
    DESTINATION ${PLACER_CPP_DIR}
    )