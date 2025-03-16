#project(thirdparty)

add_subdirectory(3d-placer)
add_subdirectory(DREAMPlace)

# include(ExternalProject)

# ExternalProject_Add(DREAMPlace
#     SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/DREAMPlace
#     CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${CMAKE_INSTALL_PREFIX}/thirdparty/DREAMPlace
# )