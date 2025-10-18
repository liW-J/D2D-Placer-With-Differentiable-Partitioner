'''
Author: JeanneWillis hi@jeannewillis.cn
Date: 2025-02-24 18:32:05
LastEditors: JeanneWillis hi@jeannewillis.cn
LastEditTime: 2025-10-19 02:53:56
FilePath: /D2D-placer/config/configure.py.in
Description: 
'''

import sys

sys.path.append("/home/placer/D2D-placer/install/thirdparty/DREAMPlace")
sys.path.append(
    "/home/placer/D2D-placer/install/thirdparty/DREAMPlace/dreamplace")
sys.path.append("/home/placer/D2D-placer/install/src")

compile_configurations = {
    "CMAKE_CXX_COMPILER": "/usr/bin/c++",
    "CMAKE_CC_COMPILER": "",
    "CMAKE_BUILD_TYPE": "Release",
    "CMAKE_CXX_ABI": "1",
    "CMAKE_CXX_STANDARD": "17",
    "CMAKE_INSTALL_PREFIX": "/home/placer/D2D-placer/install",
    "PYTHON": "",
    "Boost_DIR": "",
    "Boost_INCLUDE_DIRS": "",
    "ZLIB_INCLUDE_DIRS": "/usr/include",
    "ZLIB_LIBRARIES": "/usr/lib/x86_64-linux-gnu/libz.so",
    "CUDA_FOUND": "",
    "CUDA_TOOLKIT_ROOT_DIR": "",
    "CMAKE_CUDA_FLAGS": "",
    "CAIRO_FOUND": "",
    "CAIRO_INCLUDE_DIRS": "/usr/include/cairo",
    "CAIRO_LIBRARIES": "/usr/lib/x86_64-linux-gnu/libcairo.so",
    "PLACER_CPP_DIR": "/home/placer/D2D-placer/install/placer/ops",
    "PLACER_CPP_INCLUDE_DIR":
    "/home/placer/D2D-placer/install/placer/ops/include",
    "PLACER_UNITTES_DIR": "/home/placer/D2D-placer/install/unittest",
    "PLACER_BENCHMARKS_DIR": "/home/placer/D2D-placer/install/benchmarks",
    "PLACER_TEST_DIR": "/home/placer/D2D-placer/install/test",
    "PLACER_THIRDPARTY_DIR": "/home/placer/D2D-placer/install/thirdparty",
    "PLACER_RESULT_DIR": "/home/placer/D2D-placer/install/unittest/refinement_unittest/results",
    "PLACER_RUNTMP_DIR": "/home/placer/D2D-placer/install/unittest/refinement_unittest/run_tmp",
    "PLACER_SOURCE_DIR": "/home/placer/D2D-placer/placer",
    "PLACER_INSTALL_DIR": "/home/placer/D2D-placer/install/placer",
    "PLACER_BUILD_DIR": "",
}
