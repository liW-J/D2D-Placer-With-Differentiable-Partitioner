# @file TorchExtension.cmake
# @author Zizheng Guo
# @brief Use CMake to compile PyTorch extensions

# Activate new FindPython mode, specified in pybind11Config.cmake.in...
# This one is recommended from CMake 3.12+.
# It should try to find the Python associated with the environment variable.
if(NOT DEFINED DREAMPLACE_ROOT)
    set(DREAMPLACE_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/DREAMPlace)
endif()

find_package(Python COMPONENTS Interpreter Development)
if(NOT D2D_TORCH_EXTENSION_DEFER_TARGETS)
  add_subdirectory(${DREAMPLACE_ROOT}/thirdparty/pybind11)
endif()

execute_process(COMMAND ${Python_EXECUTABLE} -c 
  "import torch; print(torch.__path__[0]); print(int(getattr(torch.cuda, '_is_compiled', lambda: torch.version.cuda is not None)())); print(torch.__version__); print(torch.version.cuda or '');"
  OUTPUT_VARIABLE TORCH_OUTPUT OUTPUT_STRIP_TRAILING_WHITESPACE)
string(REPLACE "\n" ";" TORCH_OUTPUT_LIST ${TORCH_OUTPUT})
list(GET TORCH_OUTPUT_LIST 0 TORCH_INSTALL_PREFIX)
list(GET TORCH_OUTPUT_LIST 1 TORCH_ENABLE_CUDA)
list(GET TORCH_OUTPUT_LIST 2 TORCH_VERSION)
list(GET TORCH_OUTPUT_LIST 3 TORCH_CUDA_VERSION)
string(REPLACE "." ";" TORCH_VERSION_LIST ${TORCH_VERSION})
list(GET TORCH_VERSION_LIST 0 TORCH_VERSION_MAJOR)
list(GET TORCH_VERSION_LIST 1 TORCH_VERSION_MINOR)

message(STATUS TORCH_INSTALL_PREFIX=${TORCH_INSTALL_PREFIX})
message(STATUS TORCH_VERSION=${TORCH_VERSION_MAJOR}.${TORCH_VERSION_MINOR})
message(STATUS TORCH_CUDA_VERSION=${TORCH_CUDA_VERSION})

if ("${TORCH_VERSION_MAJOR}.${TORCH_VERSION_MINOR}" VERSION_LESS 1.6)
  message(SEND_ERROR "require PyTorch version >=1.6")
#elseif ("${TORCH_VERSION_MAJOR}.${TORCH_VERSION_MINOR}" VERSION_GREATER_EQUAL 1.8)
#  message(SEND_ERROR "require PyTorch version < 1.8")
endif()

if (TORCH_ENABLE_CUDA)
  if(POLICY CMP0146)
    cmake_policy(SET CMP0146 OLD)
  endif()

  set(_CUDA_HINT_ROOTS)
  if(CUDA_TOOLKIT_ROOT_DIR)
    list(APPEND _CUDA_HINT_ROOTS "${CUDA_TOOLKIT_ROOT_DIR}")
  endif()
  foreach(_CUDA_ENV_VAR CUDA_HOME CUDA_PATH CUDA_ROOT)
    if(DEFINED ENV{${_CUDA_ENV_VAR}} AND EXISTS "$ENV{${_CUDA_ENV_VAR}}")
      list(APPEND _CUDA_HINT_ROOTS "$ENV{${_CUDA_ENV_VAR}}")
    endif()
  endforeach()
  if(DEFINED ENV{CONDA_PREFIX} AND EXISTS "$ENV{CONDA_PREFIX}")
    list(APPEND _CUDA_HINT_ROOTS "$ENV{CONDA_PREFIX}")
  endif()
  if(DEFINED ENV{MAMBA_ROOT_PREFIX} AND DEFINED ENV{CONDA_DEFAULT_ENV}
      AND EXISTS "$ENV{MAMBA_ROOT_PREFIX}/envs/$ENV{CONDA_DEFAULT_ENV}")
    list(APPEND _CUDA_HINT_ROOTS "$ENV{MAMBA_ROOT_PREFIX}/envs/$ENV{CONDA_DEFAULT_ENV}")
  endif()
  if(Python_EXECUTABLE)
    get_filename_component(_PYTHON_BIN_DIR "${Python_EXECUTABLE}" DIRECTORY)
    get_filename_component(_PYTHON_PREFIX "${_PYTHON_BIN_DIR}" DIRECTORY)
    if(EXISTS "${_PYTHON_PREFIX}")
      list(APPEND _CUDA_HINT_ROOTS "${_PYTHON_PREFIX}")
    endif()
  endif()
  list(APPEND _CUDA_HINT_ROOTS /usr/local/cuda)
  if(_CUDA_HINT_ROOTS)
    list(REMOVE_DUPLICATES _CUDA_HINT_ROOTS)
  endif()

  if(NOT CUDA_NVCC_EXECUTABLE)
    find_program(_CUDA_NVCC_EXECUTABLE nvcc
      HINTS ${_CUDA_HINT_ROOTS}
      PATH_SUFFIXES bin)
    if(_CUDA_NVCC_EXECUTABLE)
      set(CUDA_NVCC_EXECUTABLE "${_CUDA_NVCC_EXECUTABLE}" CACHE FILEPATH "Path to nvcc" FORCE)
    endif()
  endif()
  if(CUDA_NVCC_EXECUTABLE AND NOT CUDA_TOOLKIT_ROOT_DIR)
    get_filename_component(_CUDA_NVCC_BIN_DIR "${CUDA_NVCC_EXECUTABLE}" DIRECTORY)
    get_filename_component(_CUDA_TOOLKIT_ROOT_DIR "${_CUDA_NVCC_BIN_DIR}" DIRECTORY)
    set(CUDA_TOOLKIT_ROOT_DIR "${_CUDA_TOOLKIT_ROOT_DIR}" CACHE PATH "CUDA Toolkit root" FORCE)
  endif()

  find_package(CUDA 9.0)
  if (NOT CUDA_FOUND)
    message(WARNING
      "PyTorch was built with CUDA ${TORCH_CUDA_VERSION}, but CMake could not "
      "find a CUDA toolkit. Set CUDA_TOOLKIT_ROOT_DIR or CUDA_NVCC_EXECUTABLE.")
    set(TORCH_ENABLE_CUDA 0 CACHE BOOL "Whether enable CUDA" FORCE)
  endif(NOT CUDA_FOUND)
endif()
message(STATUS TORCH_ENABLE_CUDA=${TORCH_ENABLE_CUDA})

find_library(TORCH_PYTHON_LIBRARY torch_python PATHS "${TORCH_INSTALL_PREFIX}/lib" REQUIRED)
find_library(TORCH_LIBRARY torch PATHS "${TORCH_INSTALL_PREFIX}/lib" REQUIRED)
find_library(C10_LIBRARY c10 PATHS "${TORCH_INSTALL_PREFIX}/lib" REQUIRED)
find_library(C10_CUDA_LIBRARY c10_cuda PATHS "${TORCH_INSTALL_PREFIX}/lib")
find_library(TORCH_CPU_LIBRARY torch_cpu PATHS "${TORCH_INSTALL_PREFIX}/lib" REQUIRED)
find_library(TORCH_CUDA_LIBRARY torch_cuda PATHS "${TORCH_INSTALL_PREFIX}/lib")

if (EXISTS ${TORCH_INSTALL_PREFIX}/include)
  # torch version 1.4+
  set(TORCH_HEADER_PREFIX ${TORCH_INSTALL_PREFIX}/include)
elseif (EXISTS ${TORCH_INSTALL_PREFIX}/lib/include)
  # torch version 1.0
  set(TORCH_HEADER_PREFIX ${TORCH_INSTALL_PREFIX}/lib/include)
endif()
set(TORCH_INCLUDE_DIRS
  ${TORCH_HEADER_PREFIX}
  ${TORCH_HEADER_PREFIX}/torch/csrc/api/include)

set(LINK_LIBS ${C10_LIBRARY} ${TORCH_CPU_LIBRARY})
if (TORCH_ENABLE_CUDA)
  set(LINK_LIBS ${LINK_LIBS}
    ${C10_CUDA_LIBRARY}
    ${TORCH_CUDA_LIBRARY})
endif()

if(NOT D2D_TORCH_EXTENSION_DEFER_TARGETS)
  add_library(torch STATIC IMPORTED)
  set_target_properties(torch PROPERTIES
    IMPORTED_LOCATION "${TORCH_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${TORCH_INCLUDE_DIRS}"
    INTERFACE_LINK_LIBRARIES "${LINK_LIBS}"
    INTERFACE_COMPILE_OPTIONS "-D_GLIBCXX_USE_CXX11_ABI=${CMAKE_CXX_ABI}"
    )
endif()

# CXX only 
function(add_torch_extension target_name)
  set(multiValueArgs EXTRA_INCLUDE_DIRS EXTRA_LINK_LIBRARIES EXTRA_DEFINITIONS)
  cmake_parse_arguments(ARG "" "" "${multiValueArgs}" ${ARGN})
  if (TORCH_ENABLE_CUDA)
    cuda_add_library(${target_name} STATIC ${ARG_UNPARSED_ARGUMENTS})
  else()
    # remove cuda files 
    list(FILTER ARG_UNPARSED_ARGUMENTS EXCLUDE REGEX ".*cu$")
    list(FILTER ARG_UNPARSED_ARGUMENTS EXCLUDE REGEX ".*cuh$")
    add_library(${target_name} STATIC ${ARG_UNPARSED_ARGUMENTS})
  endif()
  target_include_directories(${target_name} PRIVATE ${ARG_EXTRA_INCLUDE_DIRS} ${TORCH_INCLUDE_DIRS})
  if(D2D_TORCH_EXTENSION_DEFER_TARGETS)
    target_link_directories(${target_name} PRIVATE "${TORCH_INSTALL_PREFIX}/lib")
    target_link_libraries(${target_name} "-L${TORCH_INSTALL_PREFIX}/lib" ${ARG_EXTRA_LINK_LIBRARIES} ${TORCH_LIBRARY} ${LINK_LIBS} pybind11::module)
  else()
    target_link_libraries(${target_name} ${ARG_EXTRA_LINK_LIBRARIES} torch pybind11::module)
  endif()
  target_compile_definitions(${target_name} PRIVATE 
    TORCH_EXTENSION_NAME=${target_name}
    TORCH_VERSION_MAJOR=${TORCH_VERSION_MAJOR}
    TORCH_VERSION_MINOR=${TORCH_VERSION_MINOR}
    ENABLE_CUDA=${TORCH_ENABLE_CUDA}
    ${ARG_EXTRA_DEFINITIONS})
  set_target_properties(${target_name} PROPERTIES 
    POSITION_INDEPENDENT_CODE ON
    CXX_VISIBILITY_PRESET "hidden"
    CUDA_VISIBILITY_PRESET "hidden"
    )
endfunction()

function(add_pytorch_extension target_name)
  set(multiValueArgs EXTRA_INCLUDE_DIRS EXTRA_LINK_LIBRARIES EXTRA_DEFINITIONS)
  cmake_parse_arguments(ARG "" "" "${multiValueArgs}" ${ARGN})
  if (TORCH_ENABLE_CUDA)
    set(CUDA_SRCS "${ARG_UNPARSED_ARGUMENTS}")
    list(FILTER CUDA_SRCS INCLUDE REGEX ".*cu$")
    if (CUDA_SRCS)
      cuda_add_library(${target_name}_cuda_tmp STATIC ${CUDA_SRCS})
      target_include_directories(${target_name}_cuda_tmp PRIVATE ${ARG_EXTRA_INCLUDE_DIRS})
      target_link_libraries(${target_name}_cuda_tmp ${ARG_EXTRA_LINK_LIBRARIES})
      target_compile_definitions(${target_name}_cuda_tmp PRIVATE 
        TORCH_EXTENSION_NAME=${target_name}
        TORCH_MAJOR_VERSION=${TORCH_MAJOR_VERSION}
        TORCH_MINOR_VERSION=${TORCH_MINOR_VERSION}
        ENABLE_CUDA=${TORCH_ENABLE_CUDA}
        ${ARG_EXTRA_DEFINITIONS})
      set_target_properties(${target_name}_cuda_tmp PROPERTIES 
        POSITION_INDEPENDENT_CODE ON
        CXX_VISIBILITY_PRESET "hidden"
        CUDA_VISIBILITY_PRESET "hidden"
        )
    endif()
  endif()
  list(FILTER ARG_UNPARSED_ARGUMENTS EXCLUDE REGEX ".*cu$")
  pybind11_add_module(${target_name} MODULE ${ARG_UNPARSED_ARGUMENTS})
  target_include_directories(${target_name} PRIVATE ${ARG_EXTRA_INCLUDE_DIRS} ${TORCH_INCLUDE_DIRS})
  if(D2D_TORCH_EXTENSION_DEFER_TARGETS)
    target_link_directories(${target_name} PRIVATE "${TORCH_INSTALL_PREFIX}/lib")
  endif()
  if (TORCH_ENABLE_CUDA AND CUDA_SRCS)
    if(D2D_TORCH_EXTENSION_DEFER_TARGETS)
      target_link_libraries(${target_name} PRIVATE ${target_name}_cuda_tmp "-L${TORCH_INSTALL_PREFIX}/lib" ${ARG_EXTRA_LINK_LIBRARIES} ${TORCH_LIBRARY} ${TORCH_PYTHON_LIBRARY} ${LINK_LIBS})
    else()
      target_link_libraries(${target_name} PRIVATE ${target_name}_cuda_tmp ${ARG_EXTRA_LINK_LIBRARIES} torch ${TORCH_PYTHON_LIBRARY})
    endif()
  else()
    if(D2D_TORCH_EXTENSION_DEFER_TARGETS)
      target_link_libraries(${target_name} PRIVATE "-L${TORCH_INSTALL_PREFIX}/lib" ${ARG_EXTRA_LINK_LIBRARIES} ${TORCH_LIBRARY} ${TORCH_PYTHON_LIBRARY} ${LINK_LIBS})
    else()
      target_link_libraries(${target_name} PRIVATE ${ARG_EXTRA_LINK_LIBRARIES} torch ${TORCH_PYTHON_LIBRARY})
    endif()
  endif()
  target_compile_definitions(${target_name} PRIVATE 
    TORCH_EXTENSION_NAME=${target_name}
    TORCH_VERSION_MAJOR=${TORCH_VERSION_MAJOR}
    TORCH_VERSION_MINOR=${TORCH_VERSION_MINOR}
    ENABLE_CUDA=${TORCH_ENABLE_CUDA}
    ${ARG_EXTRA_DEFINITIONS})
endfunction()
