ENV_PREFIX ?=
DEFAULT_CMAKE := $(shell if [ -n "$(ENV_PREFIX)" ] && [ -x "$(ENV_PREFIX)/bin/cmake3" ]; then printf '%s\n' "$(ENV_PREFIX)/bin/cmake3"; elif [ -n "$(ENV_PREFIX)" ] && [ -x "$(ENV_PREFIX)/bin/cmake" ]; then printf '%s\n' "$(ENV_PREFIX)/bin/cmake"; elif command -v cmake3 >/dev/null 2>&1; then command -v cmake3; elif command -v cmake >/dev/null 2>&1; then command -v cmake; fi)
DEFAULT_PYTHON := $(shell if [ -n "$(ENV_PREFIX)" ] && [ -x "$(ENV_PREFIX)/bin/python" ]; then printf '%s\n' "$(ENV_PREFIX)/bin/python"; elif [ -n "$(ENV_PREFIX)" ] && [ -x "$(ENV_PREFIX)/bin/python3" ]; then printf '%s\n' "$(ENV_PREFIX)/bin/python3"; elif command -v python3 >/dev/null 2>&1; then command -v python3; else command -v python 2>/dev/null; fi)
DEFAULT_C_COMPILER := $(shell if [ -n "$(ENV_PREFIX)" ] && [ -x "$(ENV_PREFIX)/bin/cc" ]; then printf '%s\n' "$(ENV_PREFIX)/bin/cc"; fi)
DEFAULT_CXX_COMPILER := $(shell if [ -n "$(ENV_PREFIX)" ] && [ -x "$(ENV_PREFIX)/bin/c++" ]; then printf '%s\n' "$(ENV_PREFIX)/bin/c++"; fi)
DEFAULT_CUDA_TOOLKIT_ROOT := $(shell if [ -n "$$CUDA_HOME" ]; then printf '%s\n' "$$CUDA_HOME"; elif [ -n "$$CUDA_PATH" ]; then printf '%s\n' "$$CUDA_PATH"; fi)
DEFAULT_BOOST_INCLUDE_DIR := $(shell if [ -n "$(ENV_PREFIX)" ] && [ -f "$(ENV_PREFIX)/include/boost/version.hpp" ]; then printf '%s\n' "$(ENV_PREFIX)/include"; fi)
DEFAULT_BOOST_LIBRARY_DIR := $(shell if [ -n "$(ENV_PREFIX)" ] && { [ -e "$(ENV_PREFIX)/lib/libboost_graph.so" ] || [ -e "$(ENV_PREFIX)/lib/libboost_graph.a" ]; }; then printf '%s\n' "$(ENV_PREFIX)/lib"; fi)
DEFAULT_BISON_EXECUTABLE := $(shell if [ -n "$(ENV_PREFIX)" ] && [ -x "$(ENV_PREFIX)/bin/bison" ]; then printf '%s\n' "$(ENV_PREFIX)/bin/bison"; fi)
DEFAULT_FLEX_EXECUTABLE := $(shell if [ -n "$(ENV_PREFIX)" ] && [ -x "$(ENV_PREFIX)/bin/flex" ]; then printf '%s\n' "$(ENV_PREFIX)/bin/flex"; fi)
DEFAULT_ZLIB_INCLUDE_DIR := $(shell if [ -n "$(ENV_PREFIX)" ] && [ -f "$(ENV_PREFIX)/include/zlib.h" ]; then printf '%s\n' "$(ENV_PREFIX)/include"; fi)
DEFAULT_ZLIB_LIBRARY := $(shell if [ -n "$(ENV_PREFIX)" ] && [ -e "$(ENV_PREFIX)/lib/libz.so" ]; then printf '%s\n' "$(ENV_PREFIX)/lib/libz.so"; fi)

CMAKE ?= $(DEFAULT_CMAKE)
BUILD_DIR ?= $(CURDIR)/build
INSTALL_PREFIX ?= $(CURDIR)/install
CMAKE_BUILD_TYPE ?= Release
PYTHON_EXECUTABLE ?= $(DEFAULT_PYTHON)
C_COMPILER ?= $(DEFAULT_C_COMPILER)
CXX_COMPILER ?= $(DEFAULT_CXX_COMPILER)
CUDA_TOOLKIT_ROOT_DIR ?= $(DEFAULT_CUDA_TOOLKIT_ROOT)
BOOST_INCLUDE_DIR ?= $(DEFAULT_BOOST_INCLUDE_DIR)
BOOST_LIBRARY_DIR ?= $(DEFAULT_BOOST_LIBRARY_DIR)
BOOST_GRAPH_LIBRARY ?= $(if $(BOOST_LIBRARY_DIR),$(BOOST_LIBRARY_DIR)/libboost_graph.so)
BOOST_REGEX_LIBRARY ?= $(if $(BOOST_LIBRARY_DIR),$(BOOST_LIBRARY_DIR)/libboost_regex.so)
BISON_EXECUTABLE ?= $(DEFAULT_BISON_EXECUTABLE)
FLEX_EXECUTABLE ?= $(DEFAULT_FLEX_EXECUTABLE)
FLEX_INCLUDE_DIR ?= $(if $(DEFAULT_FLEX_EXECUTABLE),$(ENV_PREFIX)/include)
ZLIB_INCLUDE_DIR ?= $(DEFAULT_ZLIB_INCLUDE_DIR)
ZLIB_LIBRARY ?= $(DEFAULT_ZLIB_LIBRARY)
CMAKE_ARGS ?=
BUILD_ENV ?= env -u LD_LIBRARY_PATH

CMAKE_CONFIGURE_ARGS := -S $(CURDIR) -B $(BUILD_DIR)
CMAKE_CONFIGURE_ARGS += -DCMAKE_INSTALL_PREFIX=$(INSTALL_PREFIX)
CMAKE_CONFIGURE_ARGS += -DCMAKE_BUILD_TYPE=$(CMAKE_BUILD_TYPE)
CMAKE_CONFIGURE_ARGS += -DCMAKE_POLICY_VERSION_MINIMUM=3.5

ifneq ($(strip $(PYTHON_EXECUTABLE)),)
CMAKE_CONFIGURE_ARGS += -DPython_EXECUTABLE=$(PYTHON_EXECUTABLE)
endif
ifneq ($(strip $(C_COMPILER)),)
CMAKE_CONFIGURE_ARGS += -DCMAKE_C_COMPILER=$(C_COMPILER)
endif
ifneq ($(strip $(CXX_COMPILER)),)
CMAKE_CONFIGURE_ARGS += -DCMAKE_CXX_COMPILER=$(CXX_COMPILER)
endif
ifneq ($(strip $(CUDA_TOOLKIT_ROOT_DIR)),)
CMAKE_CONFIGURE_ARGS += -DCUDA_TOOLKIT_ROOT_DIR=$(CUDA_TOOLKIT_ROOT_DIR)
endif
ifneq ($(strip $(BOOST_INCLUDE_DIR)),)
CMAKE_CONFIGURE_ARGS += -DBoost_INCLUDE_DIR=$(BOOST_INCLUDE_DIR)
endif
ifneq ($(strip $(BOOST_LIBRARY_DIR)),)
CMAKE_CONFIGURE_ARGS += -DBoost_LIBRARY_DIR_DEBUG=$(BOOST_LIBRARY_DIR)
CMAKE_CONFIGURE_ARGS += -DBoost_LIBRARY_DIR_RELEASE=$(BOOST_LIBRARY_DIR)
endif
ifneq ($(strip $(BOOST_GRAPH_LIBRARY)),)
CMAKE_CONFIGURE_ARGS += -DBoost_GRAPH_LIBRARY_DEBUG=$(BOOST_GRAPH_LIBRARY)
CMAKE_CONFIGURE_ARGS += -DBoost_GRAPH_LIBRARY_RELEASE=$(BOOST_GRAPH_LIBRARY)
endif
ifneq ($(strip $(BOOST_REGEX_LIBRARY)),)
CMAKE_CONFIGURE_ARGS += -DBoost_REGEX_LIBRARY_DEBUG=$(BOOST_REGEX_LIBRARY)
CMAKE_CONFIGURE_ARGS += -DBoost_REGEX_LIBRARY_RELEASE=$(BOOST_REGEX_LIBRARY)
endif
ifneq ($(strip $(BISON_EXECUTABLE)),)
CMAKE_CONFIGURE_ARGS += -DBISON_EXECUTABLE=$(BISON_EXECUTABLE)
endif
ifneq ($(strip $(FLEX_EXECUTABLE)),)
CMAKE_CONFIGURE_ARGS += -DFLEX_EXECUTABLE=$(FLEX_EXECUTABLE)
endif
ifneq ($(strip $(FLEX_INCLUDE_DIR)),)
CMAKE_CONFIGURE_ARGS += -DFLEX_INCLUDE_DIR=$(FLEX_INCLUDE_DIR)
endif
ifneq ($(strip $(ZLIB_INCLUDE_DIR)),)
CMAKE_CONFIGURE_ARGS += -DZLIB_INCLUDE_DIR=$(ZLIB_INCLUDE_DIR)
endif
ifneq ($(strip $(ZLIB_LIBRARY)),)
CMAKE_CONFIGURE_ARGS += -DZLIB_LIBRARY=$(ZLIB_LIBRARY)
CMAKE_CONFIGURE_ARGS += -DZLIB_LIBRARY_RELEASE=$(ZLIB_LIBRARY)
CMAKE_CONFIGURE_ARGS += -DCOIN_ZLIB_LIBRARY=$(ZLIB_LIBRARY)
endif

.PHONY: all configure install clean distclean

all: configure
	$(BUILD_ENV) $(MAKE) -C $(BUILD_DIR)

configure:
	@if [ -z "$(strip $(CMAKE))" ]; then \
		echo "Error: CMake was not found. Install CMake 3.x, add it to PATH, pass CMAKE=/path/to/cmake3, or pass ENV_PREFIX=/path/to/env."; \
		exit 127; \
	fi
	$(BUILD_ENV) $(CMAKE) $(CMAKE_CONFIGURE_ARGS) $(CMAKE_ARGS)

install: all
	$(BUILD_ENV) $(MAKE) -C $(BUILD_DIR) install

clean:
	@if [ -d "$(BUILD_DIR)" ]; then $(BUILD_ENV) $(MAKE) -C "$(BUILD_DIR)" clean; fi

distclean:
	$(BUILD_ENV) $(CMAKE) -E rm -rf "$(BUILD_DIR)" "$(INSTALL_PREFIX)"
