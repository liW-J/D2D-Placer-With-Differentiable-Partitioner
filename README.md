# 3D(D2D) Placer

# Dependency

- [Python](https://www.python.org/) 3.5/3.6/3.7/3.8/3.9

- [Pytorch](https://pytorch.org/) 1.6/1.7/1.8/2.0

  - Other versions may also work, but not tested

- [GCC](https://gcc.gnu.org/)

  - Recommend GCC 7.5 (with `c++17` support).
  - Do not recommend GCC 9 or later due to backward compatibility issues.
  - Other compilers may also work, but not tested.

- [Boost](https://www.boost.org) >= 1.55.0
  - Need to install and visible for linking
- [Bison](https://www.gnu.org/software/bison) >= 3.3

  - Need to install

(Optional)

  - If installed and found, GPU acceleration will be enabled.
  - Otherwise, only CPU implementation is enabled.

- GPU architecture compatibility 6.0 or later (Optional)
  - Code has been tested on GPUs with compute compatibility 6.0, 7.0, and 7.5.
  - Please check the [compatibility](https://developer.nvidia.com/cuda-gpus) of the GPU devices.
  - The default compilation target is compatibility 6.0. This is the minimum requirement and lower compatibility is not supported for the GPU feature.
  - For compatibility 7.0, it is necessary to set the CMAKE_CUDA_FLAGS to -gencode=arch=compute_70,code=sm_70.
- [Cairo](https://github.com/freedesktop/cairo) (Optional)

  - If installed and found, the plotting functions will be faster by using C/C++ implementation.
  - Otherwise, python implementation is used.

- [NTUPlace3](http://eda.ee.ntu.edu.tw/research.htm) (Optional)
  - If the binary is provided, it can be used to perform detailed placement.

To pull git submodules in the root directory

```
git submodule init
git submodule update
```

Or alternatively, pull all the submodules when cloning the repository.

```
git clone --recursive https://github.com/limbo018/DREAMPlace.git
```

# How to Install Python Dependency

Go to the root directory.

```
pip install -r requirements.txt
```

# How to Build

Two options are provided for building: with and without [Docker](https://hub.docker.com).

## Build with Docker

You can use the Docker container to avoid building all the dependencies yourself.

1. Install Docker on [Windows](https://docs.docker.com/docker-for-windows/), [Mac](https://docs.docker.com/docker-for-mac/) or [Linux](https://docs.docker.com/install/).
2. To enable the GPU features, install [NVIDIA-docker](https://github.com/NVIDIA/nvidia-docker); otherwise, skip this step.
3. Navigate to the repository.
4. Get the docker container with either of the following options.
   - Option 1: pull from the cloud [limbo018/dreamplace](https://hub.docker.com/r/limbo018/dreamplace).
   ```
   docker pull limbo018/dreamplace:cuda
   ```
   - Option 2: build the container.
   ```
   docker build . --file Dockerfile --tag your_name/dreamplace:cuda
   ```
5. Enter bash environment of the container. Replace `limbo018` with your name if option 2 is chosen in the previous step.

Run with GPU on Linux.

```
docker run --gpus 1 -it -v $(pwd):/DREAMPlace limbo018/dreamplace:cuda bash
```

Run with GPU on Windows.

```
docker run --gpus 1 -it -v /dreamplace limbo018/dreamplace:cuda bash
```

Run without GPU on Linux.

```
docker run -it -v $(pwd):/DREAMPlace limbo018/dreamplace:cuda bash
```

Run without GPU on Windows.

```
docker run -it -v /dreamplace limbo018/dreamplace:cuda bash
```

6. `cd /DREAMPlace`.
7. Go to [next section](#build-without-docker) to complete building within the container.

## Build without Docker

[CMake](https://cmake.org) is adopted as the makefile system.
To build, go to the root directory.

```
mkdir build
cd build # we call this <build directory>
cmake .. -DCMAKE_INSTALL_PREFIX=<installation directory> -DPython_EXECUTABLE=$(which python)
make
make install
```
Where `<build directory>` is the directory where you compile the code, and `<installation directory>` is the directory where you want to install DREAMPlace (e.g., `../install`).
Third party submodules are automatically built except for [Boost](https://www.boost.org).

To clean, go to the root directory.

```
rm -r build
```
`<build directory>` can be removed after installation if you do not need incremental compilation later. 

Here are the available options for CMake.

- CMAKE_INSTALL_PREFIX: installation directory
  - Example `cmake -DCMAKE_INSTALL_PREFIX=path/to/your/directory`
- CMAKE_CUDA_FLAGS: custom string for NVCC (default -gencode=arch=compute_60,code=sm_60)
  - Example `cmake -DCMAKE_CUDA_FLAGS=-gencode=arch=compute_60,code=sm_60`
- CMAKE_CXX_ABI: 0|1 for the value of \_GLIBCXX_USE_CXX11_ABI for C++ compiler, default is 0.
  - Example `cmake -DCMAKE_CXX_ABI=0`
  - It must be consistent with the \_GLIBCXX_USE_CXX11_ABI for compling all the C++ dependencies, such as Boost and PyTorch.
  - PyTorch in default is compiled with \_GLIBCXX_USE_CXX11_ABI=0, but in a customized PyTorch environment, it might be compiled with \_GLIBCXX_USE_CXX11_ABI=1.

# How to Get Benchmarks

To get ISPD 2005 and 2015 benchmarks, run the following script from the directory.

```
python benchmarks/ispd2005_2015.py
```

# How to Run

Before running, make sure the benchmarks have been downloaded and the python dependency packages have been installed.
Go to the **install directory** and run with JSON configuration file for full placement.

```
cd <installation directory>
python dreamplace/Placer.py test/ispd2005/adaptec1.json
```

Test individual `pytorch` op with the unit tests in the root directory.

```
cd <installation directory>
python unittest/ops/hpwl_unittest.py
```

# Configurations

Descriptions of options in JSON configuration file can be found by running the following command.

```
cd <installation directory>
python dreamplace/Placer.py --help
```

The list of options as follows will be shown.

| JSON Parameter | Default                | Description                                                                               |
| -------------- | ---------------------- | ----------------------------------------------------------------------------------------- |
| aux_input      | required for Bookshelf | input .aux file                                                                           |
| lef_input      | required for LEF/DEF   | input LEF file                                                                            |
| def_input      | required for LEF/DEF   | input DEF file                                                                            |
| verilog_input  | optional for LEF/DEF   | input VERILOG file, provide circuit netlist information if it is not included in DEF file |
| gpu            | 1                      | enable gpu or not                                                                         |
