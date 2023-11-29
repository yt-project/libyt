---
layout: default
title: How to Install
nav_order: 2
---
# How to Install
{: .no_toc }
<details open markdown="block">
  <summary>
    Table of contents
  </summary>
  {: .text-delta }
- TOC
{:toc}
</details>
---

## C Library -- libyt

There are two ways to install `libyt`:
- [CMake](#cmake)
- [Make](#make)

> :information_source: `libyt` will rely more on CMake and drop support for Make in the future.

### Options
The options are all independent of each other.

- **`-DSERIAL_MODE` (OFF/ON)**
  - **Parallel Mode (OFF)**: compile `libyt` using MPI and run `libyt` in parallel.
  - **Serial Mode (ON)**: compile `libyt` using GCC and run `libyt` in serial.
- **`-DINTERACTIVE_MODE` (OFF/ON)**
  - **Normal Mode (OFF)**: shut down and terminate all the processes including simulation, if there are errors during in situ analysis using Python.
  - **Interactive Mode (ON)**: will not terminate the processes if errors occur while using Python for in situ analysis. It supports interactive Python prompt. This is like normal Python prompt with access to simulation data.
- **`-DSUPPORT_TIMER` (OFF/ON)**
  - **(ON)**: support time profiling. (See [Time Profiling]({% link DebugAndTimeProfiling/TimeProfiling.md %}#time-profiling))

### CMake
1. Toggle options, set paths and generate files to build the project. This can be done through either (a) or (b):

   a. Set it through editing `CMakeLists.txt` at root:

   ```cmake
   option(SERIAL_MODE       "Compile library for serial process" OFF)
   option(INTERACTIVE_MODE  "Use interactive mode"               OFF)
   option(SUPPORT_TIMER     "Support timer"                      OFF)
   
   set(PYTHON_PATH    "" CACHE PATH "Path to Python installation prefix")
   set(MPI_PATH       "" CACHE PATH "Path to MPI installation prefix")
   set(READLINE_PATH  "" CACHE PATH "Path to Readline installation prefix")
   ```
   b. Set the options and paths through command line when generating files. Possible flags are:
   ```bash
   -DSERIAL_MODE=ON      # default is OFF
   -DINTERACTIVE_MODE=ON # default is OFF
   -DSUPPORT_TIMER=ON    # default is OFF
   -DPYTHON_PATH=<python-install-prefix>
   -DMPI_PATH=<mpi-install-prefix>
   -DREADLINE_PATH=<readline-install-prefix>
   ```
   If you wish to set the GCC compiler in serial mode, export the environment variable `CC` and `CXX` before running `cmake`:
   ```bash
   export CC=/software/gcc/bin/gcc
   export CXX=/software/gcc/bin/g++
   ```
After the above:
```bash
cd libyt # the root of the project
cmake -B <build-dir-name> -S . <options-from-(b)>
```

2. Build the project:
```bash
cmake --build <build-dir-name>
```
3. Install the library:
```bash
cmake --install <build-dir-name> --prefix <install-to-dir> 
```

#### Example
- The following builds `libyt` in serial mode using user designated GCC compiler and then installs the library in `/home/user/softwares/libyt`:
```bash
cd libyt
export CC=/software/gcc/8.4.0/bin/gcc
export CXX=/software/gcc/8.4.0/bin/g++
rm -rf build
cmake -B build -S . -DSERIAL_MODE=ON
cmake --build build
cmake --install build --prefix /home/user/softwares/libyt
```

- The following builds `libyt` in parallel mode using user designated MPI compiler and then installs the library in `/home/user/softwares/libyt`:
```bash
cd libyt
rm -rf build
cmake -B build -S . -DMPI_PATH=/software/openmpi/4.1.1-gnu # can omit -DMPI_PATH if it is set in CMakeLists.txt
cmake --build build
cmake --install build --prefix /home/user/softwares/libyt
```

### Make
1. Comment or uncomment options to switch on or off, the options are independent to each other:
```makefile
OPTIONS += -DSERIAL_MODE
OPTIONS += -DINTERACTIVE_MODE
OPTIONS += -DSUPPORT_TIMER
```

2. Set dependency paths in `libyt/src/Makefile`, update `PYTHON_PATH`, `PYTHON_VERSION`, `NUMPY_PATH` and `MPI_PATH`:
```makefile
# Dependency Paths: other required dependencies
##########################################################################
PYTHON_PATH    := $(YOUR_PYTHON_PATH)        # Must have
PYTHON_VERSION := $(YOUR_PYTHON_VERSION)     # Must have
NUMPY_PATH     := $(YOUR_NUMPY_PATH)         # Must have
MPI_PATH       := $(YOUR_MPI_PATH)
GCC_PATH       := $(YOUR_GCC_PATH)
READLINE_PATH  := $(YOUR_READLINE_PATH)
```
- **PYTHON_PATH**: Python installation prefix, under this folder, there should be folders like `include`, `lib` etc.
- **PYTHON_VERSION**: Python `x.y` version, put `x.y` only.
- **NUMPY_PATH**: Path to where `numpy` is installed. We can use `pip` to look up, and NUMPY_PATH is `<path>/numpy`.
  ```bash
  $ pip list -v | grep numpy
  Package   Version       Location   Installer
  --------- ------------- ---------- -----------
  numpy     <version>     <path>     pip
  ```
- **MPI_PATH**: MPI installation prefix, under this folder, there should be folders like `include`, `lib` etc. 
  > :warning: Make sure you are using the same MPI to compile `libyt` and your simulation code.
- **GCC_PATH**: GCC installation prefix, if it is not on system path. This is only needed when compiling `libyt` in [Serial (`-DSERIAL_MODE`)](#options).
- **READLINE_PATH**: [GNU `readline` library](https://tiswww.case.edu/php/chet/readline/rltop.html) path, under this folder, there should contain `include`, `lib` etc. This is only needed when compiling `libyt` in [Interactive Mode (`-DINTERACTIVE_MODE`)](#options).

3. Set installation prefix if you wish to install `libyt` to other location besides this repository. Set `INSTALL_PREFIX` in `libyt/src/Makefile`:
```makefile
# Installation Paths: if not set, it will install to current directory
##########################################################################
INSTALL_PREFIX := $(YOUR_INSTALL_PREFIX)
```
Go to `libyt/src` folder:
```bash
make clean
make
make install
```
`libyt` is now installed in `$(INSTALL_PREFIX)` if it is set, otherwise, it will install under this repository: 
- `include`: Contain `libyt.h`. This is the header file for `libyt` API.
- `lib`: Contain the shared library for simulation to link to.

## Python Package -- yt_libyt
To use `yt` as the core analytic tool, we need to install `yt_libyt`, a `yt` frontend for `libyt`. 
The frontend will work with any version of [`yt`](https://yt-project.org/) with Python version >= 3.6.

### Requirements
- `Python` >= 3.7
- `yt`
- `mpi4py`

> :warning: Please make sure `mpi4py` used in Python and MPI used in simulation are matched. Check how to install `mpi4py` [here](https://mpi4py.readthedocs.io/en/stable/install.html#installation).

### Install yt_libyt
- Install from source:
```bash
git clone https://github.com/data-exp-lab/yt_libyt.git
cd yt_libyt
pip install .
```

- Install from PyPI:
```bash
pip install yt-libyt
```
