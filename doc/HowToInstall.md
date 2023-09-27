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
### Options
Comment or uncomment options to switch on or off. The options are independent and do not affect each other.
```makefile
OPTIONS += -DSERIAL_MODE
OPTIONS += -DINTERACTIVE_MODE
OPTIONS += -DSUPPORT_TIMER
```
- **`OPTIONS += -DSERIAL_MODE` (OFF/ON)**
  - **Parallel Mode (OFF)**: compile `libyt` using MPI and run `libyt` in parallel. 
  - **Serial Mode (ON)**: compile `libyt` using GCC and run `libyt` in serial.
- **`OPTIONS += -DINTERACTIVE_MODE` (OFF/ON)**
  - **Normal Mode (OFF)**: shut down and terminate all the processes including simulation, if there are errors during in situ analysis using Python.
  - **Interactive Mode (ON)**: will not terminate the processes if errors occur while using Python for in situ analysis. It supports interactive Python prompt. This is like normal Python prompt with access to simulation data.
- **`OPTIONS += -DSUPPORT_TIMER` (OFF/ON)**
  - **(ON)**: support time profiling. (See [Time Profiling]({% link DebugAndTimeProfiling/TimeProfiling.md %}#time-profiling))

### Set Dependency Paths
In `libyt/src/Makefile`, update `PYTHON_PATH`, `PYTHON_VERSION`, `NUMPY_PATH` and `MPI_PATH`:
```makefile
# Dependency Paths: other required dependencies
#######################################################################################################
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

### Set Installation Prefix and Install
If you wish to install `libyt` to other location besides this repository, set `INSTALL_PREFIX` in `libyt/src/Makefile`:
```makefile
# Installation Paths: if not set, it will install to current directory
#######################################################################################################
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
- `Python` >= 3.6
- `yt`
- `mpi4py`

> :warning: Please make sure `mpi4py` used in Python and MPI used in simulation are matched. Check how to install `mpi4py` [here](https://mpi4py.readthedocs.io/en/stable/install.html#installation).

### Install
- `yt_libyt`

```bash
git clone https://github.com/data-exp-lab/yt_libyt.git
cd yt_libyt
pip install .
```
