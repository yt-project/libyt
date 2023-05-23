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

## libyt
### Options
- **Normal Mode**
  - Normal mode will shut down and terminate all the processes including simulation, if there are errors during in situ analysis using Python.

- **Interactive Mode**: Uncomment `OPTIONS += -DINTERACTIVE_MODE` in `libyt/src/Makefile` to use this mode.
  - Interactive mode will not terminate the processes if errors occur while using Python for in situ analysis.
  - It supports interactive Python prompt. This is like normal Python prompt with access to simulation data.

### Set Paths
In `libyt/src/Makefile`, update `PYTHON_PATH`, `PYTHON_VERSION`, `NUMPY_PATH` and `MPI_PATH`:
```makefile
# Your paths
############################################################
PYTHON_PATH    := $(YOUR_PYTHON_PATH)
PYTHON_VERSION := $(YOUR_PYTHON_VERSION)
NUMPY_PATH     := $(YOUR_NUMPY_PATH)
MPI_PATH       := $(YOUR_MPI_PATH)
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
- **READLINE_PATH**: [GNU `readline` library](https://tiswww.case.edu/php/chet/readline/rltop.html) path, under this folder, there should contain `include`, `lib` etc. This is only needed in [Interactive Mode](#options).


### Install
After updating `Makefile` in `libyt/src`, go to `libyt/src` folder
```bash
make clean
make
make install    # this copies library to ../lib, will add feature of install to other path in the future.
```
We use this cloned `libyt` folder as installation folder.
- `libyt/include`: Contain `libyt.h`. This is the header file for `libyt` API.
- `libyt/lib`: Contain the shared library for simulation to link to.

## yt_libyt
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
