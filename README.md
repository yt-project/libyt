# libyt
[![build-test](https://github.com/cindytsai/libyt/actions/workflows/build-test.yml/badge.svg?branch=master)](https://github.com/cindytsai/libyt/actions/workflows/build-test.yml)

`libyt` is a C++ library for Python package [`yt`](https://yt-project.org/).  It aims to let simulation code uses [`yt`](https://yt-project.org/) to do inline-analysis, while the code is still running. In this way, we can skip the step of writing data to local disk first before doing any analysis. This greatly reduce the disk usage, and increase the temporal resolution.

- **Implement `libyt` into your code** :arrow_right:
  - [Installation](#installation)
  - [User Guide](#user-guide)
  - [Example](#example)
- **See `libyt` working progress** :arrow_right: [Project Board](https://github.com/calab-ntu/libyt/projects/1)
- **See how `libyt` is developed** :arrow_right: [`libyt` Milestone](https://hackmd.io/@Viukb0eMS-aeoZQudVyJ2w/ryCYwu0xF)
- **Supported `yt` functionalities**:

  |       `yt` Function      | Supported | Notes                                    |
  |:---------:|:---------:|------------------------------------------|
  | `find_max`               |     V     |                                          |
  | `ProjectionPlot`         |     V     |                                          |
  | `OffAxisProjectionPlot`  |     V     |                                          |
  | `SlicePlot`              |     V     |                                          |
  | `OffAxisSlicePlot`       |     V     |                                          |
  | `covering_grid`          |     V     |                                          |
  | 1D `create_profile`      |     V     |                                          |
  | 2D `create_profile`      |     V     |                                          |
  | `ProfilePlot`            |     V     |                                          |
  | `PhasePlot`              |     V     |                                          |
  | `LinePlot`               |     V     |                                          |
  | Halo Analysis            |           | Not test yet.                            |
  | Isocontours              |     V     |  |
  | `volume_render`          |     V     | Only when MPI size is even will it work. |
  | `ParticlePlot`           |     V     |                                          |
  | `ParticleProjectionPlot` |     V     |                                          |

## Installation
### libyt
#### Set Path
In `libyt/src/Makefile`, update `PYTHON_PATH`, `PYTHON_VERSION`, `NUMPY_PATH` and `MPI_PATH`:
```makefile
# Your paths
############################################################
PYTHON_PATH    := $(YOUR_PYTHON_PATH)
PYTHON_VERSION := $(YOUR_PYTHON_VERSION)
NUMPY_PATH     := $(YOUR_NUMPY_PATH)
MPI_PATH       := $(YOUR_MPI_PATH)
```
> :warning: Make sure you are using the same MPI to compile `libyt` and your simulation code.

#### Options
- **Normal Mode**: Normal mode will shut down and terminate all the processes including simulation if there are errors during in situ analysis using Python.

- **Interactive Mode**: Interactive mode will not terminate the processes if there are errors during in situ analysis using Python. Interactive mode is more like an add-ons for normal mode. 
  - To use interactive mode, we need `readline` library and switch `-DINTERACTIVE_MODE` to on in `Makefile`. Please set the path, if `readline` library is not on your system include search path.

  ```makefile
  READLINE_PATH  := $(YOUR_READLINE_PATH)

  # Interactive Mode: supports reloading inline script, active python prompt and does not halt when
  # error occurs. Require readline library, add READLINE_PATH if it is not inside include search path.
  OPTIONS += -DINTERACTIVE_MODE
  ```

#### Compile, Link, and Headers
Compile and move `libyt.so.*` to `libyt/lib` folder:
```bash
make clean
make
cp libyt.so* ../lib/
```

Include `libyt.h` header which is in `libyt/include` and library in your simulation code. We should also include `libyt_interactive_mode.h` in interactive mode.

### yt_libyt
Even though `libyt` can call arbitrary Python module, `yt` is the core analytic tool used in `libyt`. This is singled out as an individual frontend for `yt`, and this will work with any version of [`yt`](https://yt-project.org/) with Python version > 3.6.


```bash
pip install mpi4py
```
> :warning: Please make sure `mpi4py` you used in Python and in simulation are matched. You can also check how to install `mpi4py` [here](https://mpi4py.readthedocs.io/en/stable/install.html#installation).

Clone [`data-exp-lab/yt_libyt`](https://github.com/data-exp-lab/yt_libyt.git):
```bash
git clone https://github.com/data-exp-lab/yt_libyt.git
cd yt_libyt
pip install .
```

## User Guide
This guide will walk you through how to implement `libyt` into your code. And how you can convert your everyday used `yt` script to do inline-analysis. All the user guide are in [`doc`](./doc) folder.


- Implement `libyt` to your code step by step
  - [Initialize - `yt_initialize`](./doc/Initialize.md#initialize)
  - [Set `yt` Parameter - `yt_set_Parameters`](./doc/SetYTParameter.md#set-yt-parameter)
  - [Set Code Specific Parameter - `yt_set_UserParameter*`](./doc/SetCodeSpecificParameter.md#set-code-or-user-specific-parameter)
  - [Set Fields Information - `yt_get_FieldsPtr`](./doc/SetFieldsInformation.md#set-fields-information)
  - [Set Particles Information - `yt_get_ParticlesPtr`](./doc/SetParticlesInformation.md#set-particles-information)
  - [Set Local Grids Information - `yt_get_GridsPtr`](./doc/SetLocalGridsInformation.md#set-local-grids-information)
  - [Commit Your Settings - `yt_commit`](./doc/CommitYourSettings.md#commit-your-settings)
  - [Perform Inline-Analysis - `yt_run_Function` and `yt_run_FunctionArguments`](./doc/PerformInlineAnalysis.md#perform-inline-analysis)
  - [Activate Interactive Mode - `yt_run_InteractiveMode`](./doc/ActivateInteractiveMode.md#activate-interactive-mode) (Only available in interactive mode)
  - [Free Resource - `yt_free`](./doc/FreeResource.md#free-resource)
  - [Finalize - `yt_finalize`](./doc/Finalize.md#finalize)
- [Inline Python Script](./doc/InlinePythonScript.md#inline-python-script)

## Example
- A simple demo in [`libyt/example.cpp`](./example/example.cpp)
- [`gamer`](https://github.com/gamer-project/gamer/tree/master/src/YT)
