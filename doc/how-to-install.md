# How to Install

## C Library -- libyt

Go through [basic requirements](#basic-requirements) and [options](#options) and set the dependencies paths. Then compile and install `libyt` using CMake.


### Basic Requirements
- CMake (>=3.15)
- GCC compiler (>4.8): Should be able to support `c++14`.
  - `CXX`: Path to `g++` compiler.
  - `CC`: Path to `gcc` compiler.
- Python (>=3.7): 
  - `PYTHON_PATH`: Python installation prefix, the path should contain folders like `include`, `lib` etc. 
  - `NumPy`: Should have `NumPy` installed.

### Options 
The options are mutually independent to each other. 

##### `-DSERIAL_MODE`

- Default: `-DSERIAL_MODE=OFF`

|                     | Notes                      | Required Paths | Required Python Packages                     |
|---------------------|----------------------------|----------------|----------------------------------------------|
| **Parallel Mode** (OFF) | Compile `libyt` using MPI. | - `MPI_PATH`   | - [`mpi4py`](https://mpi4py.readthedocs.io/) |
| **Serial Mode** (ON)    | Compile `libyt` using GCC. |                |                                              |

:::
###### Required Paths
:::
- `MPI_PATH`: MPI installation prefix, the path should contain folders like `include`, `lib` etc.
  > {octicon}`alert;1em;sd-text-danger;` Make sure you are using the same MPI to compile `libyt` and your simulation code.

:::
###### Required Python Packages
:::
- `mpi4py`: This is Python bindings for the Message Passing Interface (MPI) standard.
  > {octicon}`alert;1em;sd-text-danger;` Please make sure `mpi4py` used in Python and MPI used in simulation are matched. Check how to install `mpi4py` [here](https://mpi4py.readthedocs.io/en/stable/install.html#installation).

##### `-DINTERACTIVE_MODE`

- Default: `-DINTERACTIVE_MODE=OFF`

|                       | Notes                                                                                                                                                                                                                                  | Required Paths   |
|-----------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------|
| Normal Mode (OFF)     | Shut down and terminate all the processes including simulation, if error occurs during in situ analysis.                                                                                                                               |                  |
| Interactive Mode (ON) | Will not terminate the processes if error occurs while doing in situ analysis and supports [interactive Python prompt]({% link InSituPythonAnalysis/InteractivePythonPrompt.md %}#interactive-python-prompt) and [reloading script]({% link InSituPythonAnalysis/ReloadingScript.md %}#reloading-script). | - `READLINE_PATH` |

- `READLINE_PATH`: [GNU `readline` library](https://tiswww.case.edu/php/chet/readline/rltop.html) installation prefix, the path should contain folders like `include`, `lib` etc.

##### `-DJUPYTER_KERNEL`

- Default: `-DJUPYTER_KERNEL=OFF`

|  | Notes                                                                                                                                                                                | Required Paths | Required Python Packages                                                                                                                                                                    |
|---|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Jupyter Kernel (ON) | Activate Jupyter kernel and enable JupyterLab UI. (See  [Jupyter Notebook Access]({% link InSituPythonAnalysis/JupyterNotebookAccess/JupyterNotebook.md %}#jupyter-notebook-access)) | - `nlohmann_json_DIR` <br> - `cppzmq_DIR` <br> - `xtl_DIR` <br> - `xeus_DIR` <br> - `xeus-zmq_DIR` <br> - `ZeroMQ_DIR` <br> | - [`jupyter_libyt`](#jupyter_libyt) <br> - [`jupyter-client`](https://jupyter-client.readthedocs.io/en/stable/index.html) <br> - (Optional)[`jedi`](https://jedi.readthedocs.io/en/latest/) |

:::
###### Required Paths
:::
- `nlohmann_json_DIR` (>=3.2.0, <4.0.0): Path to `nlohmann_jsonConfig.cmake` after installing [`nlohmann_json`](https://github.com/nlohmann/json).
- `cppzmq_DIR` (>=4.8.1, <5.0.0): Path to `cppzmqConfig.cmake` after installing [`cppzmq`](https://github.com/zeromq/cppzmq).
- `xtl_DIR` (>=0.7.0, <0.8.0): Path to `xtlConfig.cmake` after installing [`xtl`](https://github.com/xtensor-stack/xtl).
- `xeus_DIR` (>=3.0.0, <4.0.0): Path to `xeusConfig.cmake` after installing [`xeus`](https://github.com/jupyter-xeus/xeus).
- `xeus-zmq_DIR` (1.x release): Path to `xeus-zmqConfig.cmake` after installing [`xeus-zmq`](https://github.com/jupyter-xeus/xeus-zmq).
- `ZeroMQ_DIR` (>=4.2.5, <5.0.0): Path to `ZeroMQConfig.cmake` after installing [`ZeroMQ`](https://github.com/zeromq/libzmq). (Some system may already have ZeroMQ installed, which doesn't need to provide the path explicitly.)

> {octicon}`info;1em;sd-text-info;` `nlohmann_json`, `cppzmq`, `xtl`, `xeus`, and `ZeroMQ` are all `xeus-zmq`'s dependencies. Check [here](https://github.com/jupyter-xeus/xeus-zmq?tab=readme-ov-file#building-from-sources) for how to install `xeus-zmq`.

:::
###### Required Python Packages
:::
- `jupyter_libyt`: Customized kernel provisioner for libyt Jupyter kernel.
- `jupyter-client` (>=8.0.0): Jupyter client.
- `jedi`: Support auto-completion in Jupyter Notebook and JupyterLab. This is optional. (If you have IPython installed, you might already have this.)


##### `-DSUPPORT_TIMER`

- Default: `-DSUPPORT_TIMER=OFF`

|                     | Notes                                                                                                             | Required Paths |
|---------------------|-------------------------------------------------------------------------------------------------------------------|----------------|
| Time Profiling (ON) | Support time profiling. (See  [Time Profiling]({% link DebugAndTimeProfiling/TimeProfiling.md %}#time-profiling)) |                |

### CMake
1. Toggle options, set paths and generate files to build the project. This can be done through either (a) or (b):

   (a) Set it through editing `CMakeLists.txt` at root directory. For example, this uses option [`-DSERIAL_MODE=OFF`](#-dserial_modeonoff-defaultoff) and provides `MPI_PATH`:
   ```cmake
   option(SERIAL_MODE "Compile library for serial process" ON)
   set(MPI_PATH "<path-to-mpi-prefix>" CACHE PATH "Path to MPI installation prefix (-DSERIAL_MODE=OFF)")
   ```

   (b) Set the options and paths through command line. For example, the flags here are equivalent to the above:
   ```bash
   -DSERIAL_MODE=OFF -DMPI_PATH=<path-to-mpi-prefix>
   ```

2. (Optional) Set the GCC compiler, export the environment variable `CC` to target `gcc` compiler and `CXX` to target `g++` compiler before running `cmake`. For example:
   ```bash
   export CC=/software/gcc/bin/gcc
   export CXX=/software/gcc/bin/g++
   ```
   > {octicon}`info;1em;sd-text-info;` It should support `c++14`.
   
3. Generate files for project, `<1-(b)>` contains the flags in step 1-(b):
   ```bash
   cd libyt # go to the root of the project
   cmake -B <build-dir-name> -S . <1-(b)>
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
cd libyt                                                     # go to project root directory
export CC=/software/gcc/8.4.0/bin/gcc                        # set gcc compiler
export CXX=/software/gcc/8.4.0/bin/g++                       # set g++ compiler
rm -rf build                                                 # clean up previous build
cmake -B build -S . -DSERIAL_MODE=ON                         # generate files for project
cmake --build build                                          # build the project
cmake --install build --prefix /home/user/softwares/libyt    # install
```

- The following builds `libyt` in parallel mode using user designated MPI compiler and then installs the library in `/home/user/softwares/libyt`:
```bash
cd libyt
rm -rf build
cmake -B build -S . -DSERIAL_MODE=OFF -DMPI_PATH=/software/openmpi/4.1.1-gnu
cmake --build build
cmake --install build --prefix /home/user/softwares/libyt
```

## Required Python Package
To use [`yt`](https://yt-project.org/) as the core analytic tool, we need to install `yt_libyt`, a `yt` frontend for `libyt`.

### yt
- Project website: [https://yt-project.org/](https://yt-project.org/)
- Install from PyPI:
```bash
pip install yt
```

### yt_libyt
- Project website: [https://github.com/data-exp-lab/yt_libyt](https://github.com/data-exp-lab/yt_libyt)
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


### jupyter_libyt
- Project website: [https://github.com/yt-project/jupyter_libyt](https://github.com/yt-project/jupyter_libyt)
- Install from source:
```bash
git clone https://github.com/yt-project/jupyter_libyt.git
cd jupyter_libyt
pip install .
```
