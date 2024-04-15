# How to Install

## libyt C Library

### Prerequisites
- CMake (>=3.15)
- pkg-config: Generally, Linux and macOS already have pkg-config installed.
- GCC compiler (>4.8): It should be able to support `c++14`.
  - `CXX`: Path to `g++` compiler.
  - `CC`: Path to `gcc` compiler.
- Python (>=3.7): The Python environment we want to use when doing in situ analysis.
  - `PYTHON_PATH`: Python installation prefix, the path contains folders `include`, `lib` etc. 
  - `NumPy`: Should have `NumPy` installed.

### Step-by-Step Instructions
1. Toggle options, set paths and generate files to build the project. This can be done through either (a) or (b):

   (a) Set it through editing `CMakeLists.txt` at root directory. For example, this uses option [`-DSERIAL_MODE=OFF`](#-dserial_mode-off) and provides `MPI_PATH`:
   ```cmake
   option(SERIAL_MODE "Compile library for serial process" OFF)
   set(MPI_PATH "<path-to-mpi-prefix>" CACHE PATH "Path to MPI installation prefix (-DSERIAL_MODE=OFF)")
   ```

   (b) Set the options and paths through command line. For example, the flags here are equivalent to the above:
   ```bash
   -DSERIAL_MODE=OFF -DMPI_PATH=<path-to-mpi-prefix>
   ```

2. [Optional] Set the GCC compiler, export the environment variable `CC` to target `gcc` compiler and `CXX` to target `g++` compiler before running `cmake`. For example:
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

4. Build the project:
   ```bash
   cmake --build <build-dir-name>
   ```

5. Install the library:
   ```bash
   cmake --install <build-dir-name> --prefix <libyt-install-prefix> 
   ```

### Example
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
  cd libyt                                                                       # go to project root directory
  rm -rf build                                                                   # clean up previous build
  cmake -B build -S . -DSERIAL_MODE=OFF -DMPI_PATH=/software/openmpi/4.1.1-gnu   # set mpi path and generate files for project
  cmake --build build                                                            # build the project
  cmake --install build --prefix /home/user/softwares/libyt                      # install
  ```

### Details
#### Options (=Default Value)
The options are mutually independent to each other. 

###### `-DSERIAL_MODE` (=`OFF`)

:::{table}
:width: 100%

|                         | Notes                        | Dependency | Python dependency                          |
|-------------------------|------------------------------|------------|--------------------------------------------|
| **Parallel Mode** (OFF) | Compile `libyt` using MPI.   | `MPI_PATH` | [`mpi4py`](https://mpi4py.readthedocs.io/) |
| **Serial Mode** (ON)    | Compile `libyt` using GCC.   |            |                                            |
:::

> {octicon}`alert;1em;sd-text-danger;` Make sure Python bindings for MPI (`mpi4py`), MPI used for compiling simulation and `libyt` are the same. Check how to install `mpi4py` [here](https://mpi4py.readthedocs.io/en/stable/install.html#installation).

###### `-DINTERACTIVE_MODE` (=`OFF`)

|                           | Notes                                                                                                                                                                                                                                                                                        | Dependency      |
|---------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------|
| **Normal Mode** (OFF)     | Shut down and terminate all the processes including simulation, if error occurs during in situ analysis.                                                                                                                                                                                     |                 |
| **Interactive Mode** (ON) | Will not terminate the processes if error occurs while doing in situ analysis. Support [Interactive Python Prompt](./in-situ-python-analysis/interactive-python-prompt.md#interactive-python-prompt) and [Reloading Script](./in-situ-python-analysis/reloading-script.md#reloading-script). | `READLINE_PATH` |

###### `-DJUPYTER_KERNEL` (=`OFF`)

|                              | Notes                                                                                                                                                                             | Dependency                                                                                                 | Python dependency                                                                                                                                                                     |
|------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Jupyter Kernel Mode** (ON) | Activate Jupyter kernel and enable JupyterLab UI. (See  [Jupyter Notebook Access](./in-situ-python-analysis/jupyter-notebook/jupyter-notebook-access.md#jupyter-notebook-access)) | `nlohmann_json_DIR` <br> `cppzmq_DIR` <br> `xtl_DIR` <br> `xeus_DIR` <br> `xeus-zmq_DIR` <br> `ZeroMQ_DIR` | [`jupyter_libyt`](#jupyter_libyt) <br> [`jupyter-client`](https://jupyter-client.readthedocs.io/en/stable/index.html) <br> [`jedi`](https://jedi.readthedocs.io/en/latest/)(Optional) |

###### `-DSUPPORT_TIMER` (=`OFF`)

:::{table}
:width: 100%

|                          | Notes                                                                                                  |
|--------------------------|--------------------------------------------------------------------------------------------------------|
| **Time Profiling** (ON)  | Support time profiling. (See [Time Profiling](./debug-and-profiling/time-profiling.md#time-profiling)) |
:::

#### Dependency Table

- **Dependency path** indicates the required path variable name in CMake, and the required version.
- **Notes** are things worth notice.
- **Option** indicates under what circumstances will we need this dependency.
- **Get by libyt** indicates whether libyt will fetch and build the dependency itself, if the paths aren't provided. The downloaded content will be stored under `libyt/vendor`.

| Dependency path                            | Notes                                                                                                                                                                                                                                                                                      | Option                  | Get by libyt |
|--------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------|:------------:|
| `MPI_PATH`                                 | MPI installation prefix. The path contains folders `include` and `lib`. <br> {octicon}`alert;1em;sd-text-danger;` Make sure you are using the same MPI to compile `libyt` and your simulation code.                                                                                        | `-DSERIAL_MODE=OFF`     |      No      |
| `READLINE_PATH`                            | [GNU `readline` library](https://tiswww.case.edu/php/chet/readline/rltop.html) installation prefix. The path contains folders `include` and `lib`. <br> {octicon}`info;1em;sd-text-info;` Generally, this library exists in Linux and macOS, we don't need to explicitly provide the path. | `-DINTERACTIVE_MODE=ON` |      No      |
| `nlohmann_json_DIR` <br> (>=3.2.0, <4.0.0) | Path to `nlohmann_jsonConfig.cmake` after installing [`nlohmann_json`](https://github.com/nlohmann/json).                                                                                                                                                                                  | `-DJUPYTER_KERNEL=ON`   |     Yes      |
| `xtl_DIR` <br> (>=0.7.0, <0.8.0)           | Path to `xtlConfig.cmake` after installing [`xtl`](https://github.com/xtensor-stack/xtl).                                                                                                                                                                                                  | `-DJUPYTER_KERNEL=ON`   |     Yes      |
| `ZeroMQ_DIR` <br> (>=4.2.5, <5.0.0)        | Path to `ZeroMQConfig.cmake` after installing [`ZeroMQ`](https://github.com/zeromq/libzmq). (Some system may already have ZeroMQ installed.)                                                                                                                                               | `-DJUPYTER_KERNEL=ON`   |     Yes      |
| `cppzmq_DIR` <br> (>=4.8.1, <5.0.0)        | Path to `cppzmqConfig.cmake` after installing [`cppzmq`](https://github.com/zeromq/cppzmq).                                                                                                                                                                                                | `-DJUPYTER_KERNEL=ON`   |     Yes      |
| `xeus_DIR` <br> (>=3.0.0, <4.0.0)          | Path to `xeusConfig.cmake` after installing [`xeus`](https://github.com/jupyter-xeus/xeus).                                                                                                                                                                                                | `-DJUPYTER_KERNEL=ON`   |     Yes      | 
| `xeus-zmq_DIR` <br> (1.x release)          | Path to `xeus-zmqConfig.cmake` after installing [`xeus-zmq`](https://github.com/jupyter-xeus/xeus-zmq).                                                                                                                                                                                    | `-DJUPYTER_KERNEL=ON`   |     Yes      | 


## Python Dependency

- **Python package** required when performing in situ analysis.
- **Notes** are things worth notice.
- **Option** indicates under what circumstances will we need this package.

:::{table}
:width: 100%

| Python package                                                                                | Notes                                                                                                                                                                                                                                                                                            | Option                |
|-----------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------|
| [`yt`](https://yt-project.org/)                                                               | The core analytic tool.                                                                                                                                                                                                                                                                          | Always                |
| [`yt_libyt`](https://github.com/data-exp-lab/yt_libyt)                                        | `yt` frontend for `libyt`.                                                                                                                                                                                                                                                                       | Always                |
| [`mpi4py`](https://mpi4py.readthedocs.io/en/stable/)                                          | Python bindings for the Message Passing Interface (MPI) standard. <br> {octicon}`alert;1em;sd-text-danger;` Make sure `mpi4py` used in Python and MPI used in simulation are matched. (Check how to install `mpi4py` [here](https://mpi4py.readthedocs.io/en/stable/install.html#installation).) | `-DSERIAL_MODE=OFF`   |
| [`jupyter_libyt`](https://github.com/yt-project/jupyter_libyt)                                | Jupyter kernel provisioner for `libyt`.                                                                                                                                                                                                                                                          | `-DJUPYTER_KERNEL=ON` |
| [`jupyter-client`](https://jupyter-client.readthedocs.io/en/latest/index.html) <br> (>=8.0.0) | Jupyter Client.                                                                                                                                                                                                                                                                                  | `-DJUPYTER_KERNEL=ON` |
| [`jedi`](https://jedi.readthedocs.io/en/latest/)                                              | Support auto-completion in Jupyter Notebook and JupyterLab. (We will have this if IPython is already installed.)                                                                                                                                                                                 | `-DJUPYTER_KERNEL=ON` |
:::

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

## FAQs

### Get Errors when Using CMake

Make sure the folder where CMake generates build files to is empty or not exist yet by removing the folder:
```bash
cd libyt
rm -rf <build-folder>
cmake -S . -B <build-folder>
```

### Unable to Link to Dependencies Fetched by libyt After Installation

Keep `libyt` project repo after installation. `libyt` fetches and stores dependencies under `libyt/vendor` folder, so that the content can be reused in different builds.
