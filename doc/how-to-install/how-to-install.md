# How to Install

```{toctree}
:hidden:

details
```

> {octicon}`info;1em;sd-text-info;` Go through [Install](#install) and get all [Python Dependencies](#python-dependencies) to get full feature of `libyt`.
> Ignore [Details](#details) unless we want to tweak `libyt` based on our needs.

## libyt C Library

### Install
- **CMake** (>=3.15)
- **pkg-config**: Generally, Linux and macOS already have pkg-config installed.
- **GCC compiler** (>4.8): It should be able to support `c++14`.
  - `CXX`: Path to `g++` compiler.
  - `CC`: Path to `gcc` compiler.
- **Python** (>=3.7): The Python environment we want to use when doing in situ analysis.
  - `PYTHON_PATH`: Python installation prefix, the path contains folders `include`, `lib` etc. 
  - `NumPy`: Should have `NumPy` installed.
  - Other [Python Dependencies](#python-dependencies)
- **MPI**: MPI used for compiling simulations and `libyt` needs to be the same.
  - `MPI_PATH`: MPI installation prefix, the path contains folders `include`, `lib` etc.
- **Readline**: [GNU `readline` library](https://tiswww.case.edu/php/chet/readline/rltop.html) is already installed on Linux and macOS generally. If not, we can get through system package manager or compile from source ourselves. (Use `--with-curses` when configuring if we compile from source.)
  - `READLINE_PATH`: `readline` installation prefix. Provide the path if it is not in system search path.

**Follow the steps to install `libyt` that is fault-tolerant to Python code, and supports interactive Python prompt, reloading script, and Jupyter Notebook access:**

1. Shallow clone `libyt` and enter the folder:
   ```bash
   git clone --depth 1 https://github.com/yt-project/libyt "libyt"
   cd libyt
   ```
2. [Optional] Set `gcc` and `g++` compiler:
   ```bash
   export CC=<path-to-gcc>
   export CXX=<path-to-g++>
   ```
3. Generate build files in `build` folder:
   - **Serial Mode (using GCC)**:
     ```bash
     rm -rf build
     cmake -S . -B build -DSERIAL_MODE=ON \
                         -DINTERACTIVE_MODE=ON \
                         -DJUPYTER_KERNEL=ON \
                         -DPYTHON_PATH=<your-python-prefix>
     ```
   - **Parallel Mode (using MPI)**:
     ```bash
     rm -rf build
     cmake -S . -B build -DINTERACTIVE_MODE=ON \
                         -DJUPYTER_KERNEL=ON \
                         -DPYTHON_PATH=<your-python-prefix> \
                         -DMPI_PATH=<your-mpi-prefix>
     ```
4. Build project and install `libyt`:
   ```bash
   cmake --build build 
   cmake --install build --prefix <libyt-install-prefix>
   ```
5. `libyt` is now installed at `<libyt-install-prefix>` folder.


## Python Dependencies

- **Python package** required when performing in situ analysis.
- **Notes** are things worth notice.
- **Option** indicates under what circumstances will we need this package.

:::{table}
:width: 100%

| Python package                                                                                | Notes                                                                                                                                                                                                                                                                                            | Option                |
|-----------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------|
| [`NumPy`](https://numpy.org/) <br> (<2.0)                                                     | The fundamental package for scientific computing with Python.                                                                                                                                                                                                                                    | Always                |
| [`yt`](https://yt-project.org/)                                                               | The core analytic tool.                                                                                                                                                                                                                                                                          | Always                |
| [`yt_libyt`](https://github.com/data-exp-lab/yt_libyt)                                        | `yt` frontend for `libyt`.                                                                                                                                                                                                                                                                       | Always                |
| [`mpi4py`](https://mpi4py.readthedocs.io/en/stable/)                                          | Python bindings for the Message Passing Interface (MPI) standard. <br> {octicon}`alert;1em;sd-text-danger;` Make sure `mpi4py` used in Python and MPI used in simulation are matched. (Check how to install `mpi4py` [here](https://mpi4py.readthedocs.io/en/stable/install.html#installation).) | `-DSERIAL_MODE=OFF`   |
| [`jedi`](https://jedi.readthedocs.io/en/latest/)                                              | Support auto-completion in Jupyter Notebook and JupyterLab. (We will have this if IPython is already installed.)                                                                                                                                                                                 | `-DJUPYTER_KERNEL=ON` |
| [`jupyter-client`](https://jupyter-client.readthedocs.io/en/latest/index.html) <br> (>=8.0.0) | Jupyter Client.                                                                                                                                                                                                                                                                                  | `-DJUPYTER_KERNEL=ON` |
| [`jupyter_libyt`](https://github.com/yt-project/jupyter_libyt)                                | Jupyter kernel provisioner for `libyt`.                                                                                                                                                                                                                                                          | `-DJUPYTER_KERNEL=ON` |
:::

> {octicon}`alert;1em;sd-text-danger;` `jupyter-client` and `jupyter_libyt` are used for launching Jupyter Notebook and JupyterLab. Make sure the Python environment used for launching the notebook have them installed.
> 
> The Python used in in situ analysis which is also for compiling `libyt` and the Python for launching Jupyter Notebook/JupyterLab might be different.
> For example, when running `libyt` in HPC cluster and connecting to it through your local laptop. (See [Jupyter Notebook Access](../in-situ-python-analysis/jupyter-notebook/jupyter-notebook-access.md))

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
