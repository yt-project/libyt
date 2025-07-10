# How to Install

```{toctree}
:hidden:

details
```

```{seealso}
Go through this page to install `libyt` that is
 fault-tolerant to Python code, supports interactive Python prompt, supports reloading script, and supports Jupyter Notebook access.
 
To turn off some of the features, go to [Details](./details.md).
```

## libyt

**Requirements:**

**[CMake](https://cmake.org/)** (>=3.15)
: A software build system.

**[pkg-config](https://www.freedesktop.org/wiki/Software/pkg-config/)**
: Generally, Linux and macOS already have pkg-config installed. It is a helper tool for finding libraries and compiling applications.

**Compiler** (It should support `c++14`)
: - `CXX`: Path to `g++` compiler.
  - `CC`: Path to `gcc` compiler.

**[Python](https://www.python.org/)** (>=3.7)
: The Python environment we want to use when doing in situ analysis. If we compile it from source, use `--enable-shared` when configuring.
  - `PYTHON_PATH`: Python installation prefix, the path contains folders `include`, `lib` etc.
  - `numpy` and other [Python Dependencies](#python-dependencies)

**[OpenMPI](https://www.open-mpi.org/)** or **[MPICH](https://www.mpich.org/)**
: MPI used for compiling simulations and for compiling `libyt` should be the same.
  - `MPI_PATH`: MPI installation prefix, the path contains folders `include`, `lib` etc.

**[Readline](https://tiswww.case.edu/php/chet/readline/rltop.html)**
: Readline library is already installed on Linux and macOS generally. If not, we can get through system package manager or compile from source ourselves. If we compile it from source, use `--with-curses` when configuring.
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

  ```{eval-rst}
  .. seealso::
     If you encounter errors like:
  
     .. code-block:: text
        
        Could NOT find LibUUID (missing: LibUUID_LIBRARY LibUUID_INCLUDE_DIR)

     These are caused by missing dependencies and can be installed by system package manager.
  ```

4. Build project and install `libyt`:
   ```bash
   cmake --build build 
   cmake --install build --prefix <libyt-install-prefix>
   ```
5. `libyt` is now installed at `<libyt-install-prefix>` folder.

## Python Dependencies

- **Option** indicates under what circumstances will we need this package.

:::{table}
:width: 100%

| Python package                                                                 | pip              |  Required version                                                                                                                                                                                                                                                                                |  Notes                                                                                                                                                                                                                                                                                            | Option                |
|--------------------------------------------------------------------------------|------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------|
| [`NumPy`](https://numpy.org/)                                                  | `numpy`          |  <2.0                                                                                                                                                                                                                                                                                            | The fundamental package for scientific computing with Python.                                                                                                                                                                                                                                    | Always                                                                                                                                                                                                                                                                                            |
| [`yt`](https://yt-project.org/)                                                | `yt`             |                                                                                                                                                                                                                                                                                                  | The core analytic tool.                                                                                                                                                                                                                                                                          | Always                                                                                                                                                                                                                                                                                            |
| [`yt_libyt`](https://github.com/data-exp-lab/yt_libyt)                         | `yt-libyt`       |                                                                                                                                                                                                                                                                                                  | `yt` frontend for `libyt`.                                                                                                                                                                                                                                                                       | Always                                                                                                                                                                                                                                                                                            |
| [`mpi4py`](https://mpi4py.readthedocs.io/en/stable/)                           | `mpi4py`         |                                                                                                                                                                                                                                                                                                  | Python bindings for the Message Passing Interface (MPI) standard. <br> {octicon}`alert;1em;sd-text-danger;` Make sure `mpi4py` used in Python and MPI used in simulation are matched. (Check how to install `mpi4py` [here](https://mpi4py.readthedocs.io/en/stable/install.html#installation).) | `-DSERIAL_MODE=OFF`                                                                                                                                                                                                                                                                               |
| [`jedi`](https://jedi.readthedocs.io/en/latest/)                               | `jedi`           |                                                                                                                                                                                                                                                                                                  | Support auto-completion in Jupyter Notebook and JupyterLab. (We will have this if IPython is already installed.)                                                                                                                                                                                 | `-DJUPYTER_KERNEL=ON`                                                                                                                                                                                                                                                                             |
| [`jupyter-client`](https://jupyter-client.readthedocs.io/en/latest/index.html) | `jupyter-client` |  >=8.0.0                                                                                                                                                                                                                                                                                         | Jupyter Client.                                                                                                                                                                                                                                                                                  | `-DJUPYTER_KERNEL=ON`                                                                                                                                                                                                                                                                             |
| [`jupyter_libyt`](https://github.com/yt-project/jupyter_libyt)                 | `jupyter-libyt`  |                                                                                                                                                                                                                                                                                                  | Jupyter kernel provisioner for `libyt`.                                                                                                                                                                                                                                                          | `-DJUPYTER_KERNEL=ON`                                                                                                                                                                                                                                                                             |
:::

```{note}
`jupyter-client` and `jupyter_libyt` are used for launching Jupyter Notebook and JupyterLab. Make sure the Python environment used for launching the notebook have them installed.

The Python used in in situ analysis which is also for compiling `libyt` and the Python used for launching Jupyter Notebook/JupyterLab might be different.
For example, when running `libyt` in HPC cluster, the Python environment is different from the one starting Jupyter Notebook on your local laptop. (See [Jupyter Notebook Access](../in-situ-python-analysis/jupyter-notebook/jupyter-notebook-access.md))
```

## FAQs

- [Get errors when using CMake](../FAQs.md#get-errors-when-using-cmake)
- [Get errors when linking shared library libyt.so](../FAQs.md#get-errors-when-linking-shared-library-libytso)
- [Unable to link to dependencies fetched by libyt after installation](../FAQs.md#unable-to-link-to-dependencies-fetched-by-libyt-after-installation)
