# Example

The [`example`](https://github.com/yt-project/libyt/blob/main/example) demonstrates how to implement `libyt` in adaptive mesh refinement simulation.
The steps related to implementation of `libyt` is commented like this:
```c++
// ==========================================
// libyt: 0. include libyt header
// ==========================================

// ==========================================
// libyt: [Optional] ...
// ==========================================
```

The example has a set of pre-calculated data.
It assigns the data to MPI processes randomly to simulate the actual code of having data distributed on different processes. (Though in real world, data won't distribute randomly.) 

The example initializes `libyt`, loads data to `libyt` in simulation's iterative process inside for loop, and finalizes it before terminating the simulation. To know the step by step details, see [`example/example.cpp`](https://github.com/yt-project/libyt/blob/main/example/example.cpp).


## How to Compile and Run

1. Install `libyt` with [`-DSERIAL_MODE=OFF`](./how-to-install.md#-dserial_mode-off) and install Python package [`yt_libyt`](https://github.com/data-exp-lab/yt_libyt).
2. Install [`yt`](https://yt-project.org/). This example uses `yt` as the core analytic method.
3. Update `MPI_PATH` and `LIBYT_PATH` in `example/Makefile`, which is their installation prefix. The folder should contain `include`, `lib` etc.
   ```makefile
   MPI_PATH := $(YOUR_MPI_PATH)
   LIBYT_PATH := $(YOUR_LIBYT_PATH)
   ```
   > {octicon}`alert;1em;sd-text-danger;` Make sure you are using the same MPI to compile `libyt` and the example.
4. Go to `example` folder and compile.
   ```bash
   make clean
   make
   ```
5. Run `example`.
   ```bash
   mpirun -np 4 ./example
   ```

## Playground

### Activate Interactive Mode

To try out **interactive mode**, compile `libyt` with [`-DINTERACTIVE_MODE=ON`](./how-to-install.md#-dinteractive_mode-off).
Interactive mode supports [Reloading Script](./in-situ-python-analysis/reloading-script.md#reloading-script) and [Interactive Python Prompt](./in-situ-python-analysis/interactive-python-prompt.md#interactive-python-prompt) features.

Un-comment these blocks in `example.cpp`:
```c++
// file: example/example.cpp
// =======================================================================================================
// libyt: 9. activate python prompt in interactive mode
// =======================================================================================================
// Only supports when compiling libyt in interactive mode (-DINTERACTIVE_MODE)
// Start reloading script if error occurred when running inline functions, or it detects "LIBYT_STOP" file.
if (yt_run_ReloadScript("LIBYT_STOP", "RELOAD", "test_reload.py") != YT_SUCCESS) {
    fprintf(stderr, "ERROR: yt_run_ReloadScript failed!\n");
    exit(EXIT_FAILURE);
}

// Interactive prompt will start only if it detects "LIBYT_STOP" file.
if (yt_run_InteractiveMode("LIBYT_STOP") != YT_SUCCESS) {
    fprintf(stderr, "ERROR: yt_run_InteractiveMode failed!\n");
    exit(EXIT_FAILURE);
}
```

### Activate libyt Jupyter Kernel

To try out **jupyter kernel mode**, compile `libyt` with [`-DJUPYTER_KERNEL=ON`](./how-to-install.md#-djupyter_kernel-off).

Then un-comment this block in `example.cpp`:
```c++
// file: example/example.cpp
// =======================================================================================================
// libyt: 9. activate libyt Jupyter kernel for Jupyter Notebook / JupyterLab access
// =======================================================================================================
// Only supports when compiling libyt in jupyter kernel mode (-DJUPYTER_KERNEL)
// Activate libyt kernel when detects "LIBYT_STOP" file.
// False for making libyt find empty port to bind to by itself.
// True for using connection file provided by user, file name must be "libyt_kernel_connection.json".
if (yt_run_JupyterKernel("LIBYT_STOP", false) != YT_SUCCESS) {
    fprintf(stderr, "ERROR: yt_run_JupyterKernel failed!\n");
    exit(EXIT_FAILURE);
}
```

Create `LIBYT_STOP` and put it in the same folder where example executable is.
`libyt` will launch libyt Jupyter kernel and bind to unused ports automatically. See [Jupyter Notebook Access](./in-situ-python-analysis/jupyter-notebook/jupyter-notebook-access.md#jupyter-notebook-access).

### Update Python Script
The example uses `inline_script.py` for in situ Python analysis. 
To change the Python script name, simpy change `param_libyt.script` and do not include file extension `.py` in `example.cpp`. 

```c++
// file: example/example.cpp
yt_param_libyt param_libyt;
param_libyt.verbose = YT_VERBOSE_INFO;   // libyt log level
param_libyt.script = "inline_script";    // inline python script, excluding ".py"
param_libyt.check_data = false;          // check passed in data or not
```

To know more about writing inline Python script, you can refer to [Using Python for In Situ Analysis in libyt](./in-situ-python-analysis/index.md#using-python-for-in-situ-analysis-in-libyt).

