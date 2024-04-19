# Example

The [`example`](https://github.com/yt-project/libyt/blob/main/example) demonstrates how to implement `libyt` in adaptive mesh refinement (AMR) grid simulation.
The example has a set of pre-calculated data.
It assigns the data to MPI processes randomly to simulate the actual code of having data distributed on different processes. (Though in real world, data won't distribute randomly.) 

The steps related to implementation of `libyt` is commented like this:
```c++
// ==========================================
// libyt: 0. include libyt header
// ==========================================

// ==========================================
// libyt: [Optional] ...
// ==========================================
```

The example initializes `libyt`, loads data to `libyt` in every simulation time step in the iterative process and uses [`yt`](https://yt-project.org/) to do in situ Python analysis, and finalizes it before terminating the simulation. 
The code can be found in [`example/amr-example/example.cpp`](https://github.com/yt-project/libyt/blob/main/example/amr-example/example.cpp).

## Building and Running the Example

### Using CMake

1. Follow [Install](./how-to-install.md#install).
2. Enter example folder. Assume we build the project under `libyt/build`:
   ```bash
   cd libyt                        # go to libyt project root folder
   cd build/example/amr-example    # go to example folder
   ```

### Using Make

1. Follow [Install](./how-to-install.md#install).
2. Go to `libyt/example/amr-example` folder.
   ```bash
   cd libyt/example/amr-example
   ```
3. Update `MPI_PATH` and `LIBYT_PATH` in `Makefile`. They are installation prefix. The prefix will contain `include` and `lib` folder.
   ```makefile
   MPI_PATH := $(YOUR_MPI_PATH)
   LIBYT_PATH := $(YOUR_LIBYT_PATH)
   ```
   > {octicon}`alert;1em;sd-text-danger;` Make sure you are using the same MPI to compile `libyt` and the example.
4. Compile the code:
   - **Serial Mode (using GCC)**:
     ```bash
     make OPTIONS=-DSERIAL_MODE 
     ```
   - **Parallel Mode (using MPI)**:
     ```bash
     make
     ```

### Running the Example

1. Run the example:
    - **Serial Mode (using GCC)**:
      ```bash
      ./example
      ```
    - **Parallel Mode (using MPI)**:
      ```bash
      mpirun -np 2 ./example
      ```
2. The output results we get from the first step:
   - Density projection along z-axis, `FigName000000000_Projection_z_density.png`:

     ```{image} _static/img/AMRExample-Step1-ProjDensZ.png
     :align: center
     :scale: 50%
     ```
   - Density profile with respect to x coordinate, `FigName000000000_1d-Profile_x_density.png`:

     ```{image} _static/img/AMRExample-Step1-ProfDensX.png
     :align: center
     :scale: 50%
     ```
     
   - Slice plot of the reciprocal of the density field, `FigName000000000_Slice_z_InvDens.png`:

     ```{image} _static/img/AMRExample-Step1-SliceInvDensZ.png
     :align: center
     :scale: 50%
     ```
     

## What's Next

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

