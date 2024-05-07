# Example

The example `libyt/example/amr-example` demonstrates how to implement `libyt` in an adaptive mesh refinement (AMR) grid simulation.
The example has a set of pre-calculated data.
It assigns the data to MPI processes randomly to simulate data distributed on different processes. (Though in real world, data won't distribute randomly.) 

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
3. Compile the code:
   - **Serial Mode (using GCC)**:
     ```bash
     make OPTIONS=-DSERIAL_MODE LIBYT_PATH=<libyt-install-prefix>
     ```
   - **Parallel Mode (using MPI)**:
     ```bash
     make MPI_PATH=<mpi-install-prefix> LIBYT_PATH=<libyt-install-prefix> 
     ```
     > {octicon}`alert;1em;sd-text-danger;` Make sure you are using the same MPI to compile `libyt` and the example.

### Running the Example

1. Run the example:
    - **Serial Mode (using GCC)**:
      ```bash
      ./example
      ```
    - **Parallel Mode (using MPI)**:
      ```bash
      OMPI_MCA_osc=sm,pt2pt mpirun -np 2 ./example
      ```
      > {octicon}`info;1em;sd-text-info;` `OMPI_MCA_osc=sm,pt2pt` is for using one-sided MPI communications.
2. The output results we get from the first step:
   - Density projection along z-axis, `FigName000000000_Projection_z_density.png`:

     ```{tab} ProjectionPlot
     ```{image} _static/img/AMRExample-Step1-ProjDensZ.png
     :align: center
     :scale: 50%
     ```

     ```{tab} Python
     ```python
     def yt_inline_ProjectionPlot( fields ):
         ds = yt_libyt.libytDataset()
         prjz = yt.ProjectionPlot(ds, 'z', fields)
     
         if yt.is_root():
             prjz.save()
     ```     

   - Density profile with respect to x coordinate, `FigName000000000_1d-Profile_x_density.png`:

     ```{tab} ProfilePlot
     ```{image} _static/img/AMRExample-Step1-ProfDensX.png
     :align: center
     :scale: 50%
     ```
     
     ```{tab} Python
     ```python
     def yt_inline_ProfilePlot():
         ds = yt_libyt.libytDataset()
         profile = yt.ProfilePlot(ds, "x", ["density"])
     
         if yt.is_root():
             profile.save()
     ```

   - Slice plot of the reciprocal of the density field, `FigName000000000_Slice_z_InvDens.png`:

     ```{tab} SlicePlot
     ```{image} _static/img/AMRExample-Step1-SliceInvDensZ.png
     :align: center
     :scale: 50%
     ```
     
     ```{tab} Python
     ```python
     def yt_derived_field_demo():
         ds = yt_libyt.libytDataset()
         slc = yt.SlicePlot(ds, "z", ("gamer", "InvDens"))
   
         if yt.is_root():
             slc.save()
     ```
     
   - Particle plot of the level of a grid. Each grid has one particle with its position set at the center of the grid with value equals to grid level. The result is `FigName000000000_Particle_z_Level.png`. We can only see some tiny dots in the figure, since there is only one particle in each grid. (The plot tries to demonstrate the particle functionality.)

     ```{tab} ParticlePlot
     ```{image} _static/img/AMRExample-Step1-ParticleLevelZ.png
     :align: center
     :scale: 50%
     ```
     
     ```{tab} Python
     ```python
     def yt_inline_ParticlePlot():
         ds = yt_libyt.libytDataset()
         par = yt.ParticlePlot(ds, "particle_position_x", "particle_position_y", "Level", center = 'c')
     
         if yt.is_root():
             par.save()
     ```
   
   After loading simulation data using libyt API, we call Python functions defined in `inline_script.py` using [`yt_run_Function`](./libyt-api/run-python-function.md#yt_run_function)/[`yt_run_FunctionArguments`](./libyt-api/run-python-function.md#yt_run_functionarguments) and use `yt` to analyze the data and plot figures. For how to use `yt` see:
     - [Using `yt`](./in-situ-python-analysis/using-yt.md)

---

## What's Next

###### Change Python Script Name
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

###### Other Python Entry Points for Interactive Data Analysis

**To try interactive Python prompt:**
1. Go to `amr-example` build folder.
2. Create `LIBYT_STOP`. The file is signal for `libyt` to decide whether to launch the file-based Python prompt.
3. Run the program. See more in [Interactive Python Prompt](./in-situ-python-analysis/interactive-python-prompt.md).

**To try file-based Python prompt:**
1. Go to `amr-example` build folder.
2. Create `LIBYT_RELOAD`. The file is signal for `libyt` to decide whether to launch the file-based Python prompt.
3. Run the program. See more in [Reload Script](./in-situ-python-analysis/reloading-script.md).

**To try Jupyter Notebook / JupyterLab access:**
1. Go to `amr-example` build folder.
2. Create `LIBYT_JUPYTER`. The file is signal for `libyt` to decide whether to launch Jupyter kernel used for Jupyter Notebook / JupyterLab access.
3. Run the program. See more in [Jupyter Notebook Access](./in-situ-python-analysis/jupyter-notebook/jupyter-notebook-access.md).