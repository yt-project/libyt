# Quick Start

The quick start initializes `libyt`, runs Python functions defined in inline Python script, activates Python prompt, and then finalizes it.
It does not load any data from C/C++ application to Python yet. The Python prompt is just raw Python interpreter activate by libyt API.

After reading quick start, we will have an overview of how to use `libyt` for in situ analysis.

### Building and Running the Project

```{note}
The quick start assumes the operating system is either Linux or macOS.
```

1. Follow [Install](./how-to-install/how-to-install.md#libyt).
2. Enter quick start folder. Assume we build the project under `libyt/build`:
   ```bash
   cd libyt                        # go to libyt project root folder
   cd build/example/quick-start    # go to quick start folder
   ```
3. Run the quick start example:
   - **Serial Mode (using GCC)**:
     ```bash
     ./quick-start
     ```
   - **Parallel Mode (using MPI)**:
     ```bash
     OMPI_MCA_osc=sm,pt2pt mpirun -np 2 ./quick-start
     ```
     > {octicon}`info;1em;sd-text-info;` `OMPI_MCA_osc=sm,pt2pt` is for using one-sided MPI communications.
4. This is the last few lines of the output if we run in serial mode. (If we run in parallel mode, then there will be multiple `HELLO WORLD!!` and `<class 'str'> 1 ...` printed.)
   ```text
   [YT_INFO   ] Importing YT inline analysis script "inline_script" ... done
   [YT_INFO   ] Performing YT inline analysis print_hello_world() ...
   HELLO WORLD!!
   [YT_INFO   ] Performing YT inline analysis print_hello_world() ... done.
   [YT_INFO   ] Performing YT inline analysis print_args('1',2,3.0) ...
   <class 'str'> 1
   <class 'int'> 2
   <class 'float'> 3.0
   [YT_INFO   ] Performing YT inline analysis print_args('1',2,3.0) ... done.
   ```

   In the quick start example, we call Python function defined in `inline_script.py` using [`yt_run_Function`](./libyt-api/run-python-function.md#yt_run_function)/[`yt_run_FunctionArguments`](./libyt-api/run-python-function.md#yt_run_functionarguments).

   ```{tab} C/C++
   ```c++
   // quick-start.cpp
   if (yt_run_Function("print_hello_world") != YT_SUCCESS) {
       exit(EXIT_FAILURE);
   }

   if (yt_run_FunctionArguments("print_args", 3, "\'1\'", "2", "3.0") != YT_SUCCESS) {
       exit(EXIT_FAILURE);
   } 
   
   ```
   
   ```{tab} Python
   ```python
   # inline_script.py
   def print_hello_world():
       print("HELLO WORLD!!")

   
   def print_args(*args):
       for i in args:
           print(type(i), i)
   ```

   The demo prints `HELLO WORLD!!` and prints input arguments passed into libyt API. 
   When calling the API, every process runs the Python code synchronously, and each process prints the same output. Thus, there will be multiple outputs if we run in parallel mode.
   
   In actual production run, Python package, like [`yt`](https://yt-project.org/), uses [`mpi4py`](https://mpi4py.readthedocs.io/en/stable/) to make each process collaborates and conduct analysis using Python.

5. After outputs from the Python functions, we see:
   ```text
   =====================================================================
   Inline Function                              Status         Run
   ---------------------------------------------------------------------
   * print_hello_world                          success         V
   * print_args                                 success         V
   =====================================================================
   [YT_INFO   ] Flag file 'LIBYT_STOP' is detected ... entering interactive mode
   >>>
   ```
   This is a Python prompt activated by libyt API, and it also prints out [Status Board](./in-situ-python-analysis/libyt-defined-command.md#status-board) which shows the Python function status.
   The prompt will only be activated if `LIBYT_STOP` file is detected.
   ```c++
   if (yt_run_InteractiveMode("LIBYT_STOP") != YT_SUCCESS) {
       exit(EXIT_FAILURE);
   }
   ```

   The Python prompt accepts:
   - [libyt Defined Commands](./in-situ-python-analysis/libyt-defined-command.md) to control Python function calls, retrieve error messages, etc. Type `%libyt help` to see the help messages.
   - Python statements

   This is like normal Python prompt. The prompt takes inputs from the user, and then it broadcast the inputs to other MPI processes (if we run in parallel mode), finally every process runs the same Python code synchronously.
   Since there is currently no data loaded, we are just playing with a normal Python prompt activate through libyt API ([`yt_run_InteractiveMode`](./libyt-api/yt_run_interactivemode.md)).

6. Enter `%libyt exit` will exit the prompt:
   ```shell
   >>> %libyt exit
   ```

---

### What's Next

###### Loading Data using libyt API and Do Analysis using yt and Python
Quick start has no data loaded to Python, so we are only playing with a raw Python interpreter. 
`libyt` currently supports adaptive mesh refinement (AMR) data structures.
The following shows how to load AMR data and do in situ analysis using [`yt`](https://yt-project.org/)
- [Example](./example.md)

###### Other Python Entry Points for Interactive Data Analysis
Besides Python prompt user interface ([`yt_run_InteractiveMode`](./libyt-api/yt_run_interactivemode.md)) shown here, `libyt` also supports file-based Python prompt ([`yt_run_ReloadScript`](./libyt-api/yt_run_reloadscript.md)) and Jupyter Notebook / JupyterLab access ([`yt_run_JupyterKernel`](./libyt-api/yt_run_jupyterkernel.md)).

**To try file-based Python prompt:**
1. Go to `quick-start` build folder.
2. Create `LIBYT_RELOAD`. The file is signal for `libyt` to decide whether to launch the file-based Python prompt.
3. Run the program. See more in [Reload Script](./in-situ-python-analysis/reloading-script.md).

**To try Jupyter Notebook / JupyterLab access:**
1. Go to `quick-start` build folder.
2. Create `LIBYT_JUPYTER`. The file is signal for `libyt` to decide whether to launch Jupyter kernel used for Jupyter Notebook / JupyterLab access.
3. Run the program. See more in [Jupyter Notebook Access](./in-situ-python-analysis/jupyter-notebook/jupyter-notebook-access.md).
