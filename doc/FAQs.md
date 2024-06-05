# FAQs

## When Installing

### Get errors when using CMake

Make sure the folder where CMake generates build files is empty or not exist yet by removing the folder:
```bash
cd libyt
rm -rf <build-folder>
cmake -S . -B <build-folder>
```

### Get errors when linking shared library libyt.so

If we compile Python from source, and we get errors like this when compiling `libyt`:
```text
ld: <some-path-to-python-library>: relocation R_X86_64_32S against symbol `_Py_FalseStruct' can not be used when making a shared object; recompile with -fPIC
ld: failed to set dynamic section sizes: bad value
```
We need to re-compile and install Python and add `--enable-shared` when configuring.

The simplest way to avoid this is to install Python from package manager.

### Unable to link to dependencies fetched by libyt after installation

Keep `libyt` project repo after installation. `libyt` fetches and stores dependencies under `libyt/vendor` folder, so that the content can be reused in different builds.

---

## When Running Applications

### How does libyt run Python script?
`libyt` runs Python script synchronously, which means every MPI process runs the same piece of Python code.
They do the job together under the process space of MPI tasks using [`mpi4py`](https://mpi4py.readthedocs.io/en/stable/index.html).

### Have problems when running libyt in parallel mode

If we happen to get errors related to one-sided MPI, for example:
```text
*** An error occurred in MPI_Win_attach
*** reported by process [3353411585,1]
*** on win rdma window 3
*** MPI_ERR_RMA_ATTACH: Could not attach RMA segment
*** MPI_ERRORS_ARE_FATAL (processes in this win will now abort,
***    and potentially your MPI job)
3 more processes have sent help message help-mpi-errors.txt / mpi_errors_are_fatal
Set MCA parameter "orte_base_help_aggregate" to 0 to see all help / error messages
```
Remember to set `OMPI_MCA_osc=sm,pt2pt` before running the application. It is for one-sided MPI communication. For example:

```bash
OMPI_MCA_osc=sm,pt2pt mpirun -np 3 ./example
```

### Why does my program hang and how do I solve it?
Though `libyt` can execute any Python module, when it comes to reading simulation data, it requires every MPI process to participate.
The program hanging problem is due to only some MPI processes are accessing the data, but not all of them.

**Please do**:
1. Check if there is an `if` statements that makes MPI processes non-symmetric. For example, only root process runs the statement:
    ```python
    def func():
        if yt.is_root():
            ...  # <-- This statement only executes in MPI root rank
    ```
2. Move the statement out of `if yt.is_root()` (for the case here).

> {octicon}`calendar;1em;sd-text-secondary;` When accessing simulation data, `libyt` requires every process to participate.
> We are working on this in both `yt` and `libyt`.

### Have Problems when Using Interactive Prompt

#### Why can't I find the prompt `>>>`?
`>>> `  is probably immersed inside the output.
We can hit enter again, which is to provide an empty statement, and it will come out.

We can make prompt more smoothly by setting [`YT_VERBOSE`](./libyt-api/yt_initialize.md#yt_param_libyt) to `YT_VERBOSE_INFO`.

#### Where can I use interactive mode?
`libyt` interactive Python prompt only works on local machine or submit the job to HPC platforms using interactive jobs like `qsub -I` in PBS scheduler.
The reason is that the user interface is exposed in the terminal.
