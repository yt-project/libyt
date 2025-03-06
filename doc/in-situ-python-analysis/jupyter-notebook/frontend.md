# Using Jupyter Notebook and JupyterLab


## Frontend

![](../../_static/img/JupyterNB-Basics.png)

### Python Prompt and libyt Defined Commands

The cell takes Python statements and [libyt Defined Commands](../libyt-defined-command.md#libyt-defined-commands).
It can also import other Python modules.

```{tip} 
Put Python statements and libyt defined commands in separate cells.
```

:::
#### How does it execute Python statements and libyt defined commands?
:::
1. Takes user inputs in the cell.
2. MPI root process broadcasts the inputs to other processes.
3. Every MPI process executes the same piece of input synchronously.
4. Get outputs from each process and print feedbacks on the screen.

```{attention}
Changes is kept and maintained in user's inline [`script`](../../libyt-api/yt_initialize.md#yt_param_libyt)'s namespace, and it will be brought to the following round of in situ analysis.
```

### Auto-Completion

Use `TAB` key to do auto-completion.

[`jedi`](https://jedi.readthedocs.io/en/latest/) must be installed to use this feature. Generally, if you have `IPython` installed, you probably already have `jedi`.

### User Interface

This is the same as traditional Jupyter Notebook UI. Be careful when sending kernel related instructions (ex: shutdown/interrupt/restart a kernel).

```{attention}
If the notebook is connected to libyt kernel, restarting only exits the kernel, and libyt API [`yt_run_JupyterKernel`](../../libyt-api/yt_run_jupyterkernel.md#yt_run_jupyterkernel) returns. It won't restart another kernel by itself.
```

## How to Exit

Go to `Running Terminals and Kernels` -> `Shutdown Kernel`.

## Reconnecting to libyt Jupyter Kernel

After exiting the kernel, the simulation process may continue its job.
While waiting for a new libyt kernel being launched, we don't need to close the entire notebook to reconnect to it.
We can choose the libyt kernel from the kernel list once libyt kernel is activated again.