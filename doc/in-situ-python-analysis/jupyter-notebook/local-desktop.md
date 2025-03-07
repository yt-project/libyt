# Example -- Connecting to Kernel on Local Machine

## Running Simulation

1. Use libyt API [`yt_run_JupyterKernel`](../../libyt-api/yt_run_jupyterkernel.md#yt_run_jupyterkernel) in simulation and compile:
   ```c++
   if (yt_run_JupyterKernel("LIBYT_STOP", false) != YT_SUCCESS) {
       // some error message
   }
   ```
   It will launch libyt kernel once it detects `LIBYT_STOP`.
   To make `libyt` find and bind to unused port automatically, set it to `false`.
2. Wait for connection from Jupyter Notebook / JupyterLab (See below [Connecting to Kernel](#connecting-to-kernel))
3. Shutdown the kernel (See [How to Exit](./frontend.md#how-to-exit)). 

## Connecting to Kernel

Go through [Starting Jupyter Notebook/JupyterLab](./jupyter-notebook-access.md#starting-jupyter-notebookjupyterlab).
