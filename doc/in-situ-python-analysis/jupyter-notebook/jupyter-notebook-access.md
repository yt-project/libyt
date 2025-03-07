# Jupyter Notebook Access

```{toctree}
:hidden:

frontend
local-desktop
remote-cluster
```

## Requirements

- Compile `libyt` in **jupyter kernel mode** ([`-DJUPYTER_KERNEL=ON`](../../how-to-install/details.md#-djupyter_kernel-off)).
- Call libyt API [`yt_run_JupyterKernel`](../../libyt-api/yt_run_jupyterkernel.md#yt_run_jupyterkernel).
- Python package [`jupyter_libyt`](https://github.com/yt-project/jupyter_libyt) and [`jedi`](https://jedi.readthedocs.io/en/latest/). If `ipython` is already installed, `jedi` is likely installed as well.

## Setting Up

During simulation runtime, `libyt` will activate libyt Jupyter kernel (libyt kernel). 
We need another process to start Jupyter Notebook/JupyterLab and connect to libyt kernel. 

```{tip}
The process of starting Jupyter Notebook/JupyterLab and running simulation are separate and independent. Which means the Python used to launch the Jupyter Notebook/JupyterLab doesn't need to be the same as the one used in the simulation.
```

### Starting Jupyter Notebook/JupyterLab

:::
##### Step 1 -- Get `jupyter_libyt`
:::

```bash
pip install jupyter-libyt
```
:::
##### Step 2 -- Make sure libyt kernel is discoverable by Jupyter
:::

::: 
###### `jupyter-libyt` == 0.1.0 
:::

After installing `libyt`, add `<libyt-install-prefix>/share/jupyter` to [`JUPYTER_PATH`](https://docs.jupyter.org/en/latest/use/jupyter-directories.html#envvar-JUPYTER_PATH).
```bash
export JUPYTER_PATH=<libyt-install-prefix>/share/jupyter:$JUPYTER_PATH
```

:::
###### `jupyter-libyt` >= 0.2.0
:::

Installing `jupyter-libyt` will also install the kernel spec for `libyt_kernel`.

```{caution}
Since `jupyter-libyt` 0.2.0, it has dropped the support for Python 3.7. If you are using Python 3.7, please use `jupyter-libyt` 0.1.0.
```

:::
###### Check if `libyt_kernel` is listed:
:::

```bash
jupyter kernelspec list
```

:::
##### Step 3 -- Set Env variable `LIBYT_KERNEL_INFO_DIR`
:::

Export environment variable `LIBYT_KERNEL_INFO_DIR` to where the simulation executable directory is.
```bash
export LIBYT_KERNEL_INFO_DIR=<path-to-simulation-dir>
```

```{tip}
Jupyter Notebook / JupyterLab will grab the connection information file `libyt_kernel_connection.json` in the folder `LIBYT_KERNEL_INFO_DIR`.
```

:::
##### Step 4 -- Launch Jupyter Notebook/JupyterLab
:::

```bash
jupyter notebook # launch Jupyter Notebook
jupyter lab      # launch JupyterLab
```

:::
##### Step 5 -- Connect to libyt kernel
:::

Click `Libyt` to connect to libyt kernel once the simulation is running and libyt kernel is activated.

### Running Simulation

1. Compile `libyt` in **jupyter kernel mode** ([`-DJUPYTER_KERNEL=ON`](../../how-to-install/details.md#-djupyter_kernel-off)).
2. Call libyt API [`yt_run_JupyterKernel`](../../libyt-api/yt_run_jupyterkernel.md#yt_run_jupyterkernel). When flag file is detected, `libyt` will activate libyt kernel.
3. libyt API [`yt_run_JupyterKernel`](../../libyt-api/yt_run_jupyterkernel.md#yt_run_jupyterkernel) returns `YT_SUCCESS` after libyt kernel shuts down by Jupyter Notebook / JupyterLab (see [How to Exit](./frontend.md#how-to-exit)).
4. Simulation can continue its process.

## Example

- [Connecting to Kernel on Local Machine](./local-desktop.md#example----connecting-to-kernel-on-local-machine)
- [Connecting to Kernel on HPC Cluster](./remote-cluster.md#example----connecting-to-kernel-on-hpc-cluster)

## Known Limitations
- The functionality is limited to taking Python inputs and printing outputs from all the MPI processes only. `libyt` hasn't done implementing Jupyter's full feature, like data streaming, supporting `ipwidgets` yet.
- See [Limitations in MPI Related Python Tasks](../limitation.md#limitations-in-mpi-related-python-tasks).
