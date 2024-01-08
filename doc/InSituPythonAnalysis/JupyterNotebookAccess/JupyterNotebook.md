---
layout: default
title: Jupyter Notebook Access
parent: In Situ Python Analysis
has_children: true
nav_order: 6
has_toc: false
---
# Jupyter Notebook Access
{: .no_toc }
<details open markdown="block">
  <summary>
    Table of contents
  </summary>
  {: .text-delta }
- TOC
{:toc}
</details>
---

## Requirements

- Compile `libyt` in [**jupyter kernel mode**]({% link HowToInstall.md %}#-djupyter_kernelonoff-defaultoff).
- Call libyt API [`yt_run_JupyterKernel`]({% link libytAPI/ActivateJupyterKernel.md %}#yt_run_jupyterkernel).
- Python package [`jupyter_libyt`]({% link HowToInstall.md %}#jupyter_libyt), [`jupyter-client`](https://jupyter-client.readthedocs.io/en/stable/index.html), and [`jedi`](https://jedi.readthedocs.io/en/latest/).

## Setting Up

During simulation runtime, `libyt` will activate libyt Jupyter kernel. 
We need another process to start Jupyter Notebook/JupyterLab and connect to libyt Jupyter kernel. 

### Starting Jupyter Notebook / JupyterLab

> :information_source: The process of starting Jupyter Notebook/JupyterLab and running simulation are separate and independent.

1. Get [`jupyter_libyt`]({% link HowToInstall.md %}#jupyter_libyt) and [`jupyter-client`](https://jupyter-client.readthedocs.io/en/stable/index.html).
2. After installing `libyt`, add `<libyt-install-dir>/share/jupyter` to `JUPYTER_PATH`.
   ```bash
   export JUPYTER_PATH=<libyt-install-dir>/share/jupyter:$JUPYTER_PATH
   ```
   Check if `libyt_kernel` is listed:
   ```bash
   jupyter kernelspec list
   ```
3. Export environment variable `LIBYT_KERNEL_INFO_DIR` to where the simulation executable directory is.
   ```bash
   export LIBYT_KERNEL_INFO_DIR=<path-to-simulation-dir>
   ```
4. Launch Jupyter Notebook/JupyterLab
   ```bash
   jupyter notebook 
   jupyter-lab
   ```
5. Click `Libyt` to connect to libyt Jupyter kernel once the libyt Jupyter kernel is activated.

### Running Simulation

1. Compile `libyt` in [**jupyter kernel mode**]({% link HowToInstall.md %}#-djupyter_kernelonoff-defaultoff).
2. Call libyt API [`yt_run_JupyterKernel`]({% link libytAPI/ActivateJupyterKernel.md %}#yt_run_jupyterkernel). When flag file is detected, `libyt` will activate libyt Jupyter kernel.
3. libyt API [`yt_run_JupyterKernel`]({% link libytAPI/ActivateJupyterKernel.md %}#yt_run_jupyterkernel) returns `YT_SUCCESS` after libyt Jupyter kernel shuts down (see [How to Exit](#how-to-exit)).
4. Simulation can continue its process.

### Example

- [Connecting to kernel on local machine]({% link InSituPythonAnalysis/JupyterNotebookAccess/JupyterLocalMachine.md %}#connecting-to-kernel-on-local-machine)
- [Connecting to kernel on HPC cluster]({% link InSituPythonAnalysis/JupyterNotebookAccess/JupyterLocalMachine.md %}#connecting-to-kernel-on-hpc-cluster)

## Using Jupyter Notebook / JupyterLab

### Basics

![](../../assets/imgs/JupyterNB-Basics.png)

#### Python Prompt and libyt Defined Commands

The cell takes Python statements and [libyt Defined Commands]({% link InSituPythonAnalysis/libytCommand.md %}#libyt-defined-commands).
It can also import other Python modules

##### How does it execute Python statements and libyt defined commands?
{: .no_toc }
1. Takes user inputs in the cell.
2. MPI root process broadcasts the inputs to other processes.
3. Every other MPI process executes the same piece of input synchronously.
4. Get outputs from each process and print feedbacks on the screen.

> :warning: Changes is kept and maintain in user's inline script's namespace in situ analysis in the following round.

#### Auto-Completion

Use `TAB` key to do auto-completion. 

> :information_source: [`jedi`](https://jedi.readthedocs.io/en/latest/) must be installed to use this feature. Generally, if you have `IPython` installed, you probably already have `jedi`.

#### User Interface

This is traditional Jupyter Notebook UI.

> :lizard: If the notebook is connected to libyt Jupyter kernel, restarting only exits the kernel, and libyt API [`yt_run_JupyterKernel`]({% link libytAPI/ActivateJupyterKernel.md %}#yt_run_jupyterkernel) returns. It won't restart another kernel.

### How to Exit

Go to "Running Terminals and Kernels" -> "Shutdown Kernel". 

"Interrupt the kernel" button (:black_medium_square:) won't have any effect.


### Reconnecting to libyt Jupyter Kernel

After exiting the kernel and returning to simulation process, we don't need to close the entire notebook to reconnect to the new libyt Jupyter kernel launched by the following step.
We can press "Restart" button (:arrows_counterclockwise:) to reconnect to libyt Jupyter kernel.

> :lizard: Make sure you already disconnect libyt Jupyter kernel before reconnecting. Otherwise, libyt Jupyter kernel will simply shutdown the kernel. There will be a pop-up window asking "if you want to restart" if you are already connected to a libyt kernel. 

## Known Limitations
- The functionality is limited to taking Python inputs and printing outputs from all the MPI processes only. `libyt` hasn't done implementing Jupyter's full feature, like data streaming, supporting `ipwidgets` yet.
- See [Limitations in MPI Related Python Tasks]({% link InSituPythonAnalysis/Limitations.md %}#limitations-in-mpi-related-python-tasks).

## FAQs
