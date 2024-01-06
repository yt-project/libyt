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

> :information_source: The process of starting Jupyter Notebook/JupyterLab and running simulation are independent.

1. Get [`jupyter_libyt`]({% link HowToInstall.md %}#jupyter_libyt) and [`jupyter-client`](https://jupyter-client.readthedocs.io/en/stable/index.html).
2. After installing `libyt`, add `<libyt-install-dir>/share/jupyter` to `JUPYTER_PATH`.
   ```bash
   export JUPYTER_PATH=<libyt-install-dir>/share/jupyter:$JUPYTER_PATH
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
5. Click `Libyt` to connect to libyt Jupyter kernel once the simulation is running.

### Simulation

1. Compile `libyt` in [**jupyter kernel mode**]({% link HowToInstall.md %}#-djupyter_kernelonoff-defaultoff).
2. Call libyt API [`yt_run_JupyterKernel`]({% link libytAPI/ActivateJupyterKernel.md %}#yt_run_jupyterkernel). When flag file is detected, `libyt` will activate libyt Jupyter kernel.

### Example

- [Connecting to kernel on local machine]({% link InSituPythonAnalysis/JupyterNotebookAccess/JupyterLocalMachine.md %}#connecting-to-kernel-on-local-machine)
- [Connecting to kernel on HPC Cluster]({% link InSituPythonAnalysis/JupyterNotebookAccess/JupyterLocalMachine.md %}#connecting-to-kernel-on-hpc-cluster)

## Using Jupyter Notebook / JupyterLab

### Basic

### Reconnecting to libyt Jupyter Kernel

### Exit

## Known Limitations

## FAQs