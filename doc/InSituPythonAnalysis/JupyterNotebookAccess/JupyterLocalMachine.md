---
layout: default
title: Example -- Local Machine
parent: Jupyter Notebook Access
grand_parent: In Situ Python Analysis
nav_order: 2
---
# Connecting to Kernel on Local Machine
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

## Running Simulation

1. Use libyt API [`yt_run_JupyterKernel`]({% link libytAPI/ActivateJupyterKernel.md %}#yt_run_jupyterkernel) in simulation and compile:
   ```c++
   if (yt_run_JupyterKernel("LIBYT_STOP", false) != YT_SUCCESS) {
       // some error message
   }
   ```
   It will launch libyt Jupyter kernel once it detects `LIBYT_STOP`.
   To make `libyt` find and bind to unused port automatically, set it to `false`.
2. Wait for connection from Jupyter Notebook / JupyterLab (See below [Connecting to Kernel](#connecting-to-kernel))
3. Shutdown the kernel. ([How to Exit]({% link InSituPythonAnalysis/JupyterNotebookAccess/JupyterNotebook.md %}#how-to-exit))

## Connecting to Kernel

Go through [Starting Jupyter Notebook / JupyterLab]({% link InSituPythonAnalysis/JupyterNotebookAccess/JupyterNotebook.md %}#starting-jupyter-notebook--jupyterlab).

