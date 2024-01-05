---
layout: default
title: yt_run_JupyterKernel -- Activate Jupyter Kernel
parent: libyt API
nav_order: 11
---
# Activate Jupyter Kernel
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

## yt_run_JupyterKernel
```cpp
int yt_run_JupyterKernel(const char* flag_file_name, bool use_connection_file);
```
- Usage: Activate [Jupyter kernel]({% link InSituPythonAnalysis/JupyterNotebook.md %}#jupyter-notebook-access) if file `flag_file_name` is detected. `libyt` uses user provided connection file if `use_connection_file` is `true`. If it is false, it will pick and bind to ports and generate the connection file. The connection file name is always `libyt_kernel_connection.json`.
- Return:
    - `YT_SUCCESS`
    - `YT_FAIL`: When `libyt` is not compiled with `-DJUPYTER_KERNEL`, it returns `YT_FAIL`.

> :information_source: Must compile `libyt` with `-DJUPYTER_KERNEL`. See [How to Install]({% link HowToInstall.md %}#-djupyter_kernelonoff-defaultoff).

> :information_source: The API only launches a Jupyter kernel that exposes simulation data. For how to connect to the kernel, please go to [Jupyter Notebook Access]({% link InSituPythonAnalysis/JupyterNotebook.md %}#jupyter-notebook-access). 

## Connection File
The connection file name must be `libyt_kernel_connection.json`. This is the same connection file used in Jupyter with different file name.

For example:
```json
{
  "transport": "tcp",  
  "ip": "127.0.0.1",  
  "control_port": 53545,  
  "shell_port": 63994,  
  "stdin_port": 58177,  
  "iopub_port": 51243,  
  "hb_port": 61501,  
  "signature_scheme": "hmac-sha256",  
  "key": "64e13a6faaf1470eb1f86df565543923"
}
```

## Example
This forces `libyt` to use user provided connection file. `libyt` will launch Jupyter kernel when it detects `LIBYT_STOP` file.
```cpp
#include "libyt.h"
...
if (yt_run_JupyterKernel("LIBYT_STOP", true) != YT_SUCCESS) {
    fprintf(stderr, "ERROR: yt_run_JupyterKernel failed!\n");
    exit(EXIT_FAILURE);
}
```