# `yt_run_JupyterKernel` -- Activate Jupyter Kernel

## `yt_run_JupyterKernel`
```cpp
int yt_run_JupyterKernel(const char* flag_file_name, bool use_connection_file);
```
- Usage: Activate Jupyter kernel if file `flag_file_name` is detected. `libyt` uses user provided connection file if `use_connection_file` is `true`. If `false`, it will pick and bind to ports and generate the connection file itself. The connection file name must be `libyt_kernel_connection.json`.
- Return:
    - `YT_SUCCESS`
    - `YT_FAIL`: When `libyt` is not compiled with `-DJUPYTER_KERNEL=ON`, it returns `YT_FAIL`.

> {octicon}`info;1em;sd-text-info;` Must compile `libyt` with [`-DJUPYTER_KERNEL=ON`]({% link HowToInstall.md %}#-djupyter_kernelonoff-defaultoff).

> {octicon}`info;1em;sd-text-info;` The API only launches a Jupyter kernel that exposes simulation data. For how to connect to the kernel, please go to [Jupyter Notebook Access]({% link InSituPythonAnalysis/JupyterNotebookAccess/JupyterNotebook.md %}#jupyter-notebook-access). 

## Connection File
The connection file name must be `libyt_kernel_connection.json`. This is the same connection file used in Jupyter but with different file name.
(Please refer to [Jupyter Client -- Connection files](https://jupyter-client.readthedocs.io/en/stable/kernels.html#connection-files))
`libyt` use this connection file to bind to ports and construct kernel.

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
