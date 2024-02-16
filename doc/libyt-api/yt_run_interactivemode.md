# `yt_run_InteractiveMode` -- Activate Interactive Python Prompt

## `yt_run_InteractiveMode`
```cpp
int yt_run_InteractiveMode(const char* flag_file_name);
```
- Usage: Activate [Interactive Python Prompt](../in-situ-python-analysis/interactive-python-prompt.md#interactive-python-prompt) when file `flag_file_name` is detected.
- Return: 
  - `YT_SUCCESS`
  - `YT_FAIL`: When `libyt` is not compiled with `-DINTERACTIVE_MODE`, it returns `YT_FAIL`.

> {octicon}`info;1em;sd-text-info;` Must compile `libyt` with [`-DINTERACTIVE_MODE`](../how-to-install.md#dinteractive-mode).

## Example
```cpp
#include "libyt.h"
...
if (yt_run_InteractiveMode("LIBYT_STOP") != YT_SUCCESS) {
    fprintf(stderr, "ERROR: yt_run_InteractiveMode failed!\n");
    exit(EXIT_FAILURE);
}
```
