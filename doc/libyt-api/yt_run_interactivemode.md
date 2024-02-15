# `yt_run_InteractiveMode` -- Activate Interactive Python Prompt

## `yt_run_InteractiveMode`
```cpp
int yt_run_InteractiveMode(const char* flag_file_name);
```
- Usage: Activate [interactive Python prompt]({% link InSituPythonAnalysis/InteractivePythonPrompt.md %}#interactive-python-prompt) when file `flag_file_name` is detected in the same directory where simulation executable is.
- Return: 
  - `YT_SUCCESS`
  - `YT_FAIL`: When `libyt` is not compiled with `-DINTERACTIVE_MODE`, it returns `YT_FAIL`.

> {octicon}`info;1em;sd-text-info;` Must compile `libyt` with [`-DINTERACTIVE_MODE`]({% link HowToInstall.md %}#dinteractive-mode).

## Example
```cpp
#include "libyt.h"
...
if (yt_run_InteractiveMode("LIBYT_STOP") != YT_SUCCESS) {
    fprintf(stderr, "ERROR: yt_run_InteractiveMode failed!\n");
    exit(EXIT_FAILURE);
}
```
