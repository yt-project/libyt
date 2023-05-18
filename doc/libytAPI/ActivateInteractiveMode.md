# Activate Interactive Mode

## yt\_run\_InteractiveMode
```cpp
int yt_run_InteractiveMode(const char* flag_file_name);
```
> :information_source: Must compile `libyt` with -DINTERACTIVE_MODE
- Usage: It will first run other inline python functions that are set to [run](../InSituPythonAnalysis/InteractivePythonPrompt.md#run-1) but haven't called by [`yt_run_Function`](PerformInlineAnalysis.md#yt_run_function) or [`yt_run_FunctionArguments`](PerformInlineAnalysis.md#yt_run_functionarguments) yet. Then it activates [interactive python prompt](../InSituPythonAnalysis/InteractivePythonPrompt.md#interactive-python-prompt) when there are errors occurred or file `flag_file_name` is detected.
- Return: `YT_SUCCESS` or `YT_FAIL`

## Example
```cpp
#include "libyt.h"
...
if (yt_run_InteractiveMode("LIBYT_STOP") != YT_SUCCESS) {
    fprintf(stderr, "ERROR: yt_run_InteractiveMode failed!\n");
    exit(EXIT_FAILURE);
}
```