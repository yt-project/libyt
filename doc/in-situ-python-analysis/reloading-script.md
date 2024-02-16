# Reloading Script

## Requirements

- Compile `libyt` in **interactive mode** ([`-DINTERACTIVE_MODE=ON`](../how-to-install.md#dinteractive-mode)).
- Call libyt API [`yt_run_ReloadScript`](../libyt-api/yt_run_reloadscript.md#yt-run-reloadscript). 

## Reloading Script
Reloading script feature is a file-based [Interactive Python Prompt](./interactive-python-prompt.md#interactive-python-prompt), such that user creates specific file to send instructions to `libyt` and gets output from specific file.

### How to Reload and Exit
**The API**:
```c++
int yt_run_ReloadScript(const char* flag_file, const char* reload, const char* script);
```
**Below is the workflow**:

![](../_static/svg/ReloadingScript.svg)

**A concrete example**:
```c++
if (yt_run_ReloadScript("LIBYT_STOP", "RELOAD", "test_reload.py") != YT_SUCCESS) {
    fprintf(stderr, "ERROR: yt_run_ReloadScript failed!\n");
    exit(EXIT_FAILURE);
}
```
1. The code enters reloading script phase when it finds `LIBYT_STOP` file or an error occurred in inline functions. 
2. `libyt` starts reloading the script `test_reload.py` once `RELOAD` file is detected. 
3. If it successfully runs the script, the output will be inside `RELOAD_SUCCESS` file. If it failed to run the script, the output and error messages will be inside `RELOAD_FAILED` file. Either `RELOAD_SUCCESS` or `RELOAD_FAILED` will be generated. 
4. To exit, create `RELOAD_EXIT` in the folder.

### Python Statements and libyt Defined Commands

The script can have both Python statements and [libyt Defined Commands](./libyt-defined-command.md#libyt-defined-commands). We need to separate Python statements and libyt commands using `#LIBYT_COMMANDS`. libyt commands should be after `#LIBYT_COMMANDS` tag and commented out libyt defined commands.

**For example**:
```python
import numpy as np

def func():
    print("HELLO WORLD")

#LIBYT_COMMANDS
# %libyt status
# %libyt run func
```

## Known Limitations
- See [Limitations in MPI Related Python Tasks](./limitation.md#limitations-in-mpi-related-python-tasks).

## FAQs

### When Can I Reload Script?
`libyt` supports reloading script feature if it is compiled with [`-DINTERACTIVE_MODE=ON`](../how-to-install.md#dinteractive-mode).
The root process reads the file, so it would work on your local desktop and in HPC clusters.
