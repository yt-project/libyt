# Inline Python Script

## How libyt Loads Inline Python script and Runs Python Functions?
The inline Python script and the simulation executable should be placed in the same location.
`libyt` imports inline Python script only once at initialization stage when calling [`yt_initialize`](../libyt-api/yt_initialize.md#yt_initialize). 

Each MPI process runs the same Python script. The imported script will also serve as the namespace. 
All of our in situ Python analysis are done inside this namespace. 

The namespace contains function objects in the script. We can use [`yt_run_Function`](../libyt-api/run-python-function.md#yt_run_function) 
and [`yt_run_FunctionArguments`](../libyt-api/run-python-function.md#yt_run_functionarguments) to call them during simulation process.

## What Happens if the Python Function Crashed During the Analysis?
If `libyt` is compiled in **normal mode** ([`-DINTERACTIVE_MODE=OFF`](../how-to-install.md#-dinteractive_mode)), it is not fault-tolerant to Python, 
so the whole simulation will shut down.

Use **interactive mode** ([`-DINTERACTIVE_MODE=ON`](../how-to-install.md#-dinteractive_mode)) or **jupyter kernel mode** ([`-DJUPYTER_KERNEL=ON`](../how-to-install.md#-djupyter_kernel)) if we want our in situ Python analysis to be fault-tolerant.

## Can I Update Python Functions Defined in the Script?
We can only update Python functions in **interactive mode** ([`-DINTERACTIVE_MODE=ON`](../how-to-install.md#-dinteractive_mode)) or **jupyter kernel mode** ([`-DJUPYTER_KERNEL=ON`](../how-to-install.md#-djupyter_kernel)).

Since every new added Python object is maintained inside inline Python script's namespace, you can update a Python function by re-define the function again, so that the old function is overwritten.

We can update via one of these methods: [Interactive Python Prompt](./interactive-python-prompt.md#interactive-python-prompt), [Reloading Script](./reloading-script.md#reloading-script), and [Jupyter Notebook Access](./jupyter-notebook/jupyter-notebook-access.md#jupyter-notebook-access).
