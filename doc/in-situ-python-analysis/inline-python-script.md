# Inline Python Script

## How libyt Loads Inline Python script and Runs Python Functions?
The inline Python script and the simulation executable should be placed in the same location.
`libyt` imports inline Python script only once at initialization stage when calling [`yt_initialize`]({% link libytAPI/Initialize.md %}#yt_initialize). 

Each MPI process runs the same Python script. The imported script will also serve as the namespace. 
All of our in situ Python analysis are done inside this namespace. 

The namespace contains function objects in the script. We can use [`yt_run_Function`]({% link libytAPI/PerformInlineAnalysis.md %}#yt_run_function) 
and [`yt_run_FunctionArguments`]({% link libytAPI/PerformInlineAnalysis.md %}#yt_run_functionarguments) to call them during simulation process.

## What Happens if the Python Function Crashed?
If `libyt` is compiled in [**normal mode**]({% link HowToInstall.md %}#options), it is not fault-tolerant to Python, 
so the whole simulation will shut down.

Use **interactive mode** ([`-DINTERACTIVE_MODE=ON`]({% link HowToInstall.md %}#-dinteractive_modeonoff-defaultoff)) or **jupyter kernel mode** ([`-DJUPYTER_KERNEL=ON`]({% link HowToInstall.md %}#-djupyter_kernelonoff-defaultoff)) if we want our in situ Python analysis to be fault-tolerant.

## Can I Update Python Functions Defined in the Script?
We can only update Python functions in **interactive mode** ([`-DINTERACTIVE_MODE=ON`]({% link HowToInstall.md %}#-dinteractive_modeonoff-defaultoff)) or **jupyter kernel mode** ([`-DJUPYTER_KERNEL=ON`]({% link HowToInstall.md %}#-djupyter_kernelonoff-defaultoff)).

Since every new added Python object is maintained inside inline Python script's namespace, you can update a Python function by re-define the function again, so that the old function is overwritten.
(See [Interactive Python Prompt]({% link InSituPythonAnalysis/InteractivePythonPrompt.md %}#interactive-python-prompt), [Reloading Script]({% link InSituPythonAnalysis/ReloadingScript.md %}#reloading-script), and [Jupyter Notebook Access]({% link InSituPythonAnalysis/JupyterNotebookAccess/JupyterNotebook.md %}#jupyter-notebook-access).)
