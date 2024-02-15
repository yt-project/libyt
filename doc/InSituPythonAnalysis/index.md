# In Situ Python Analysis

```{toctree}
:hidden:

InlinePythonScript
UsingYT
libytPythonModule
libytCommand
InteractivePythonPrompt
ReloadingScript
JupyterNotebookAccess/JupyterNotebook
Limitations
FAQs
```

## Using Python for In Situ Analysis in libyt

- [**Inline Python Script**]({% link InSituPythonAnalysis/InlinePythonScript.md %}#inline-python-script): how inline Python script works?
- [**Using yt**]({% link InSituPythonAnalysis/UsingYT.md %}#using-yt-for-in-situ-python-analysis): how to use `yt` for in situ analysis?
- [**libyt Python Module**]({% link InSituPythonAnalysis/libytPythonModule.md %}#libyt-python-module): the Python module where libyt API binds simulation data to.

## User Interface

- [**libyt Defined Commands**]({% link InSituPythonAnalysis/libytCommand.md %}#libyt-defined-commands): the commands allow users to load script, export history, set function to run or idle, etc.
- [**Interactive Python Prompt**]({% link InSituPythonAnalysis/InteractivePythonPrompt.md %}#interactive-python-prompt): the prompt can process Python statements and give feedback instantly.
  > :information_source: `libyt` should be compiled using [`-DINTERACTIVE_MODE=ON`]({% link HowToInstall.md %}#-dinteractive_modeonoff-defaultoff).
- [**Reloading Script**]({% link InSituPythonAnalysis/ReloadingScript.md %}#reloading-script): this allows reloading Python script.
  > :information_source: `libyt` should be compiled using [`-DINTERACTIVE_MODE=ON`]({% link HowToInstall.md %}#-dinteractive_modeonoff-defaultoff).
- [**Jupyter Notebook Access**]({% link InSituPythonAnalysis/JupyterNotebookAccess/JupyterNotebook.md %}#jupyter-notebook-access): this allows accessing and analyzing simulation data through JupyterLab/Jupyter Notebook.
  > :information_source: `libyt` should be compiled in [`-DJUPYTER_KERNEL=ON`]({% link HowToInstall.md %}#-djupyter_kernelonoff-defaultoff).

## Limitations

- [**Limitations in MPI Related Python Tasks**]({% link InSituPythonAnalysis/Limitations.md %}#limitations-in-mpi-related-python-tasks)

## FAQs

- [**How Does libyt Run Python Script?**]({% link InSituPythonAnalysis/FAQs.md %}#how-does-libyt-run-python-script)
