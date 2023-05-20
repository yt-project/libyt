---
layout: default
title: Inline Python Script
parent: In Situ Python Analysis
nav_order: 1
---
# Inline Python Script
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

## How libyt Loads Inline Python script and Runs Python Functions?
The inline Python script and the simulation executable should be placed in the same location.
`libyt` imports inline Python script only once at initialization stage when calling [`yt_initialize`]({% link libytAPI/Initialize.md %}#yt_initialize). 
Each MPI process runs the same Python script. The imported script will also serve as the namespace. 
All of our in situ Python analysis are done inside this namespace. 

The namespace contains function objects in the script. We can use [`yt_run_Function`]({% link libytAPI/PerformInlineAnalysis.md %}#yt_run_Functions) 
and [`yt_run_FunctionArguments`]({% link libytAPI/PerformInlineAnalysis.md %}#yt_run_FunctionArguments) to call them in simulation.

Every new added objects in this namespace will be kept and brought to next round of analysis. 
Objects created when running Python function get freed once they leave the function.
Make sure we put the global variable wisely.

## What Happens if the Python Function Crashed?
If `libyt` is compiled in [**normal mode**]({% link HowToInstall.md %}#options), it is not fault-tolerant to Python, 
so the whole simulation will shut down.

Use [**interactive mode**]({% link HowToInstall.md %}#options) if we want our in situ Python analysis to be fault-tolerant. 
In interactive mode, the simulation will pause when some Python functions crash, or it is called to pause 
(See more in [`yt_run_InteractiveMode`]({% link libytAPI/ActivateInteractiveMode.md %}#yt_run_InteractiveMode)).

## Can I Update Python Functions Defined in the Script?
We can only update Python functions in [**interactive mode**]({% link HowToInstall.md %}#options).
See [**Interactive Python Prompt**]({% link InSituPythonAnalysis/InteractivePythonPrompt.md %}#interactive-python-prompt) for how to update.

> :lizard: Currently, `libyt` interactive mode only works on local machine or submit the job to HPC platforms through interactive queue like `qsub -I`.
> The reason is that the user interface is through the terminal. We will work on it to support Jupyter Notebook Interface, so that it is applicable to HPC system.

