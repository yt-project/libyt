---
layout: default
title: Interactive Python Prompt
parent: In Situ Python Analysis
nav_order: 5
---
# Interactive Python Prompt
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

## Requirements

- Compile `libyt` in [**interactive mode**]({% link HowToInstall.md %}#options).

## Using Interactive Python Prompt

### Status Board
At the start of interactive mode, `libyt` prints the status board.
Interactive Python prompt will list all the Python function it finds, either in imported script, or input from interactive Python prompt.
These are functions we can control whether to run in next round, and to access error message if it has.
```
=====================================================================
  Inline Function                              Status         Run
---------------------------------------------------------------------
  * yt_inline_ProjectionPlot                   success         V
  * yt_inline_ProfilePlot                      success         V
  * yt_inline_ParticlePlot                     success         V
  * yt_derived_field_demo                      idle            V
  * test_function                              failed          V
=====================================================================
>>> 
```

- **Inline Function**: the inline function found by `libyt`.
- **Status**: function status.
  - `success`: successfully run the function.
  - `failed`: failed to run the function.
  - `idle`: the function was set to idle, so it was ignored and did nothing.
  - `not run yet`: the function hasn't been run yet.
- **Run**: whether the function will run automatically in next round.
  - `V`: this function will run automatically in the following in situ analysis.
  - `X`: this function will idle in next in situ analysis, even if it is called through [`yt_run_FunctionArguments`]({% link libytAPI/PerformInlineAnalysis.md %}#yt_run_functionarguments) or [`yt_run_Function`]({% link libytAPI/PerformInlineAnalysis.md %}#yt_run_function) in simulation.

### Python Prompt and libyt Defined Commands
```
>>> 
```
It can run any Python statements or [**libyt defined commands**]({% link InSituPythonAnalysis/libytCommand.md %}#libyt-defined-commands) here, including importing Python modules.
This is like using normal Python prompt, except that there are functions, objects, and simulation data already defined in it.
These statements run in inline script's namespace, and will update and make changes to it. 

> :information_source: To make interactive prompt more smoothly, set lower [YT_VERBOSE]({% link libytAPI/Initialize.md %}#yt_param_libyt).

> :warning: Changes is kept and maintain in user's inline script's namespace in situ analysis in the following round.

##### How does the prompt execute Python statements and libyt defined commands?
{: .no_toc }
1. MPI root process is the user interface, which takes user inputs.
2. MPI root process broadcasts the inputs to other processes.
3. Every other MPI process executes the same piece of input synchronously.
4. Print feedbacks to the terminal.

## Known Limitations
- Cannot use arrow keys or `Crtl+D` in interactive mode.

## FAQs
### Why Can't I Find the Prompt `>>>` After I have Activated Interactive Mode?
`>>> `  is probably immersed inside the output. 
We can hit enter again, which is to provide an empty statement, and it will come out. 

We can make prompt more smoothly by setting [YT_VERBOSE]({% link libytAPI/Initialize.md %}#yt_param_libyt) to YT_VERBOSE_INFO.

### Where Can I Use Interactive Mode?
`libyt` interactive mode only works on local machine or submit the job to HPC platforms through interactive queue like `qsub -I`. 
The reason is that the user interface is exposed in the terminal.