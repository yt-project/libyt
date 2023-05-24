---
layout: default
title: Interactive Python Prompt
parent: In Situ Python Analysis
nav_order: 3
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

> :information_source: We need `libyt` to be in [**interactive mode**]({% link HowToInstall.md %}#options).

> :warning: Cannot use arrow keys or `Crtl+D` in interactive mode. 

## Status Board
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
  * yt_derived_field_demo                      success         V
  * test_function                              failed          V
=====================================================================
>>> 
```

- **Inline Function**: whenever we load inline script at [`yt_initialize`]({% link libytAPI/Initialize.md %}#yt_initialize), use [`%libyt load`](#load), or directly type in [Python prompt](#python-prompt), `libyt` will detect callables and list them here.
- **Status**: function status.
  - `success`: run successfully.
  - `failed`: failed.
  - `idle`: this function was set to idle in current step.
  - `not run yet`: not execute by `libyt` yet.
- **Run**: whether the function will run automatically in next round or not.
  - `V`: this function will run automatically in the following in situ analysis.
  - `X`: this function will idle in next in situ analysis, even if it is called through [`yt_run_FunctionArguments`]({% link libytAPI/PerformInlineAnalysis.md %}#yt_run_functionarguments) or [`yt_run_Function`]({% link libytAPI/PerformInlineAnalysis.md %}#yt_run_function) in simulation.

## Python Prompt
```
>>> 
```
MPI root process is the user interface, which takes user inputs and broadcasts to other ranks.
So every MPI process execute the same piece of input at the same time.
After that, they print feedbacks to your terminal.

We can run any Python statements or [**libyt commands**](#libyt-defined-commands) here, including importing modules. These statements run in inline script's namespace, and will update and make changes to it. This is like using normal Python prompt, except that there are functions, objects, and simulation data already defined in it. 

> :warning: Changes will maintain throughout every in situ analysis.

> :information_source: To make interactive prompt more smoothly, set lower [YT_VERBOSE]({% link libytAPI/Initialize.md %}#yt_param_libyt).

> :lizard: Currently, `libyt` interactive mode only works on local machine or submit the job to HPC platforms through interactive queue like `qsub -I`.
> The reason is that the user interface is directly through the terminal. We will work on it to support Jupyter Notebook Interface, so that it is more flexible to HPC system.

## libyt Defined Commands
> :information_source: There should be no spaces between `%` and `libyt`.

### General Commands
#### help
```
>>> %libyt help
```
Print help messages.

#### exit
```
>>> %libyt exit
```
Exit interactive mode, and continue the iterative process in simulation.

#### load
```
>>> %libyt load <file name>
```
Load and run file in inline script's namespace. This is like running statements in the file line by line in Python. We can overwrite and update Python functions and objects. Changes will maintain throughout in situ analysis.

All new functions detected in this new loaded script will be set to idle, and will not run in the following in situ analysis, unless you switch it on using [`%libyt run`](#run).

#### export
```
>>> %libyt export <file name>
```
Export successfully run libyt defined commands and Python statement into file in current in situ analysis. History of interactive prompt will be cleared when leaving prompt. 

> :information_source: Will overwrite file if `<file name>` already exist. 

#### status
```
>>> %libyt status
```
Print status board.

### Function Related Commands

#### status

```
>>> %libyt status <function name>
```
Print function's definition and error messages if it has.

###### Example
{: .no_toc }
Print the status of the function, including its definition and error message in each rank if it has.
```
>>> %libyt status func
func ... failed
[Function Def]
  def func(a):
     print(b)
  
[Error Msg]
  [ MPI 0 ]
    Traceback (most recent call last):
      File "<string>", line 2, in <module>
      File "<string>", line 1, in <module>
      File "<libyt-stdin>", line 2, in func
    NameError: name 'b' is not defined
  [ MPI 1 ] 
    ...
```

#### idle
```
>>> %libyt idle <function name>
```
Idle `<function name>` in next in situ analysis. You will see `X` at run column in status board. It will clear all the input arguments set through [`%libyt run`](#run).

###### Example
{: .no_toc }
Idle `func` in next round.
```
>>> %libyt idle func
Function func set to idle ... done
```

#### run
```
>>> %libyt run <function name> [args ...]
```
Run `<function name>` in the following in situ analysis using `[args ...]` if given. Set input arguments every time you switch this function on, because [`%libyt idle`](#idle) clears them.

> :information_source: Input arguments passed in through [`yt_run_FunctionArguments`](../libytAPI/PerformInlineAnalysis.md#yt_run_functionarguments) in simulation code have a bigger priority. When calling [`yt_run_InteractiveMode`](../libytAPI/ActivateInteractiveMode.md#yt_run_interactivemode), it runs all the functions that was set to run using [`%libyt run`](#run-1), but had not been done in simulation by calling [`yt_run_Function`](../libytAPI/PerformInlineAnalysis.md#yt_run_function) or [`yt_run_FunctionArguments`](../libytAPI/PerformInlineAnalysis.md#yt_run_functionarguments) yet.

> :warning: When using triple quotes in input arguments, use either `"""` or `'''`, but not both of them at the same time. If you really need triple quotes, stick to either one of them. For example, `%libyt run func """b""" """c"""` is good, but `%libyt run func """b""" '''c'''` leads to error.

###### Example
{: .no_toc }
This is equivalent of calling `func(a, 2, "3")` in Python in next round.
```
>>> a = 1
>>> def func(*args):
...     print(args)
>>> %libyt run func a 2 "3"
Function func set to run ... done
Run func(a,2,"3") in next iteration
```

## FAQs
### Why Can't I Find the Prompt `>>> ` After I have Activated Interactive Mode?
`>>> `  is probably immersed inside the output. 
We can hit enter again, which is to provide an empty statement, and it will come out. 

We can make prompt more smoothly by setting [YT_VERBOSE]({% link libytAPI/Initialize.md %}#yt_param_libyt) to YT_VERBOSE_INFO.

### Where Can I Use Interactive Mode?
Currently, `libyt` interactive mode only works on local machine or submit the job to HPC platforms through interactive queue like `qsub -I`. 
The reason is that the user interface is directly through the terminal. We will work on it to support Jupyter Notebook Interface, so that it is more flexible to HPC system.
