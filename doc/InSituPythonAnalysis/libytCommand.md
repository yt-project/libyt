---
layout: default
title: libyt Defined Commands
parent: In Situ Python Analysis
nav_order: 4
---
# libyt Defined Commands
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

> :information_source: There should be no spaces between `%` and `libyt`.

## General Commands
### help
```
>>> %libyt help
```
Print help messages.

### exit
```
>>> %libyt exit
```
Exit interactive mode, and continue the iterative process in simulation. 

> :information_source: `exit` only works in interactive Python prompt.

### load
```
>>> %libyt load <file name>
```
Load and run file in inline script's namespace. This runs statements in the file line by line in Python. We can overwrite and update Python functions and objects. Changes will maintain throughout in situ analysis.

All new functions detected in this new loaded script will be set to idle, and will not run in the following in situ analysis, unless you switch it on using [`%libyt run`](#run).

### export
```
>>> %libyt export <file name>
```
Export successfully run libyt defined commands and Python statement into file in current in situ analysis. History of interactive prompt will be cleared when leaving prompt.

> :information_source: Will overwrite file if `<file name>` already exist.

### status
```
>>> %libyt status
```
Print status board.

## Function Related Commands

### status

```
>>> %libyt status <function name>
```
Print function's definition and error messages if it has.

##### Example
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

### idle
```
>>> %libyt idle <function name>
```
Idle `<function name>` in next in situ analysis. You will see `X` at run column in status board. It will clear all the input arguments set through [`%libyt run`](#run).

##### Example
{: .no_toc }
Idle `func` in next round.
```
>>> %libyt idle func
Function func set to idle ... done
```

### run
```
>>> %libyt run <function name> [args ...]
```
Run `<function name>` in the following in situ analysis using `[args ...]` if given. Set input arguments every time you switch this function on, because [`%libyt idle`](#idle) clears them.

> :information_source: Input arguments passed in through [`yt_run_FunctionArguments`]({% link libytAPI/PerformInlineAnalysis.md %}#yt_run_functionarguments) in simulation code have a bigger priority. When calling [`yt_run_InteractiveMode`]({% link libytAPI/ActivateInteractiveMode.md %}#yt_run_interactivemode), it runs all the functions that was set to run using [`%libyt run`](#run), but had not been done in simulation by calling [`yt_run_Function`]({% link libytAPI/PerformInlineAnalysis.md %}#yt_run_function) or [`yt_run_FunctionArguments`]({% link libytAPI/PerformInlineAnalysis.md %}#yt_run_functionarguments) yet.

> :warning: When using triple quotes in input arguments, use either `"""` or `'''`, but not both of them at the same time. If you really need triple quotes, stick to either one of them. For example, `%libyt run func """b""" """c"""` is good, but `%libyt run func """b""" '''c'''` leads to error.

##### Example
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
