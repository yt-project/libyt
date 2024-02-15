# libyt Defined Commands

> {octicon}`info;1em;sd-text-info;` There should be no spaces between `%` and `libyt`.

## General Commands

### `help`
```
>>> %libyt help
```
Print help messages.

### `exit`
```
>>> %libyt exit
```
Exit interactive mode, and continue the iterative process in simulation. 

> {octicon}`info;1em;sd-text-info;` `exit` only works in interactive Python prompt.

### `load`
```
>>> %libyt load <file name>
```
Load and run file in inline script's namespace. This runs statements in the file line by line in Python. We can overwrite and update Python functions and objects. Changes will maintain throughout in situ analysis.

All new functions detected in this new loaded script will be set to idle, and will not run in the following in situ analysis, unless you switch it on using [`%libyt run`](#run).

### `export`
```
>>> %libyt export <file name>
```
Export successfully run libyt defined commands and Python statement into file in current in situ analysis. History of interactive prompt will be cleared when leaving prompt.

> {octicon}`info;1em;sd-text-info;` Will overwrite file if `<file name>` already exist.

### `status`
```
>>> %libyt status
```
Print [status board](#status-board).

## Function Related Commands

### `status`
```
>>> %libyt status <function name>
```
Print function's definition and error messages if it has.

:::
###### Example
:::
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

### `idle`
```
>>> %libyt idle <function name>
```
Idle `<function name>` in next in situ analysis. You will see `X` at run column in status board. It will clear all the input arguments set through [`%libyt run`](#run).

:::
###### Example
:::
Idle `func` in next round.
```
>>> %libyt idle func
Function func set to idle ... done
```

### `run`
```
>>> %libyt run <function name> [args ...]
```
Run `<function name>` in the following in situ analysis using `[args ...]` if given. Set input arguments every time you switch this function on, because [`%libyt idle`](#idle) clears them.

> {octicon}`info;1em;sd-text-info;` Input arguments passed in through [`yt_run_FunctionArguments`]({% link libytAPI/PerformInlineAnalysis.md %}#yt_run_functionarguments) in simulation code have a bigger priority. When calling [`yt_run_InteractiveMode`]({% link libytAPI/ActivateInteractiveMode.md %}#yt_run_interactivemode), it runs all the functions that was set to run using [`%libyt run`](#run), but had not been done in simulation by calling [`yt_run_Function`]({% link libytAPI/PerformInlineAnalysis.md %}#yt_run_function) or [`yt_run_FunctionArguments`]({% link libytAPI/PerformInlineAnalysis.md %}#yt_run_functionarguments) yet.

> {octicon}`alert;1em;sd-text-danger;` When using triple quotes in input arguments, use either `"""` or `'''`, but not both of them at the same time. If you really need triple quotes, stick to either one of them. For example, `%libyt run func """b""" """c"""` is good, but `%libyt run func """b""" '''c'''` leads to error.

:::
###### Example
:::
This is equivalent of calling `func(a, 2, "3")` in Python in next round.
```
>>> a = 1
>>> def func(*args):
...     print(args)
>>> %libyt run func a 2 "3"
Function func set to run ... done
Run func(a,2,"3") in next iteration
```

## Status Board
The status board contains a list of all the Python functions `libyt` finds.
These are functions we can control whether to run in next round, and to access error message if it has.
```txt
=====================================================================
  Inline Function                              Status         Run
---------------------------------------------------------------------
  * yt_inline_ProjectionPlot                   success         V
  * yt_derived_field_demo                      idle            X
  * test_function                              failed          V
=====================================================================
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
