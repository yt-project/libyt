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

> {octicon}`info;1em;sd-text-info;` `exit` only works in [Interactive Python Prompt](./interactive-python-prompt.md#interactive-python-prompt).

### `load`
```
>>> %libyt load <file name>
```
Load and run file in inline script's namespace. This runs statements in the file line by line in Python. We can overwrite and update Python functions and objects. Changes will maintain throughout in situ analysis.

All new functions detected in this new loaded script will be set to idle, unless you switch it on using [`%libyt run`](#run).

### `export`
```
>>> %libyt export <file name>
```
Export successfully executed libyt defined commands and Python statements into file `<file name>`. File will be overwritten if it already exists. History will be cleared when leaving the analysis.

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
Print current function's definition and error occurred when running inline functions.

> {octicon}`info;1em;sd-text-info;` Since we can update the function, the function definition shown will always be the newly updated one. The error messages are from the last Python function call by [`yt_run_Function`](../libyt-api/run-python-function.md#yt_run_function)/[`yt_run_FunctionArguments`](../libyt-api/run-python-function.md#yt_run_functionarguments).

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
Idle `<function name>` in next in situ analysis. You will see `X` at run column in [status board](#status-board). It will clear all the input arguments set through [`%libyt run`](#run).

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
Run `<function name>` in the following in situ analysis using `[args ...]` if given.

> {octicon}`info;1em;sd-text-info;` Input arguments passed in through [`yt_run_FunctionArguments`](../libyt-api/run-python-function.md#yt_run_functionarguments) in simulation code have a bigger priority.

> {octicon}`alert;1em;sd-text-danger;` When using triple quotes in input arguments, use either `"""` or `'''`, but not both.. If you really need triple quotes, stick to either one of them. For example, `%libyt run func """b""" """c"""` is good, but `%libyt run func """b""" '''c'''` leads to error.

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
These are functions we can control whether to run in next round, and to access error message from the last Python function call by [`yt_run_Function`](../libyt-api/run-python-function.md#yt_run_function)/[`yt_run_FunctionArguments`](../libyt-api/run-python-function.md#yt_run_functionarguments).
```text
=====================================================================
  Inline Function                              Status         Run
---------------------------------------------------------------------
  * yt_inline_ProjectionPlot                   success         V
  * yt_derived_field_demo                      idle            X
  * test_function                              failed          V
=====================================================================
```
- **Inline Function**: the inline function found by `libyt`.
- **Status**: function status in latest Python function call by [`yt_run_Function`](../libyt-api/run-python-function.md#yt_run_function)/[`yt_run_FunctionArguments`](../libyt-api/run-python-function.md#yt_run_functionarguments).
  - `success`: successfully run the function.
  - `failed`: failed to run the function.
  - `idle`: the function was set to idle, so it was ignored and did nothing.
  - `not run yet`: the function hasn't been run yet.
- **Run**: whether the function will run automatically in next round.
  - `V`: this function will run automatically in the following in situ analysis.
  - `X`: this function will idle in next in situ analysis, even if it is called by [`yt_run_Function`](../libyt-api/run-python-function.md#yt_run_function)/[`yt_run_FunctionArguments`](../libyt-api/run-python-function.md#yt_run_functionarguments) in simulation.
