# `yt_run_Function`, `yt_run_FunctionArguments` -- Call Python Function

## `yt_run_Function`
```cpp
int yt_run_Function( const char *function_name );
```
- Usage: Run Python function `function_name`. This is equivalent to run `function_name()` in Python.
- Return: `YT_SUCCESS` or `YT_FAIL`

## `yt_run_FunctionArguments`
```cpp
int yt_run_FunctionArguments( const char *function_name, int argc, ... );
```
- Usage: Run Python function `function_name` with input arguments. This API will pass total number of `argc` arguments. Please wrap your arguments as strings. For example, `"0"` for `0`, `"\'FieldName\'"` for `'FieldName'`, `"a"` for a defined Python variable `a` within namespace.
- Return: `YT_SUCCESS` or `YT_FAIL`
> {octicon}`alert;1em;sd-text-danger;` When using triple quotes in input arguments, use either `"""` or `'''`, but not both of them at the same time. If you really need triple quotes, stick to either one of them. For example, `yt_run_FunctionArguments("func", 2, """b""", """c""")` is good, but `yt_run_FunctionArguments("func", 2, """b""", '''c''')` is bad and leads to error.

> {octicon}`info;1em;sd-text-info;` These two API run functions inside script's namespace, which means we can pass in objects defined in script directly.

## Example
If our inline script is this:

```python
from mpi4py import MPI

myrank = MPI.COMM_WORLD.Get_rank()
a = "var"

def func():  
    print(myrank, ": ", "Inside func()")

def funcArgs( *args ):
    print(myrank, ": ", "Inside funcArgs(", *args, ")")  # --> print funcArgs(density var 1)
```

We can call Python function through `libyt` API in simulation code:

```cpp
/* libyt API: run func() in Python. */
if ( yt_run_Function( "func" ) != YT_SUCCESS ){  
    fprintf( stderr, "ERROR: func() failed!\n" );
    exit( EXIT_FAILURE );  
}

/* libyt API: run funcArgs('density', a, 1) in Python. */
if ( yt_run_FunctionArguments( "funcArgs", 3, "\'density\'", "a", "1" ) != YT_SUCCESS ){
    fprintf( stderr, "ERROR: funcArgs() failed!\n" );  
    exit( EXIT_FAILURE );  
}
```
