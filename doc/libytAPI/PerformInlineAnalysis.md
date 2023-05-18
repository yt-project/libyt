# Perform Inline-Analysis

## yt\_run\_Function
```cpp
int yt_run_Function( const char *function_name );
```
- Usage: Run Python function `function_name`. Can call as many times as we want.
- Return: `YT_SUCCESS` or `YT_FAIL`

## yt\_run\_FunctionArguments
```cpp
int yt_run_FunctionArguments( const char *function_name, int argc, ... );
```
- Usage: Run Python function `function_name` with input arguments. This API will pass total number of `argc` arguments. Please wrap your arguments as strings. For example, `"0"` for `0`, `"\'FieldName\'"` for `'FieldName'`, `"a"` for defined python variable `a` within namespace.
> :warning: When using triple quotes in input arguments, use either `"""` or `'''`, but not both of them at the same time. If you really need triple quotes, stick to either one of them. For example, `yt_run_FunctionArguments("func", 2, """b""", """c""")` is good, but `yt_run_FunctionArguments("func", 2, """b""", '''c''')` is bad and leads to error.

> :information_source: These two API run functions inside script's namespace, which means we can pass in objects defined in script directly.
- Return: `YT_SUCCESS` or `YT_FAIL`

## Example
Inline Python script:
```python
import yt_libyt
import yt
yt.enable_parallelism()

a = "var"

def yt_inline_ProfilePlot():  
    ds = yt_libyt.libytDataset()  
    profile = yt.ProfilePlot(ds, "x", ["density"])  
    if yt.is_root():  
        profile.save()

def yt_inline_ProjectionPlot( fields, *args ):  
    ds = yt_libyt.libytDataset()
    prjz = yt.ProjectionPlot(ds, 'z', fields)  
    if yt.is_root():
        prjz.save()
```

Call the function inside simulation code:
```cpp
/* libyt API: run yt_inline_ProfilePlot(). */
if ( yt_run_Function( "yt_inline_ProfilePlot" ) != YT_SUCCESS ){  
    fprintf( stderr, "ERROR: yt_run_Function() failed!\n" );
    exit( EXIT_FAILURE );  
}

/* libyt API: run yt_inline_ProjectionPlot('density', a, 1). */
if ( yt_run_FunctionArguments( "yt_inline_ProjectionPlot", 2, "\'density\'", "a", "1" ) != YT_SUCCESS ){
    fprintf( stderr, "ERROR: yt_run_FunctionArguments() failed!\n" );  
    exit( EXIT_FAILURE );  
}
```
