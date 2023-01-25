# Perform Inline-Analysis
## yt\_inline
```cpp
int yt_inline( char *function_name );
```
- Usage: Call Python function `function_name` defined inside your Python script. You may call as many times as you want, as long as the functions are defined in your input python script.
- Return: `YT_SUCCESS` or `YT_FAIL`

## yt\_inline\_argument
```cpp
int yt_inline_argument( char *function_name, int argc, ... );
```
- Usage: Call Python function `function_name` with input arguments in your Python script. This API will pass total number of `argc` arguments. Please wrap your arguments as strings. For example, `"0"` for `0`, and `"\'FieldName\'"` for `'FieldName'`.
- Return: `YT_SUCCESS` or `YT_FAIL`

## Example
Inline Python script:
```python
import yt
yt.enable_parallelism()

def yt_inline_ProfilePlot():  
    ds = yt.frontends.libyt.libytDataset()  
    profile = yt.ProfilePlot(ds, "x", ["density"])  
    if yt.is_root():  
        profile.save()

def yt_inline_ProjectionPlot( fields ):  
    ds = yt.frontends.libyt.libytDataset()
    prjz = yt.ProjectionPlot(ds, 'z', fields)  
    if yt.is_root():
        prjz.save()
```

Call the function inside simulation code:
```cpp
/* libyt API, call the Python function. */
if ( yt_inline( "yt_inline_ProfilePlot" ) != YT_SUCCESS ){  
    fprintf( stderr, "ERROR: yt_inline() failed!\n" );
    exit( EXIT_FAILURE );  
}

/* libyt API, call the Python function with input arguments. */
if ( yt_inline_argument( "yt_inline_ProjectionPlot", 1, "\'density\'" ) != YT_SUCCESS ){
    fprintf( stderr, "ERROR: yt_inline_argument() failed!\n" );  
    exit( EXIT_FAILURE );  
}
```
