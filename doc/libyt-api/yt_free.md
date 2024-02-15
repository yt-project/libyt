# `yt_free` -- Free libyt Resource

## `yt_free`
```cpp
int yt_free();
```
- Usage: Free resource allocated by `libyt`. We should always remember to call this after in situ analysis. Otherwise, we will get memory leakage.
- Return: `YT_SUCCESS` or `YT_FAIL`

## Example
```cpp
if ( yt_free() != YT_SUCCESS ){  
    fprintf( stderr, "ERROR: yt_free() failed!\n" );  
    exit( EXIT_FAILURE );  
}
```
