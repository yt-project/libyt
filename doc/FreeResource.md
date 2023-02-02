# Free Resource
## yt\_free\_gridsPtr
```cpp
int yt_free();
```
- Usage: Free resource allocated by `libyt` in this round of inline-analysis. After you have done calling Python script, you should call this API in each MPI rank.
- Return: `YT_SUCCESS` or `YT_FAIL`

## Example
```cpp
if ( yt_free() != YT_SUCCESS ){  
    fprintf( stderr, "ERROR: yt_free() failed!\n" );  
    exit( EXIT_FAILURE );  
}
```
