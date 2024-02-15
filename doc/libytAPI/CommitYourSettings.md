# `yt_commit` -- Commit Your Settings

## yt_commit
```cpp
int yt_commit();
```
- Usage: Tell `libyt` you are done filling in all the information. Every rank must call this function.
- Return: `YT_SUCCESS` or `YT_FAIL`

## Example
```cpp
if ( yt_commit() != YT_SUCCESS ) {
    fprintf( stderr, "ERROR: yt_commit() failed!\n" );  
    exit( EXIT_FAILURE );  
}
```
