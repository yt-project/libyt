# Commit Your Settings
## yt\_commit\_grids
```cpp
int yt_commit_grids();
```
- Usage: Tell `libyt` you are done filling in all the informations. Every rank must call this function.
- Return: `YT_SUCCESS` or `YT_FAIL`

## Example
```cpp
if ( yt_commit_grids() != YT_SUCCESS ) {
    fprintf( stderr, "ERROR: yt_commit_grids() failed!\n" );  
    exit( EXIT_FAILURE );  
}
```