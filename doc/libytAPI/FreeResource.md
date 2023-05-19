---
layout: default
title: yt_free -- Free resources
parent: libyt API
nav_order: 11
---
# Free Resources Allocated by libyt
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


## yt_free
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
