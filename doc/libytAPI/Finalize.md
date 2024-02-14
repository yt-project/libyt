---
layout: default
title: yt_finalize -- Finalize libyt
parent: libyt API
nav_order: 14
---
# Finalize
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

## yt_finalize
```cpp
int yt_finalize();
```
- Usage: Clean up embedded Python. Each rank must call this API when terminating our program.
- Return: `YT_SUCCESS` or `YT_FAIL`

## Example
```cpp
if ( yt_finalize() != YT_SUCCESS ){  
    fprintf( stderr, "ERROR: yt_finalize() failed!\n" );  
    exit( EXIT_FAILURE );  
}
```