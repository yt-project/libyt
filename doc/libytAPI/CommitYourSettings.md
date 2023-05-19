---
layout: default
title: yt_commit -- Commit settings into Python
parent: libyt API
nav_order: 8
---
# Commit Your Settings
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