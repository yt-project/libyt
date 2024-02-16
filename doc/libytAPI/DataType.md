---
layout: default
title: yt Data Type
parent: libyt API
nav_order: 15
---
# yt Data Type
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

## yt_dtype
- Usage: `libyt` defined macros for C data type. They are matched to corresponding NumPy data type and MPI data type.

|   yt_dtype    |    C Data Type     | NumPy Data Type |     MPI Data Type      |
|:-------------:|:------------------:|:---------------:|:----------------------:|
|   YT_FLOAT    |       float        |    NPY_FLOAT    |       MPI_FLOAT        |
|   YT_DOUBLE   |       double       |   NPY_DOUBLE    |       MPI_DOUBLE       |
| YT_LONGDOUBLE |    long double     | NPY_LONGDOUBLE  |    MPI_LONG_DOUBLE     |
|    YT_CHAR    |        char        |    NPY_BYTE     |    MPI_SIGNED_CHAR     |
|   YT_UCHAR    |   unsigned char    |    NPY_UBYTE    |   MPI_UNSIGNED_CHAR    |
|   YT_SHORT    |       short        |    NPY_SHORT    |       MPI_SHORT        |
|   YT_USHORT   |   unsigned short   |   NPY_USHORT    |   MPI_UNSIGNED_SHORT   |
|    YT_INT     |        int         |     NPY_INT     |        MPI_INT         |
|    YT_UINT    |    unsigned int    |    NPY_UINT     |      MPI_UNSIGNED      |
|    YT_LONG    |        long        |    NPY_LONG     |        MPI_LONG        |
|   YT_ULONG    |   unsigned long    |    NPY_ULONG    |   MPI_UNSIGNED_LONG    |
|  YT_LONGLONG  |     long long      |  NPY_LONGLONG   |     MPI_LONG_LONG      |
| YT_ULONGLONG  | unsigned long long |  NPY_ULONGLONG  | MPI_UNSIGNED_LONG_LONG |




