---
layout: default
title: yt_getGridInfo_* -- Look up passed in data
parent: libyt API
nav_order: 7
---
# Look Up Hierarchy and Data API
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

> :information_source: These APIs simply look up hierarchy array constructed by `libyt` or look up data pointer that is 
> passed in by user and wrapped by `libyt`.

> :warning: These APIs are only available after [`yt_commit`]({% link libytAPI/CommitYourSettings.md %}#yt_commit) is called.

## yt\_getGridInfo\_Dimensions
```cpp
int yt_getGridInfo_Dimensions( const long gid, int (*dimensions)[3] );
```
- Usage: Get dimension of grid `gid`. `dimensions[0]` corresponds to dimension in x-axis, `dimensions[1]` corresponds to dimension in y-axis, and `dimensions[2]` corresponds to dimension in z-axis, excluding ghost cells.
- Return: `YT_SUCCESS` or `YT_FAIL`

## yt\_getGridInfo\_LeftEdge
```cpp
int yt_getGridInfo_LeftEdge(const long gid, double (*left_edge)[3]);
```
- Usage: Get left edge of grid `gid`. `left_edge[0]` is left edge of the grid in x-axis in code length, `left_edge[1]` for y-axis, and `left_edge[2]` for z-axis.
- Return: `YT_SUCCESS` or `YT_FAIL`

## yt\_getGridInfo\_RightEdge
```cpp
int yt_getGridInfo_RightEdge(const long gid, double (*right_edge)[3]);
```
- Usage: Get right edge of grid `gid`. `right_edge[0]` is right edge of the grid in x-axis in code length, `right_edge[1]` for y-axis, and `right_edge[2]` for z-axis.
- Return: `YT_SUCCESS` or `YT_FAIL`

## yt\_getGridInfo\_ParentId
```cpp
int yt_getGridInfo_ParentId(const long gid, long *parent_id);
```
- Usage: Get parent grid id of grid `gid`. If there is no parent grid, `parent_id` will be `-1`.
- Return: `YT_SUCCESS` or `YT_FAIL`

## yt\_getGridInfo\_Level
```cpp
int yt_getGridInfo_Level(const long gid, int *level);
```
- Usage: Get level of grid `gid`. If grids are on root level, `level` will be `0`.
- Return: `YT_SUCCESS` or `YT_FAIL`

## yt\_getGridInfo\_ProcNum
```cpp
int yt_getGridInfo_ProcNum(const long gid, int *proc_num);
```
- Usage: Get MPI process number (MPI rank) of grid `gid` located on.
- Return: `YT_SUCCESS` or `YT_FAIL`

## yt\_getGridInfo\_ParticleCount
```cpp
int yt_getGridInfo_ParticleCount(const long gid, const char *ptype, long *par_count);
```
- Usage: Get number of particle `ptype` located on grid `gid`.
- Return: `YT_SUCCESS` or `YT_FAIL`
> :information_source: Particle type name `ptype` should be same as what you passed in [`yt_get_ParticlesPtr`]({% link libytAPI/SetParticlesInformation.md %}#yt_get_particlesptr).

## yt\_getGridInfo\_FieldData
```cpp
int yt_getGridInfo_FieldData( const long gid, const char *field_name, yt_data *field_data);
```
- Usage: Get the field data `field_name` in grid id `gid`. The result will be stored in `field_data`.
- Return: `YT_SUCCESS` or `YT_FAIL`
- `yt_data`
    - `data_ptr`: Data pointer.
    - `data_dimensions[3]`: Dimension of the `data_ptr` array, in the point of view of itself.
    - `data_dtype`: Data type of the array.

> :information_source: Field name `field_name` should be same as what you passed in [`yt_get_FieldsPtr`]({% link libytAPI/FieldInfo/SetFieldsInformation.md %}#yt_get_fieldsptr).

> :information_source: Do not mix grid dimensions get through [`yt_getGridInfo_Dimensions`](#yt_getgridinfo_dimensions) with data dimensions get through [`yt_getGridInfo_FieldData`](#yt_getgridinfo_fielddata). Grid dimensions are numbers of cells in [x][y][z] <--> [0][1][2], excluding ghost cells. Whereas data dimensions are just data length in data's point of view, which may consist of ghost cells.

> :warning: You should not be modifying `data_ptr`, because they are actual simulation data passed in by user when setting grid information [`yt_get_GridsPtr`]({% link libytAPI/SetLocalGridsInformation.md %}#yt_get_gridsptr).
