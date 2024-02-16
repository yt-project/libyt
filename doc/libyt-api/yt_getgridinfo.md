# `yt_getGridInfo_*` -- Look Up Data

> {octicon}`info;1em;sd-text-info;` These APIs simply look up hierarchy array constructed by `libyt` or look up data pointer that is 
> passed in by user and wrapped by `libyt`.

> {octicon}`alert;1em;sd-text-danger;` These APIs are only available after [`yt_commit`](./yt_commit.md#yt-commit) is called.

## `yt_getGridInfo_Dimensions`
```cpp
int yt_getGridInfo_Dimensions( const long gid, int (*dimensions)[3] );
```
- Usage: Get dimension of grid `gid`. `dimensions[0]` corresponds to dimension in x-axis, `dimensions[1]` corresponds to dimension in y-axis, and `dimensions[2]` corresponds to dimension in z-axis, excluding ghost cells.
- Return: `YT_SUCCESS` or `YT_FAIL`

## `yt_getGridInfo_LeftEdge`
```cpp
int yt_getGridInfo_LeftEdge(const long gid, double (*left_edge)[3]);
```
- Usage: Get left edge of grid `gid`. `left_edge[0]` is left edge of the grid in x-axis in code length, `left_edge[1]` for y-axis, and `left_edge[2]` for z-axis.
- Return: `YT_SUCCESS` or `YT_FAIL`

## `yt_getGridInfo_RightEdge`
```cpp
int yt_getGridInfo_RightEdge(const long gid, double (*right_edge)[3]);
```
- Usage: Get right edge of grid `gid`. `right_edge[0]` is right edge of the grid in x-axis in code length, `right_edge[1]` for y-axis, and `right_edge[2]` for z-axis.
- Return: `YT_SUCCESS` or `YT_FAIL`

## `yt_getGridInfo_ParentId`
```cpp
int yt_getGridInfo_ParentId(const long gid, long *parent_id);
```
- Usage: Get parent grid id of grid `gid`. If there is no parent grid, `parent_id` will be `-1`.
- Return: `YT_SUCCESS` or `YT_FAIL`

## `yt_getGridInfo_Level`
```cpp
int yt_getGridInfo_Level(const long gid, int *level);
```
- Usage: Get level of grid `gid`. If grids are on root level, `level` will be `0`.
- Return: `YT_SUCCESS` or `YT_FAIL`

## `yt_getGridInfo_ProcNum`
```cpp
int yt_getGridInfo_ProcNum(const long gid, int *proc_num);
```
- Usage: Get MPI process number (MPI rank) of grid `gid` located on.
- Return: `YT_SUCCESS` or `YT_FAIL`

## `yt_getGridInfo_ParticleCount`
```cpp
int yt_getGridInfo_ParticleCount(const long gid, const char *ptype, long *par_count);
```
- Usage: Get number of particle `ptype` located on grid `gid`.
- Return: `YT_SUCCESS` or `YT_FAIL`
> {octicon}`info;1em;sd-text-info;` Particle type name `ptype` should be same as what you passed in [`yt_get_ParticlesPtr`](./yt_get_particlesptr.md#yt-get-particlesptr).

## `yt_getGridInfo_FieldData`
```cpp
int yt_getGridInfo_FieldData(const long gid, const char *field_name, yt_data *field_data);
```
- Usage: Get the field data `field_name` in grid id `gid`. The result will be stored in `field_data`.
- Return: `YT_SUCCESS` or `YT_FAIL` if it cannot get data.
- `yt_data`
    - `data_ptr`: Data pointer.
    - `data_dimensions[3]`: Dimension of the `data_ptr` array, in the point of view of itself.
    - `data_dtype`: Data type of the array.

> {octicon}`info;1em;sd-text-info;` Field name `field_name` should be same as what you passed in [`yt_get_FieldsPtr`](./field/yt_get_fieldsptr.md#yt-get-fieldsptr).

> {octicon}`info;1em;sd-text-info;` Do not mix grid dimensions in [`yt_getGridInfo_Dimensions`](#yt-getgridinfo-dimensions) with data dimensions get through [`yt_getGridInfo_FieldData`](#yt-getgridinfo-fielddata). Grid dimensions are numbers of cells in [x][y][z] <--> [0][1][2], excluding ghost cells. Whereas data dimensions are just data length in data's point of view, which may consist of ghost cells.

> {octicon}`alert;1em;sd-text-danger;` You should not modify `data_ptr`, because they are actual simulation data passed in by user when setting grid information [`yt_get_GridsPtr`](./yt_get_gridsptr.md#yt-get-gridsptr).

## `yt_getGridInfo_ParticleData`
```cpp
int yt_getGridInfo_ParticleData(const long gid, const char *ptype, const char *attr, yt_data *par_data);
```
- Usage: Get the particle data `ptype` attribute `attr` in grid id `gid`. The result will be stored in `par_data`.
- Return: `YT_SUCCESS` or `YT_FAIL` if it cannot get data.
- `yt_data`
  - `data_ptr`: Data pointer.
  - `data_dimensions[3]`: Dimension of the `data_ptr` array, in the point of view of itself.  If `data_ptr` is a 1-dim array, the last two elements will be 0.
  - `data_dtype`: Data type of the array.

> {octicon}`info;1em;sd-text-info;` Particle type `ptype` and attribute `attr` should be the same as what you passed in [`yt_get_ParticlesPtr`](./yt_get_particlesptr.md#yt-get-particlesptr).

> {octicon}`alert;1em;sd-text-danger;` You should not modify `data_ptr`, because they are actual simulation data passed in by user when setting grid information [`yt_get_GridsPtr`](./yt_get_gridsptr.md#yt-get-gridsptr).
