# libyt Python Module

## Import `libyt` Python Module

```python
import libyt
```

> {octicon}`info;1em;sd-text-info;` `libyt` Python module is only importable during simulation runtime.

## Dictionaries

### `libyt_info`

|               Key                |          Value          | Loaded by `libyt` API | Notes                                       |
|:--------------------------------:|:-----------------------:|:---------------------:|---------------------------------------------|
|     `libyt_info["version"]`      | `(major, minor, micro)` |    `yt_initialize`    | - `libyt` version.                          |
|   `libyt_info["SERIAL_MODE"]`    |      `True/False`       |    `yt_initialize`    | - Compile with `-DSERIAL_MODE` or not       |
| `libyt_info["INTERACTIVE_MODE"]` |      `True/False`       |    `yt_initialize`    | - Compile with `-DINTERACTIVE_MODE` or not  |
|  `libyt_info["JUPYTER_KERNEL"]`  |      `True/False`       |    `yt_initialize`    | - Compile with `-DJUPYTER_KERNELF` or not   |
|  `libyt_info["SUPPORT_TIMER"]`   |      `True/False`       |    `yt_initialize`    | - Compile with `-DSUPPORT_TIMER` or not     |

### `param_yt`

|                  Key                  |           Value           | Loaded by `libyt` API | Notes                                |
|:-------------------------------------:|:-------------------------:|:---------------------:|--------------------------------------|
|        `param_yt["frontend"]`         |        `frontend`         |  `yt_set_Parameters`  |                                      |
|      `param_yt["fig_basename"]`       |      `fig_basename`       |  `yt_set_Parameters`  |                                      |
|      `param_yt["current_time"]`       |      `current_time`       |  `yt_set_Parameters`  |                                      |
|    `param_yt["current_redshift"]`     |    `current_redshift`     |  `yt_set_Parameters`  |                                      |
|      `param_yt["omega_lambda"]`       |      `omega_lambda`       |  `yt_set_Parameters`  |                                      |
|      `param_yt["omega_matter"]`       |      `omega_matter`       |  `yt_set_Parameters`  |                                      |
|     `param_yt["hubble_constant"]`     |     `hubble_constant`     |  `yt_set_Parameters`  |                                      |
|       `param_yt["length_unit"]`       |       `length_unit`       |  `yt_set_Parameters`  |                                      |
|        `param_yt["mass_unit"]`        |        `mass_unit`        |  `yt_set_Parameters`  |                                      |
|        `param_yt["time_unit"]`        |        `time_unit`        |  `yt_set_Parameters`  |                                      |
|      `param_yt["velocity_unit"]`      |      `velocity_unit`      |  `yt_set_Parameters`  |                                      |
|      `param_yt["magnetic_unit"]`      |      `magnetic_unit`      |  `yt_set_Parameters`  | - Will be set to 1, if it's not set. |
| `param_yt["cosmological_simulation"]` | `cosmological_simulation` |  `yt_set_Parameters`  |                                      |
|     `param_yt["dimensionality"]`      |     `dimensionality`      |  `yt_set_Parameters`  |                                      |
|        `param_yt["refine_by"]`        |        `refine_by`        |  `yt_set_Parameters`  |                                      |
|      `param_yt["index_offset"]`       |      `index_offset`       | `yt_set_Parameters`   | - Default value is 0.                |
|        `param_yt["num_grids"]`        |        `num_grids`        |  `yt_set_Parameters`  |                                      |
|    `param_yt["domain_left_edge"]`     |    `domain_left_edge`     |  `yt_set_Parameters`  |                                      |
|    `param_yt["domain_right_edge"]`    |    `domain_right_edge`    |  `yt_set_Parameters`  |                                      |
|       `param_yt["periodicity"]`       |       `periodicity`       |  `yt_set_Parameters`  |                                      |
|    `param_yt["domain_dimensions"]`    |    `domain_dimensions`    |  `yt_set_Parameters`  |                                      |


- Usage: Contain `yt` parameters. The values correspond to data members in  [`yt_param_yt`](../libyt-api/yt_set_parameters.md#yt_param_yt).

### `param_user`

:::{table}
:width: 100%

|        Key        | Value |           Loaded by `libyt` API            | Notes  |
|:-----------------:|:-----:|:------------------------------------------:|--------|
| `param_user[key]` | input |          `yt_set_UserParameter*`           |        |
:::

- Usage: A series of key-value pairs set through `yt_set_UserParameter*`. The pairs will also be added as new attributes if `yt_libyt` is imported.

### `hierarchy`

|                     Key                     |      Value        | Loaded by `libyt` API  | Notes                                                                    |
|:-------------------------------------------:|:-----------------:|:----------------------:|--------------------------------------------------------------------------|
|      `hierarchy["grid_left_edge"][id]`      |    `left_edge`    |      `yt_commit`       |                                                                          |
|     `hierarchy["grid_right_edge"][id]`      |   `right_edge`    |      `yt_commit`       |                                                                          |
|     `hierarchy["grid_dimensions"][id]`      | `grid_dimensions` |      `yt_commit`       |                                                                          |
| `hierarchy["par_count_list"][id][par_idex]` | `par_count_list`  |      `yt_commit`       | `par_index` corresponds to particle type order in `yt_get_ParticlesPtr`. |
|      `hierarchy["grid_parent_id"][id]`      |    `parent_id`    |      `yt_commit`       |                                                                          |
|       `hierarchy["grid_levels"][id]`        |      `level`      |      `yt_commit`       |                                                                          |
|         `hierarchy["proc_num"][id]`         |    `proc_num`     |      `yt_commit`       |                                                                          |

- Usage: Contain AMR grid hierarchy. The values and `id` are corresponding to data members in [`yt_grid`](../libyt-api/yt_get_gridsptr.md#yt_grid).

### `grid_data`

:::{table}
:width: 100%

|          Key           |   Value    | Loaded by `libyt` API  | Notes                                            |
|:----------------------:|:----------:|:----------------------:|--------------------------------------------------|
| `grid_data[id][fname]` | Field data |      `yt_commit`       | `fname` is field name set in `yt_get_FieldsPtr`. |
:::

- Usage: It only contains data in local process. The value corresponds to data member [`field_data`](../libyt-api/yt_get_gridsptr.md#field-data-and-particle-data).

### `particle_data`

:::{table}
:width: 100%

|               Key                |         Value          | Loaded by `libyt` API  | Notes                                                                            |
|:--------------------------------:|:----------------------:|:----------------------:|----------------------------------------------------------------------------------|
| `particle_data[id][ptype][attr]` |     Particle data      |      `yt_commit`       | `ptype` and `attr` are particle type and attribute set in `yt_get_ParticlesPtr`. |
:::

- Usage: It only contains data in local process. The value corresponds to data member [`particle_data`](../libyt-api/yt_get_gridsptr.md#field-data-and-particle-data) in `yt_grid`.

> {octicon}`info;1em;sd-text-info;` `grid_data` and `particle_data` is read-only. They contain the actual simulation data.

## Methods

### `derived_func`
```python
derived_func(gid : int, 
             fname : str) -> numpy.ndarray
```
- Usage: Return derived field data `fname` in grid id `gid` generated by user-defined C function (See [Derived Field](../libyt-api/field/derived-field.md#derived-field-function)). It is a local process and does not require other processes to join.

### `get_particle`
```python
get_particle(gid : int, 
             ptype : str, 
             attr : str) -> numpy.ndarray
```

- Usage: Return particle `ptype` attribute `attr` data in grid id `gid` generated by user-defined C function (See [Get Particle Attribute Function](../libyt-api/yt_get_particlesptr.md#get-particle-attribute-function)). It is a local process and does not require other processes to join.

### `get_field_remote`
```python
get_field_remote(fname_list : list, 
                 fname_list_len : int, 
                 prepare_list : list, 
                 prepare_list_len : int, 
                 fetch_gid_list : list, 
                 fetch_process_list : list, 
                 fetch_gid_list_len : int) -> dict
```
- Usage: Return a dictionary that contains requested field data. The data is scattered in different processes. It is a collective operation.

> {octicon}`alert;1em;sd-text-danger;` This is a collective operation, and it requires every MPI process to participate.

### `get_particle_remote`
```python
get_particle_remote(par_dict : dict, 
                    par_dict : dict_keys, 
                    prepare_list : list, 
                    prepare_list_len : int, 
                    fetch_gid_list : list, 
                    fetch_process_list : list, 
                    fetch_gid_list_len : int) -> dict
```
- Usage: Return a dictionary that contains requested particle data. The data is scattered in different processes. It is a collective operation.

> {octicon}`alert;1em;sd-text-danger;` This is a collective operation, and it requires every MPI process to participate.

> {octicon}`calendar;1em;sd-text-secondary;` [`get_field_remote`](#get_field_remote) and [`get_particle_remote`](#get_particle_remote) may be hard to use in general case, since we have to prepare those list by ourselves. We will improve this and make it general in the future.
