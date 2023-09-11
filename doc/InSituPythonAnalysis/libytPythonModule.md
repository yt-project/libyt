---
layout: default
title: libyt Python Module
parent: In Situ Python Analysis
nav_order: 2
---
# libyt Python Module
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

## How to Import

`libyt` Python module is only importable in runtime.

```python
import libyt
```

## Dictionaries

### libyt_info

|               Key                |          Value          | Loaded by `libyt` API | Notes                               |
|:--------------------------------:|:-----------------------:|:---------------------:|-------------------------------------|
|     `libyt_info["version"]`      | `(major, minor, micro)` |    `yt_initialize`    | - `libyt` version.                  |
| `libyt_info["interactive_mode"]` |      `True/False`       |    `yt_initialize`    | - Is it in interactive mode or not. |

### param_yt

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


- Usage: Contain `yt` parameters. The values correspond to data members in  [`yt_param_yt`]({% link libytAPI/SetYTParameter.md%}#yt_param_yt).

### param_user

|        Key        | Value |           Loaded by `libyt` API            | Notes  |
|:-----------------:|:-----:|:------------------------------------------:|--------|
| `param_user[key]` | input |          `yt_set_UserParameter*`           |        |

- Usage: A series of key-value pairs set through `yt_set_UserParameter*`. The pairs will also be added as new attributes if `yt_libyt` is imported.

### hierarchy

|                     Key                     |      Value        | Loaded by `libyt` API  | Notes                                                                    |
|:-------------------------------------------:|:-----------------:|:----------------------:|--------------------------------------------------------------------------|
|      `hierarchy["grid_left_edge"][id]`      |    `left_edge`    |      `yt_commit`       |                                                                          |
|     `hierarchy["grid_right_edge"][id]`      |   `right_edge`    |      `yt_commit`       |                                                                          |
|     `hierarchy["grid_dimensions"][id]`      | `grid_dimensions` |      `yt_commit`       |                                                                          |
| `hierarchy["par_count_list"][id][par_idex]` | `par_count_list`  |      `yt_commit`       | `par_index` corresponds to particle type order in `yt_get_ParticlesPtr`. |
|      `hierarchy["grid_parent_id"][id]`      |    `parent_id`    |      `yt_commit`       |                                                                          |
|       `hierarchy["grid_levels"][id]`        |      `level`      |      `yt_commit`       |                                                                          |
|         `hierarchy["proc_num"][id]`         |    `proc_num`     |      `yt_commit`       |                                                                          |

- Usage: Contain AMR grid hierarchy. The values and `id` are corresponding to data members in [`yt_grid`]({% link libytAPI/SetLocalGridsInformation.md %}#yt_grid).

### grid_data

|          Key           |   Value    | Loaded by `libyt` API  | Notes  |
|:----------------------:|:----------:|:----------------------:|--------|
| `grid_data[id][fname]` | Field data |      `yt_commit`       |        |

- Usage: It only contains data in local process. The value corresponds to data member [`field_data`]({% link libytAPI/SetLocalGridsInformation.md %}#field-data-and-particle-data) in `yt_grid`.

### particle_data

|               Key                |         Value          | Loaded by `libyt` API  | Notes  |
|:--------------------------------:|:----------------------:|:----------------------:|--------|
| `particle_data[id][ptype][attr]` |     Particle data      |      `yt_commit`       |        |

- Usage: It only contains data in local process. The value corresponds to data member [`particle_data`]({% link libytAPI/SetLocalGridsInformation.md %}#field-data-and-particle-data) in `yt_grid`.

> :information_source: `grid_data` and `particle_data` is read-only. They contain the actual simulation data.

### interactive_mode

> :warning: Do not touch, it is for internal `libyt` process, and it only exists in interactive mode.

## Methods

### derived_func(gid : int, fname : str) -> numpy.ndarray
- Usage: Return derived field data `fname` in grid id `gid` generated by user-defined C function (See **[Derived Field]({% link libytAPI/FieldInfo/DerivedField.md %}#derived-field-function)**). It is a local process.

### get_particle(gid : int, ptype : str, attr : str) -> numpy.ndarray
- Usage: Return particle `ptype` attribute `attr` data in grid id `gid` generated by user-defined C function(See [**Get Particle Attribute Function**]({% link libytAPI/SetParticlesInformation.md%}#get-particle-attribute-function)). It is a local process.

### get_field_remote(fname_list : list, fname_list_len : int, prepare_list : list, prepare_list_len : int, fetch_gid_list : list, fetch_process_list : list, fetch_gid_list_len : int) -> dict
- Usage: Return a dictionary that contains requested field data from remote rank.

### get_particle_remote(par_dict : dict, par_dict : dict_keys, prepare_list : list, prepare_list_len : int, fetch_gid_list : list, fetch_process_list : list, fetch_gid_list_len : int) -> dict
- Usage: Return a dictionary that contains requested particle data from remote rank.

> :lizard: `get_field_remote` and `get_particle_remote` may be hard to use in general case, since we have to prepare those list by ourselves. We will improve this and make it general in the future.