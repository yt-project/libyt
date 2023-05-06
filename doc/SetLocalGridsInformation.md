# Set Local Grids Information
## yt\_get\_GridsPtr
```cpp
int yt_get_GridsPtr( yt_grid **grids_local );
```
- Usage: Get the `yt_grid` pointer where `libyt` access grids information from. We should then fill in those information.
- Return: `YT_SUCCESS` or `YT_FAIL`

## yt\_grid
One `yt_grid` contains the hierarchy of the grid, particle counts, and field data in this grid.
### Hierarchy
- `double left_edge[3], right_edge[3]` (Default=`DBL_UNDEFINED`)
  - Usage: Grid left and right edge in code units.
- `long id` (Default=`LNG_UNDEFINED`)
  - Usage: Grid global id.
  - Valid Value: It should be contiguous starting from [`index_offset`](./SetYTParameter.md#ytparamyt) set in [`yt_set_Parameters`](./SetYTParameter.md#ytsetparameters).
- `long parent_id` (Default=`LNG_UNDEFINED`)
  - Usage: Parent grid id.
  - Valide Value:
    - Should be in between `0` and `num_grids - 1`.
    - If the grid does not have parent grid, set to `-1`.
- `int grid_dimensions[3]` (Default=`INT_UNDEFINED`)
  - Usage: Number of cells along each direction in [x][y][z] order excluding ghost cells.
- `int level` (Default=`INT_UNDEFINED`)
  - Usage: AMR level of the grid.
  - Valid Value:
    - We start root level at `0`, so it should be greater than or equal to `0`.

### Particle Counts
- `long* par_count_list` (initialized by `libyt`)
  - Usage: Number of particles in each particle type located in this grid. This `long` array has length equals to number of particle types. The particle order should be the same as your input in `par_type_list` when [Set `yt` Parameter](./SetYTParameter.md#yt_param_yt).
  - Valid Value: Should be greater than or equal to `0`.

### Field Data
- `yt_data* field_data` (initialized by `libyt`)
  - Usage: Store all the field data under this grid. This is a `yt_data` array with length equals to number of fields.
  - Data member in `yt_data`:
    - `void* data_ptr`: Data pointer to the field data of the grid.
    - `int data_dimensions[3]`: Dimension of `data_ptr`, which is the actual dimension of this pointer.
    - `yt_dtype data_dtype`: Data type of `data_ptr`.
      - Valid value for `yt_dtype`:
        - `YT_FLOAT`: C type float.
        - `YT_DOUBLE`: C type double.
        - `YT_LONGDOUBLE`: C type long double.
        - `YT_INT`: C type int.
        - `YT_LONG`: C type long.

> :information_source: If it is a cell-centered field, `libyt` will fill in `data_dimensions` according to `grid_dimensions` in [`yt_grid`](#yt_grid) and `field_ghost_cell` in [`yt_field`](./SetFieldsInformation.md#yt_field).
> Otherwise, you should always fill in `data_dimensions`, if you wish to wrap an existing data in memory.

> :information_source: You only need to set `data_dtype` when this grid's data type is different from the one set in fields'.

## Example

```cpp
/* libyt API */
yt_grid *grids_local;
yt_get_GridsPtr( &grids_local );

int index_local = 0;
for (int gid = 0; gid < param_yt.num_grids; gid = gid + 1){
    if (grids_MPI[gid] == myrank) {
        /* Fill in hierarchy. */
        for (int d = 0; d < 3; d = d+1) {
            grids_local[index_local].left_edge[d]  = sim_grids[gid].left_edge[d];
            grids_local[index_local].right_edge[d] = sim_grids[gid].right_edge[d];
            grids_local[index_local].grid_dimensions[d] = sim_grids[gid].grid_dimensions[d];
        }
        grids_local[index_local].id             = sim_grids[gid].id;
        grids_local[index_local].parent_id      = sim_grids[gid].parent_id;
        grids_local[index_local].level          = sim_grids[gid].level;

        /* Fill in particle count. */
        grids_local[index_local].par_count_list[0] = 1;

        /* Fill in field data. */
        for (int v = 0; v < param_yt.num_fields; v = v + 1){
            grids_local[index_local].field_data[v].data_ptr = sim_grids[gid].field_data[v].data_ptr;
        }

        index_local = index_local + 1;
    }
}
```
