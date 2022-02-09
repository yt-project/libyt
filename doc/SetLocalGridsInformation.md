# Set Local Grids Information
## yt\_get\_gridsPtr
```cpp
int yt_get_gridsPtr( yt_grid **grids_local );
```
- Usage: Get the `yt_grid` pointer where `libyt` access grids information from. You should then fill in those information.
- Return: `YT_SUCCESS` or `YT_FAIL`

## yt\_grid
One `yt_grid` contains the hierarchy of the grid, particle counts, and field data in this grid.
### Hierarchy
- `left_edge`, `right_edge`
  - Usage: Grid left and right edge in code units.
- `id`
  - Usage: Grid global id.
  - Valid Value: It is 0-index based and should be in between `0` and `num_grids - 1`.
- `parent_id`
  - Usage: Parent grid id.
  - Valide Value:
    - Should be in between `0` and `num_grids - 1`.
    - If the grid does not have parent grid, set to `-1`.
- `grid_dimensions`
  - Usage: Number of cells along each direction in [x][y][z] order.
- `level`
  - Usage: AMR level of the grid.
  - Valid Value:
    - We start root level at `0`, so it should be greater than or equal to `0`.

### Particle Counts
- `particle_count_list` (Default=`0`)
  - Usage: Number of particles in each particle type. The particle order should be the same as your input in `species_list` when [Set `yt` Parameter](./SetYTParameter.md#yt_param_yt).
  - Valid Value: Should be greater than or equal to `0`.

### Field Data
- `field_data`
  - Usage: Store all the field data under this grid. This is a `yt_data` array, and `libyt` will initialize this for you.
  - Valid Value: Each element in `field_data` is a `yt_data` struct.
    - `data_ptr`: Data pointer to the field data of the grid.
    - `data_dimensions[3]`: Dimension of `data_ptr`.
    - `data_dtype`: Data type of `data_ptr`.

> :information_source: If it is a cell-centered field, `libyt` will fill in `data_dimensions` according to `grid_dimensions` in [`yt_grid`](#yt_grid) and `field_ghost_cell` in [`yt_field`](./SetFieldsInformation.md#yt_field).
> Otherwise, you should always fill in `data_dimensions`, if you wish to wrap an existing data in memory.

> :information_source: You only need to set `data_dtype` when this grid's data type is different from the one set in fields'.

## Example

```cpp
/* libyt API */
yt_grid *grids_local;
yt_get_gridsPtr( &grids_local );

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
        grids_local[index_local].particle_count_list[0] = 1;

        /* Fill in field data. */
        for (int v = 0; v < param_yt.num_fields; v = v + 1){
            grids_local[index_local].field_data[v].data_ptr = sim_grids[gid].field_data[v].data_ptr;
        }

        index_local = index_local + 1;
    }
}
```
