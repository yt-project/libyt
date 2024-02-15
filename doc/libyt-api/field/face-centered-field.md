# Face-Centered Field

## Definition of Face-Centered Field
After ignoring the ghost cells, a face-centered data should only have one dimension greater than the grid's by one, considering swap axes or not. We will then do interpolation to convert it to cell-centered data.

## How libyt Convert Face-Centered to Cell-Centered Data
1. `libyt` reads simulation data wrapped by `libyt` and ignores ghost cells.
2. Do interpolation on the axis that has dimension different from the grid dimension by one.

```python
# yt_libyt/io.py
...
# data_temp is raw data.
# Remove ghost cell, and get my slice.
data_shape = data_temp.shape
data_temp = data_temp[ghost_cell[0]:(data_shape[0] - ghost_cell[1]),
                      ghost_cell[2]:(data_shape[1] - ghost_cell[3]),
                      ghost_cell[4]:(data_shape[2] - ghost_cell[5])]

# Convert to cell-centered
grid_dim = self.hierarchy["grid_dimensions"][grid.id]
if field_list[fname]["contiguous_in_x"] is True:
    grid_dim = np.flip(grid_dim)
axis = np.argwhere(grid_dim != data_temp.shape)
if axis == 0:
    data_convert = 0.5 * (data_temp[:-1, :, :] + data_temp[1:, :, :])
elif axis == 1:
    data_convert = 0.5 * (data_temp[:, :-1, :] + data_temp[:, 1:, :])
elif axis == 2:
    data_convert = 0.5 * (data_temp[:, :, :-1] + data_temp[:, :, 1:])
...
```
