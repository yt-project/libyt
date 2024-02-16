# `yt_get_FieldsPtr` -- Set Field Information

```{toctree}
:hidden:

derived-field
face-centered-field
```

## `yt_get_FieldsPtr`
```cpp
int yt_get_FieldsPtr( yt_field **field_list )
```
- Usage: Get the `yt_field` array pointer where `libyt` access fields information from. Each MPI rank should call this function and fill them in. If you don't have any fields, then skip this.
- Return: `YT_SUCCESS` or `YT_FAIL`

> {octicon}`alert;1em;sd-text-danger;` Every MPI rank must call this API and fill in the field information in the same order. `libyt` does not broadcast and sync information here.

### `yt_field`
- `const char* field_name` (Default=`NULL`)
  - Usage: Field name. It should be unique.
  > {octicon}`pencil;1em;sd-text-warning;` Make sure the lifetime of `field_name` covers the whole in situ analysis process. `libyt` only borrows this name and does not make a copy.
- `const char* field_type` (Default=`"cell-centered"`)
  - Usage: Define type of the field.
  - Valid Value:
    - `"cell-centered"`: Cell-centered data.
    - `"face-centered"`: Face-centered data. For more details, see [Face-Centered Field](./face-centered-field.md#face-centered-field).
    - `"derived_func"`: Derived field data. Simulation code will generate data when it is called by Python using user-provided C function. See [Derived Field](./derived-field.md#derived-field).
  > {octicon}`pencil;1em;sd-text-warning;` Please make sure the lifetime of `field_type` covers the whole in situ analysis process. `libyt` does not make a copy.
- `short field_ghost_cell[6]` (Default=`0`)
  - Usage: Number of ghost cell to be ignored at the beginning and the end of each dimension. This is from the point of view of the data array. <br>
    `field_ghost_cell[0]`: Number of ghost cell to be ignored at the beginning of 0-dim of the data.<br>
    `field_ghost_cell[1]`: Number of ghost cell to be ignored at the end of 0-dim of the data.<br>
    `field_ghost_cell[2]`: Number of ghost cell to be ignored at the beginning of 1-dim of the data.<br>
    `field_ghost_cell[3]`: Number of ghost cell to be ignored at the end of 1-dim of the data.<br>
    `field_ghost_cell[4]`: Number of ghost cell to be ignored at the beginning of 2-dim of the data.<br>
    `field_ghost_cell[5]`: Number of ghost cell to be ignored at the end of 2-dim of the data.<br>
  - Valid Value: Must be greater than or equal to `0`.
- `yt_dtype field_dtype` (Default=`YT_DTYPE_UNKNOWN`)
  - Usage: Data type of the field.
  - Valid Value:  [`yt_dtype`](../data-type.md#yt-dtype)
- `bool contiguous_in_x` (Default=`true`)
  - Usage: Is the 3D data array define as [z][y][x], which is x address alters first.
  - Valid Value:
    - `true`: Data is in x-address alters first orientation, which is [z][y][x].
    - `false`: Data is in z-address alters first orientation, which is [x][y][z].
- `void (*derived_func) (const int, const long *, const char *, yt_array*)` (Default=`NULL`)
  - Usage: Function pointer to generate derived field data when input grid id. This is only used in derived field (`field_type="derived_func"`). See [Derived Field](./derived-field.md#derived-field).
- `const char* field_unit` (Default=`""`)
  - Usage: Unit of the field, using `yt` unit system.
  > {octicon}`pencil;1em;sd-text-warning;` Make sure the lifetime of `field_unit` covers [`yt_commit`](../yt_commit.md#yt-commit).
- `int num_field_name_alias` (Default=`0`)
  - Usage: Number of name aliases in `field_name_alias`.
- `const char** field_name_alias` (Default=`NULL`)
  - Usage: Name aliases.
  > {octicon}`pencil;1em;sd-text-warning;` Please make sure the lifetime of `field_name_alias` covers [`yt_commit`](../yt_commit.md#yt-commit).
- `const char* field_display_name` (Default=`NULL`)
  - Usage: Display name of the field on the output figure. If not set, `yt` uses its field name instead.
  > {octicon}`pencil;1em;sd-text-warning;` Please make sure the lifetime of `field_display_name` covers [`yt_commit`](../yt_commit.md#yt-commit).

> {octicon}`info;1em;sd-text-info;` `libyt` borrows the full field information class (`class XXXFieldInfo`) from [`frontend`](../yt_set_parameters.md#yt-param-yt). It is OK not to set a field's `field_unit`, `num_field_name_alias`, `field_name_alias`, `field_display_name`, if this `field_name` is already defined inside the frontend.
> If you are adding a totally new field, do add them. `libyt` will add these new field information alongside with your original one.
> 
> Refer to [Naming and Field Information](../../in-situ-python-analysis/using-yt.md#naming-and-field-information) for how field names and yt fields are linked and reused.

## Example
```cpp
/* libyt API */  
yt_field *field_list;
yt_get_FieldsPtr( &field_list );

// cell-centered type field "Dens" 
field_list[0].field_name = "Dens";  
field_list[0].field_type = "cell-centered";  
field_list[0].field_dtype = ( typeid(real) == typeid(float) ) ? YT_FLOAT : YT_DOUBLE;  
const char *field_name_alias[] = {"Name Alias 1", "Name Alias 2", "Name Alias 3"};  
field_list[0].field_name_alias = field_name_alias;  
field_list[0].num_field_name_alias = 3;  
for(int d = 0; d < 6; d++){
    field_list[0].field_ghost_cell[d] = GHOST_CELL;  
}
```
