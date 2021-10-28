# Set Fields Information
## yt\_get\_fieldsPtr
```cpp
int yt_get_fieldsPtr( yt_field **field_list )
```
- Usage: Get the `yt_field` pointer where `libyt` access fields information from. You should then fill in those information.
- Return: `YT_SUCCESS` or `YT_FAIL`

## yt_field
- `field_name` (Default=`NULL`)
  - Usage: Field name. Please set the field name that matches your field information in [`frontend`](./SetYTParameter.md#yt_param_yt), because `libyt` borrows the frontend's `class XXXFieldInfo`.
- `field_define_type` (Default=`"cell-centered"`)
  - Usage: Define type of the field.
  - Valid Value:
    - `"cell-centered"`: Cell-centered data.
    - `"face-centered"`: Face-centered data. For more details, see [Face-Centered Field](./FaceCenteredField.md).
    - `"derived_func"`: Derived field data. When you want your simulation code to generate or convert existing data for `yt`, set to this. See [Derived Field](./DerivedField.md) on how to set them.
- `field_ghost_cell` (Default=`0`)
  - Usage: Number of ghost cell to be ignored at the beginning and the end of each dimension. This is in the point of view of the data array.
    `field_ghost_cell[0]`: Number of ghost cell to be ignored at the beginning of 0-dim of the data.
	`field_ghost_cell[1]`: Number of ghost cell to be ignored at the end of 0-dim of the data.
	`field_ghost_cell[2]`: Number of ghost cell to be ignored at the beginning of 1-dim of the data.
	`field_ghost_cell[3]`: Number of ghost cell to be ignored at the end of 1-dim of the data.
	`field_ghost_cell[4]`: Number of ghost cell to be ignored at the beginning of 2-dim of the data.
	`field_ghost_cell[5]`: Number of ghost cell to be ignored at the beginning of 2-dim of the data.
  - Valid Value: Must be greater than or equal to `0`.
- `field_dtype` (Default=`YT_DTYPE_UNKNOWN`)
  - Usage: Data type of the field.
  - Valid Value: 
    - `YT_FLOAT`: C type float.
    - `YT_DOUBLE`: C type double.
    - `YT_INT`: C type int. We do not guaranty that an `int` array will work or not, because `yt` requires a 64-bit float array ([links](https://github.com/yt-project/yt/blob/03f7a8bd0d75434034f3762c12daacb5efeb7527/yt/frontends/_skeleton/io.py#L31)).
    - `YT_LONG`: C type long. We do not guaranty this will work.
- `swap_axes` (Default=`true`)
  - Usage: Is the 3D data array define as [z][y][x], which is x address alters first.
  - Valid Value:
    - `true`: Data is in x-address alters first orientation, which is [z][y][x].
    - `false`: Data is in z-address alters first orientation, which is [x][y][z].
- `derived_func` (Default=`NULL`)
  - Usage: Function pointer to get derived field data. This is only used in derived field, which is when `field_define_type` set to `derived_func`. See [Derived Field](./DerivedField.md) for more information.
- `field_unit` (Default=`""`)
  - Usage: Unit of the field, using `yt` unit system.
- `num_field_name_alias` (Default=`0`)
  - Usage: Number of name aliases in `field_name_alias`.
- `field_name_alias` (Default=`NULL`)
  - Usage: Name aliases.
- `field_display_name` (Default=`NULL`)
  - Usage: Display name of the field on the output figure. If not set, `yt` uses its field name instead.
## Example
```cpp
/* libyt API */  
yt_field *field_list;  
yt_get_fieldsPtr( &field_list );

/* We only have one field in this example. */ 
field_list[0].field_name = "Dens";  
field_list[0].field_define_type = "cell-centered";  
field_list[0].field_dtype = ( typeid(real) == typeid(float) ) ? YT_FLOAT : YT_DOUBLE;  
char *field_name_alias[] = {"Name Alias 1", "Name Alias 2", "Name Alias 3"};  
field_list[0].field_name_alias = field_name_alias;  
field_list[0].num_field_name_alias = 3;  
for(int d = 0; d < 6; d++){
    field_list[0].field_ghost_cell[d] = GHOST_CELL;  
}
```