# `yt_get_ParticlesPtr` -- Set Particle Information

## `yt_get_ParticlesPtr`
```cpp
int yt_get_ParticlesPtr( yt_particle **particle_list );
```
- Usage: Get the `yt_particle` array pointer where `libyt` access particles information from. Each MPI rank should call this function and fill them in. If you don't have any particles, then skip this.
- Return: `YT_SUCCESS` or `YT_FAIL`

> {octicon}`alert;1em;sd-text-danger;` Every MPI rank must call this API and fill in the particle information in the same order. We do not broadcast and sync information here.

### `yt_particle`
- `const char* par_type` (set by `libyt`)
  - Usage: Name of the particle type. `libyt` only copies the pointer from [`par_type_list`]({% link libytAPI/SetYTParameter.md %}#yt_param_yt)'s data member `par_type` to this variable and does not make a hard copy. You don't need to assign it again. Refer to [Naming and Field Information]({% link InSituPythonAnalysis/UsingYT.md %}#naming-and-field-information) for how particle/attribute names and yt fields are linked and reused.
    > {octicon}`pencil;1em;sd-text-warning;` The lifetime of `par_type` should cover in situ analysis process. `libyt` only borrows this pointer and does not make a hard copy.
- `int num_attr` (set by `libyt`)
  - Usage: Number of attributes does this particle type has. `libyt` will assign your input [`par_type_list`]({% link libytAPI/SetYTParameter.md %}#yt_param_yt)'s data member `num_attr` to this variable. You may skip this.
- `yt_attribute* attr_list` (initialized by `libyt`)
  - Usage: Attribute list of this particle. This is a `yt_attribute` array with length `num_attr`.
  - Data member in `yt_attribute`:
    - `const char* attr_name` (Default=`NULL`)
      - Usage: Attribute name. Refer to [Naming and Field Information]({% link InSituPythonAnalysis/UsingYT.md %}#naming-and-field-information) for how particle/attribute names and yt fields are linked and reused.
      > {octicon}`pencil;1em;sd-text-warning;` The lifetime of `attr_name` should cover in situ analysis process. `libyt` only borrows this variable and does not make a copy.
    - `yt_dtype attr_dtype` (Default=`YT_DOUBLE`)
      - Usage: Attribute’s data type.
      - Valid Value:  [`yt_dtype`]({% link libytAPI/DataType.md %}#yt_dtype)
    - `const char* attr_unit` (Default=`""`)
      - Usage: Unit of the attribute, using `yt` unit system.
      > {octicon}`pencil;1em;sd-text-warning;` The lifetime of `attr_unit` should cover [`yt_commit`]({% link libytAPI/CommitYourSettings.md %}#yt_commit).
    - `int num_attr_name_alias` (Default=`0`)
      - Usage: Number of name aliases.
    - `const char **attr_name_alias` (Default=`NULL`)
      - Usage: A list of name aliases.
      > {octicon}`pencil;1em;sd-text-warning;` The lifetime of `attr_name_alias` should cover [`yt_commit`]({% link libytAPI/CommitYourSettings.md %}#yt_commit).
    - `const char *attr_display_name` (Default=`NULL`)
      - Usage: Display name on the output figure. If it is not set, then it will use `attr_name` instead.
      > {octicon}`pencil;1em;sd-text-warning;` The lifetime of `attr_display_name` should cover [`yt_commit`]({% link libytAPI/CommitYourSettings.md %}#yt_commit).
- `const char *coor_x, *coor_y, *coor_z` (Default=`NULL`)
  - Usage: Attribute name representing coordinate or position x, y, and z.
  > {octicon}`pencil;1em;sd-text-warning;` The lifetime of `coor_x`, `coor_y`, `coor_z` should cover the in situ analysis process. `libyt` only borrows these names and does not make a copy.
- `void (*get_par_attr) (const int, const long*, const char*, const char*, yt_array*)` (Default=`NULL`)
  - Usage: Function pointer to get particle’s attribute.

> {octicon}`info;1em;sd-text-info;` `libyt` borrows the full field and particle information class (`class XXXFieldInfo`) from [`frontend`]({% link libytAPI/SetYTParameter.md %}#yt_param_yt). It is OK not to set a particle's `attr_unit`, `num_attr_name_alias`, `attr_name_alias`, `attr_display_name`, if this `attr_name` is already inside your frontend.
> If you are adding a totally new particle attribute, please add them. `libyt` will add these new attributes information alongside with your original one.

## Get Particle Attribute Function
For each particle type, there should be one get particle attribute function `get_par_attr`. This function is able to write particle attribute to an array, just through knowing the grid id, particle type, and attribute name.

Get particle attribute function must have a prototype like this:
```cpp
void GetAttr(const int list_len, const long *list_gid, const char *par_type, const char *attr_name, yt_array *data_array);
```
- `get_par_attr(const int, const long*, const char*, const char*, yt_array*)`: generate particle attribute in that grid when input grid id, particle type, and particle attribute name.
  - `const int list_len`: number of grid id in `list_gid`.
  - `const long *list_gid`: prepare particle data inside the grid id in this list.
  - `const char *par_type`: target particle type to prepare.
  - `const char *attr_name`: target attribute to prepare.
  - `yt_array *data_array`: write generated particle data to the pointer in this array correspondingly. Fill in particle attribute inside `yt_array` array using the same order as in `list_gid`.

### `yt_array`
- Usage: a struct used in derived function and get particle attribute function.
- Data Member:
  - `long gid`: grid id.
  - `long data_length`: length of `data_ptr`.
  - `void *data_ptr`: data pointer where you should write in particle data of this grid.

> {octicon}`alert;1em;sd-text-danger;` We should always write our particle attribute data in the same order, since we get attributes separately.

## Example
`par_io_get_par_attr` function gets particle type `io` attributes. This particle type has position at the center of the grid it belongs to with value grid level (int).
```cpp
int main(){
    ...
    /* libyt API. */
    yt_particle *particle_list;  
    yt_get_ParticlesPtr( &particle_list );
    
    /* This particle "io" has 4 attributes (position X/Y/Z and level). */
    const char *attr_name[]  = {"ParPosX", "ParPosY", "ParPosZ", "Level"};
    const char *attr_name_alias[] = {"grid_level"};
    for ( int v=0; v < 4; v++ ){
        particle_list[0].attr_list[v].attr_name  = attr_name[v];
        if ( v == 3 ){  
            particle_list[0].attr_list[v].attr_dtype = YT_INT;
            particle_list[0].attr_list[v].attr_unit  = "";
            particle_list[0].attr_list[v].num_attr_name_alias = 1;
            particle_list[0].attr_list[v].attr_name_alias     = attr_name_alias;  
            particle_list[0].attr_list[v].attr_display_name   = "Level of the Grid";
        }     
        else{   
            particle_list[0].attr_list[v].attr_dtype = ( typeid(real) == typeid(float) ) ? YT_FLOAT : YT_DOUBLE;
        }
    }
    
    /* Fill in positions attribute name. */
    particle_list[0].coor_x = attr_name[0];
    particle_list[0].coor_y = attr_name[1];  
    particle_list[0].coor_z = attr_name[2];
    
    /* Fill in get attribute function pointer. */
    particle_list[0].get_par_attr = par_io_get_par_attr;
}

void par_io_get_par_attr(const int list_len, const long *gid_list, const char *par_type, const char *attribute, yt_array *data_array) {
    // loop over gid_list, and fill in particle attribute data inside data_array.
    for (int lid = 0; lid < list_len; lid++) {
        // =============================================================
        // libyt: [Optional] Use libyt look up grid info API
        // =============================================================
        int Level;
        double RightEdge[3], LeftEdge[3];
        yt_getGridInfo_Level(gid_list[lid], &Level);
        yt_getGridInfo_RightEdge(gid_list[lid], &RightEdge);
        yt_getGridInfo_LeftEdge(gid_list[lid], &LeftEdge);

        // fill in particle data.
        // we can get the length of the array to fill in like this, though this example only has one particle in each grid.
        for (int i = 0; i < data_array[lid].data_length; i++) {
            // fill in particle data according to the attribute.
            if (strcmp(attribute, "ParPosX") == 0) {
                ((real *) data_array[lid].data_ptr)[0] = 0.5 * (RightEdge[0] + LeftEdge[0]);
            } else if (strcmp(attribute, "ParPosY") == 0) {
                ((real *) data_array[lid].data_ptr)[0] = 0.5 * (RightEdge[1] + LeftEdge[1]);
            } else if (strcmp(attribute, "ParPosZ") == 0) {
                ((real *) data_array[lid].data_ptr)[0] = 0.5 * (RightEdge[2] + LeftEdge[2]);
            } else if (strcmp(attribute, "Level") == 0) {
                ((int *) data_array[lid].data_ptr)[0] = Level;
            }
        }
    }

}
```
