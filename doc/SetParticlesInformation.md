# Set Particles Information
## yt\_get\_particlesPtr
```cpp
int yt_get_particlesPtr( yt_particle **particle_list );
```
- Usage: Get the `yt_particle` array pointer where `libyt` access particles information from. You should fill them in. If you don't have any particles, then skip this.
- Return: `YT_SUCCESS` or `YT_FAIL`

## yt\_particle
- `species_name`
  - Usage: Name of the particle type. `libyt` links your input [`species_list`](./SetYTParameter.md#yt_param_yt)'s data member `species_name` to this variable. You don't need to assign it again. 
- `num_attr`
  - Usage: Number of attributes does this particle type has. `libyt` will assign your input [`species_list`](./SetYTParameter.md#yt_param_yt)'s data member `num_attr` to this variable. You may skip this.
- `attr_list`
  - Usage: Attribute list of this particle. This is a `yt_attribute` array.
  - Valid Value: Each element in `yt_attribute` array should have
    - `attr_name` (Default=`NULL`)
      - Usage: Attribute name. The lifespan of this variable should at least cover `yt_inline` and `yt_inline_argument` API, which is when [Perform Inline-Analysis](./PerformInlineAnalysis.md).
    - `attr_dtype` (Default=`YT_DOUBLE`)
      - Usage: Attribute’s data type.
      - Valide Value: 
        - `YT_FLOAT`: C type float.
        - `YT_DOUBLE`: C type double.
        - `YT_INT`: C type int. 
        - `YT_LONG`: C type long.
    - `attr_unit` (Default=`""`)
      - Usage: Unit of the attribute, using `yt` unit system.
    - `num_attr_name_alias` (Default=`0`)
      - Usage: Number of name aliases.
    - `attr_name_alias` (Default=`NULL`)
      - Usage: A list of name aliases.
    - `attr_display_name` (Default=`NULL`)
      - Usage: Display name on the output figure. If it is not set, then it will use `attr_name` instead.
- `coor_x`, `coor_y`, `coor_z` (Default=`NULL`)
  - Usage: Attribute name representing coordinate or position x, y, and z.
- `get_attr` (Default=`NULL`)
  - Usage: Function pointer to get particle’s attribute.

> :information_source: `libyt` borrows the full field information class (`class XXXFieldInfo`) from [`frontend`](./SetYTParameter.md#yt_param_yt). It is OK not to set a field's `attr_unit`, `num_attr_name_alias`, `attr_name_alias`, `attr_display_name`, if this `attr_name` is already inside your frontend.
> If you are adding a totally new particle attribute, please add them. `libyt` will add these new attributes information alongside with your original one.

## Get Attribute Function
For each particle type, there should be one get attribute function `get_attr`. This function is able to write particle attribute to a 1-dimensional array, just through knowing the grid id and attribute to get.

`get_attr` must have a prototype like this:
```cpp
void GetAttr(long gid, char *attr_name, void *output);
```
The first argument is grid id, the second argument is attribute. This function is capable of writing attribute data in grid id into the third argument, a 1-dimensional array.

> :warning: You should always write your particle data in the same order, since we get attributes separately.

## Example
```cpp
int main(){
    ...
    /* libyt API. */
    yt_particle *particle_list;  
    yt_get_particlesPtr( &particle_list );

    /* Fill in particle informaiton. */
    particle_list[0].species_name = "io"; // This two line is redundant, since libyt has already filled in.  
    particle_list[0].num_attr     = 4;    // I type it here just to make things clear.
    
    /* This particle has position and level attributes. */
    char *attr_name[]  = {"ParPosX", "ParPosY", "ParPosZ", "Level"};
    char *attr_name_alias[] = {"grid_level"}; // Alias name for attribute level  
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
    particle_list[0].get_attr = par_io_get_attr;
}

void getPositionByGID( long gid, real (*Pos)[3] ){
    // Get the center position of the grid id = gid.
}

void getLevelByGID( long gid, int *Level ){
    // Get the level of the grid id = gid.
}

/* Get attribute function. */
void par_io_get_attr(long gid, char *attribute, void *data){
    long len_array = 1;
    real Pos[3];
    getPositionByGID( gid, &Pos );

    int Level;
    getLevelByGID( gid, &Level );

    if ( strcmp(attribute, "ParPosX") == 0 ){
        ((real *)data)[0] = Pos[0];
    }
    else if ( strcmp(attribute, "ParPosY") == 0 ){
        ((real *)data)[0] = Pos[1];
    }
    else if ( strcmp(attribute, "ParPosZ") == 0 ){
        ((real *)data)[0] = Pos[2];
    }
    else if ( strcmp(attribute, "Level") == 0 ){
        ((int  *)data)[0] = Level;
    }
}
```
