# Set `yt` Parameter
## yt_set_parameter
```cpp
int yt_set_parameter( yt_param_yt *param_yt )
```
- Usage: Set `yt` parameter, number of fields, number of particle types and how many attributes do each of them have, and number of local grids exist on this MPI rank.
- Return: `YT_SUCCESS` or `YT_FAIL`

## yt_param_yt
- `frontend` (Default=`NULL`)
  - Usage: Field information of the yt `frontend` to borrow from. This should be `yt` supported frontend.
- `fig_basename` (Default=`"Fig"`)
  - Usage: Base name of the output figures. Figure name will also be followed by counter number and `yt` functionality name.
- `domain_left_edge`, `domain_right_edge`
  - Usage: Simulation left edge and right edge in code units.
- `current_time`
  - Usage: Simulation time in code units.
- `current_redshift`
  - Usage: Redshift.
- `omega_lambda`
  - Usage: Dark energy mass density.
- `omega_matter`
  - Usage: Dark matter mass density.
- `hubble_constant`
  - Usage: Dimensionless Hubble parameter at the present day.
- `length_unit`
  - Usage: Simulation length unit in cm (CGS).
- `mass_unit`
  - Usage: Simulation mass unit in g (CGS).
- `time_unit`
  - Usage: Simulation time unit in s (CGS).
- `magnetic_unit` (Default=`1.0`)
  - Usage: Simulation magnetic unit in gauss.
- `periodicity`
  - Usage: Periodicity along each dimension [x][y][z].
  - Valid Value:
    - `0`: No
    - `1`: Yes
- `cosmological_simulation`
  - Usage: Cosmological simulation dataset.
  - Valid Value:
    - `0`: No
    - `1`: Yes
- `dimensionality`
  - Usage: Dimensionality of the simulation. We only support 3 for now.
- `domain_dimensions`
  - Usage: Number of cells along each dimension on the root AMR level.
- `refine_by`
  - Usage: Refinement factor between a grid and its subgrid.
- `num_grids`
  - Usage: Total number of grids.
- `num_fields` (Default=`0`)
  - Usage: Number of fields.
- `num_species` (Default=`0`)
  - Usage: Number of particle types.
- `species_list` (Default=`NULL`)
  - Usage: Species list of particles. This should be a `yt_species` array. The lifespan of this array should at least cover `yt_inline` and `yt_inline_argument` API, which is when [Perform Inline-Analysis](./PerformInlineAnalysis.md).
  - Valid Value: Each element in `yt_species` array
    - `species_name`: Name of the particle type.
    - `num_attr`: Number of attributes does this particle type has. 
- `num_grids_local`
  - Usage: Number of local grids store on this rank now.

## Example
```cpp
yt_param_yt param_yt;  

/* Set YT parameter. */
param_yt.length_unit             = 3.0857e21;
...

/* Set frontend name, 
   so libyt can borrow the field information class of that frontend in YT. */
param_yt.frontend                = "gamer";  

/* Set figure base name. */
param_yt.fig_basename            = "FigName";

/* Set number of grids, local grids, and fields. */
param_yt.num_grids_local         = num_grids_local;
param_yt.num_grids               = num_grids;
param_yt.num_fields              = num_fields;

/* Set number of particle types and number of their attributes. */
yt_species  *species_list    = new yt_species [num_species];
species_list[0].species_name = "io";
species_list[0].num_attr     = 4;
species_list[1].species_name = "par2";
species_list[1].num_attr     = 4;
param_yt.num_species         = num_species;
param_yt.species_list        = species_list;

/* libyt API */
if ( yt_set_parameter( &param_yt ) != YT_SUCCESS )  {  
    fprintf( stderr, "ERROR: yt_set_parameter() failed!\n" );  
    exit( EXIT_FAILURE );  
}
```
