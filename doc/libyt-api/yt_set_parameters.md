# `yt_set_Parameters` -- Set yt Parameters

## `yt_set_Parameters`
```cpp
int yt_set_Parameters( yt_param_yt *param_yt )
```
- Usage: Set `yt` parameters, and other parameters like number of fields, number of particle types, number of their attributes, and number of local grids exist on this MPI rank. 
- Notes: 
  - We will reset all cosmological parameters (e.g. `current_redshift`, `omega_lambda`, `omega_matter`, `hubble_constant`) to `0` if `cosmological_simulation` is `0`.
- Return: `YT_SUCCESS` or `YT_FAIL`

### `yt_param_yt`
- `const char* frontend` (Default=`NULL`)
  - Usage: Field information of the yt `frontend` to borrow from. This should be `yt` supported frontend.
  > {octicon}`pencil;1em;sd-text-warning;` Make sure the lifetime of `frontend` covers [`yt_commit`](./yt_commit.md#yt-commit) if you set [`check_data`](./yt_initialize.md#yt-param-libyt) to `true` when initializing `libyt`.
- `const char* fig_basename` (Default=`"Fig"`)
  - Usage: Base name of the output figures. Figure name will also be followed by counter number and `yt` functionality name.
- `double domain_left_edge[3], domain_right_edge[3]` (Default=`DBL_UNDEFINED`)
  - Usage: Simulation left and right edge in code units.
- `double current_time` (Default=`DBL_UNDEFINED`)
  - Usage: Simulation time in code units.
- `double current_redshift` (Default=`DBL_UNDEFINED`)
  - Usage: Redshift.
- `double omega_lambda` (Default=`DBL_UNDEFINED`)
  - Usage: Dark energy mass density.
- `double omega_matter` (Default=`DBL_UNDEFINED`)
  - Usage: Dark matter mass density.
- `double hubble_constant` (Default=`DBL_UNDEFINED`)
  - Usage: Dimensionless Hubble parameter at the present day.
- `double length_unit` (Default=`DBL_UNDEFINED`)
  - Usage: Simulation length unit in cm (CGS).
- `double mass_unit` (Default=`DBL_UNDEFINED`)
  - Usage: Simulation mass unit in g (CGS).
- `double time_unit` (Default=`DBL_UNDEFINED`)
  - Usage: Simulation time unit in s (CGS).
- `double magnetic_unit` (Default=`1.0`)
  - Usage: Simulation magnetic unit in gauss.
- `int periodicity[3]` (Default=`INT_UNDEFINED`)
  - Usage: Periodicity along each dimension [x][y][z].
  - Valid Value:
    - `0`: No
    - `1`: Yes
- `int cosmological_simulation` (Default=`INT_UNDEFINED`)
  - Usage: Cosmological simulation dataset.
  - Valid Value:
    - `0`: No
    - `1`: Yes
- `int dimensionality` (Default=`INT_UNDEFINED`)
  - Usage: Dimensionality of the simulation. 
  > {octicon}`alert;1em;sd-text-danger;` `libyt` only support 3 for now.
- `int domain_dimensions[3]` (Default=`INT_UNDEFINED`)
  - Usage: Number of cells along each dimension on the root AMR level.
- `int refine_by` (Default=`INT_UNDEFINED`)
  - Usage: Refinement factor between a grid and its subgrid.
- `int index_offset` (Default=`0`)
  - Usage: Index offset.
- `long num_grids` (Default=`LNG_UNDEFINED`)
  - Usage: Total number of grids.
- `int num_grids_local` (Default=`0`)
  - Usage: Number of local grids store on this rank now.
- `int num_fields` (Default=`0`)
  - Usage: Number of fields.
- `int num_par_types` (Default=`0`)
  - Usage: Number of particle types.
- `yt_par_type* par_type_list` (Default=`NULL`)
  - Usage: Particle type list. This should be a `yt_par_type` array.
  - Data member in `yt_par_type`:
    - `const char* par_type`: Name of the particle type.
      > {octicon}`pencil;1em;sd-text-warning;` Make sure the lifetime of `par_type` covers the whole in situ process in `libyt`. `libyt` only borrows this name and does not make a copy.
    - `int num_attr`: Number of attributes this particle type has. 

## Example
```cpp
yt_param_yt param_yt;
param_yt.frontend = "gamer";                          // simulation frontend that libyt borrows field info from
param_yt.fig_basename = "FigName";                    // figure base name (default=Fig)
param_yt.length_unit = 3.0857e21;                     // length unit (cm)
param_yt.mass_unit = 1.9885e33;                       // mass unit (g)
param_yt.time_unit = 3.1557e13;                       // time unit (sec)
param_yt.current_time = time;                         // simulation time in code units
param_yt.dimensionality = 3;                          // dimensionality, support 3 only
param_yt.refine_by = REFINE_BY;                       // refinement factor between a grid and its subgrid
param_yt.num_grids = num_grids;                       // number of grids
param_yt.num_grids_local = num_grids_local;           // number of local grids
param_yt.num_fields = num_fields + 1;                 // number of fields, addition one for derived field demo
param_yt.num_par_types = num_par_types;               // number of particle types

yt_par_type par_type_list[num_par_types];
par_type_list[0].par_type = "io";
par_type_list[0].num_attr = 4;
param_yt.par_type_list = par_type_list;               // define name and number of attributes in each particle

for (int d = 0; d < 3; d++) {
    param_yt.domain_dimensions[d] = NGRID_1D * GRID_DIM; // domain dimensions in [x][y][z]
    param_yt.domain_left_edge[d] = 0.0;                  // domain left edge in [x][y][z]
    param_yt.domain_right_edge[d] = box_size;            // domain right edge in [x][y][z]
    param_yt.periodicity[d] = 0;                         // periodicity in [x][y][z]
}

param_yt.cosmological_simulation = 0;                 // if this is a cosmological simulation or not, 0 for false
param_yt.current_redshift = 0.5;                      // current redshift
param_yt.omega_lambda = 0.7;                          // omega lambda
param_yt.omega_matter = 0.3;                          // omega matter
param_yt.hubble_constant = 0.7;                       // hubble constant

if (yt_set_Parameters(&param_yt) != YT_SUCCESS) {
    fprintf(stderr, "ERROR: yt_set_Parameters() failed!\n");
    exit(EXIT_FAILURE);
}
```
