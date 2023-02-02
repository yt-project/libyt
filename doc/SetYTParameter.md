# Set `yt` Parameter
## yt_set_Parameters
```cpp
int yt_set_Parameters( yt_param_yt *param_yt )
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
- `num_par_types` (Default=`0`)
  - Usage: Number of particle types.
- `par_type_list` (Default=`NULL`)
  - Usage: Particle type list. This should be a `yt_par_type` array. The lifespan of the elements in this array should at least cover in situ analysis process, which is when [Perform Inline-Analysis](./PerformInlineAnalysis.md).
  - Valid Value: Each element in `yt_par_type` array
    - `par_type`: Name of the particle type.
    - `num_attr`: Number of attributes this particle type has. 
- `num_grids_local`
  - Usage: Number of local grids store on this rank now.

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
