# Using yt

## Requirements
- Python package [`yt`](https://yt-project.org/) and [`yt_libyt`](https://github.com/data-exp-lab/yt_libyt).
- If it is for parallel computing in **parallel mode** ([`-DSERIAL_MODE=OFF`](../how-to-install/details.md#-dserial_mode-off)), it needs [`mpi4py`](https://mpi4py.readthedocs.io/en/stable/install.html#installation).

## Use yt for Parallel Computing
> {octicon}`info;1em;sd-text-info;` `libyt` directly borrows parallel computation feature in `yt` using `mpi4py`. Please refer to [**Parallel Computation With yt**](https://yt-project.org/doc/analyzing/parallel_computation.html#parallel-computation-with-yt).

We should always include the first three lines, then wrap the other statements inside Python functions, 
so that we can call these functions to do in situ analysis during simulation runtime. (See [Calling Python Functions](../libyt-api/run-python-function.md#yt_run_function-yt_run_functionarguments----call-python-function).)

Because we now load data directly from `libyt`, we need to replace `yt.load()` to `yt_libyt.libytDataset()`.
Everything else stays the same.

For example, the function `yt_inline` plots a density projection plot.
```{code-block} python
:lineno-start: 1
:emphasize-lines: 1, 2, 3, 6

import yt_libyt                   # import libyt's yt frontend
import yt                         # import yt
#yt.enable_parallelism()          # make yt works in parallel computing (require mpi4py)

def yt_inline():
    ds = yt_libyt.libytDataset()  # <--> yt.load("Data")
    proj = yt.ProjectionPlot(ds, "x", ("gas", "density"))
    if yt.is_root():
        proj.save()
```

## Supported yt Functionalities
These are the functions we have tested.
Basically, everything will work in Python under parallel computation using `mpi4py`. 

|       `yt` Function      | Supported | Notes                                                               |
|:------------------------:|:---------:|---------------------------------------------------------------------|
| `find_max`               |     V     |                                                                     |
| `ProjectionPlot`         |     V     |                                                                     |
| `OffAxisProjectionPlot`  |     V     |                                                                     |
| `SlicePlot`              |     V     |                                                                     |
| `OffAxisSlicePlot`       |     V     |                                                                     |
| `covering_grid`          |     V     |                                                                     |
| 1D `create_profile`      |     V     |                                                                     |
| 2D `create_profile`      |     V     |                                                                     |
| `ProfilePlot`            |     V     |                                                                     |
| `PhasePlot`              |     V     |                                                                     |
| `LinePlot`               |     V     |                                                                     |
| Halo Analysis            |           | Not test yet.                                                       |
| Isocontours              |     V     |                                                                     |
| `volume_render`          |     V     | {octicon}`alert;1em;sd-text-danger;` Need even MPI processes.                                  |
| `ParticlePlot`           |     V     |                                                                     |
| `ParticleProjectionPlot` |     V     |                                                                     |
| Annotations              |     V     | {octicon}`alert;1em;sd-text-danger;` Some[^1] require `save()` be outside of `if yt.is_root()` |

Reading and accessing data is a collective operation, and it requires every MPI process to join.
If only some of the processes participate in reading data during a yt function, then the program will hang, 
because some processes are blocked at data reading stage and waiting for the other processes.

For example, `volume_render`, which has a restriction of working under even MPI processes only.
And plots with annotations `annotate_quiver`, `annotate_cquiver`, `annotate_velocity`, `annotate_line_integral_convolution`, 
`annotate_magnetic_field`, and `annotate_particles`, need to access data when saving figure. 
Which means every MPI process should run `save()`, and we have to move `save()` outside of `if yt.is_root()`.

> {octicon}`calendar;1em;sd-text-secondary;` Since there is no way we can know what data to access and build up a communication graph for data exchange beforehand,
> when accessing simulation data, `libyt` requires every process to participate in this.
> We are working on this in both `yt` and `libyt`.

## Distinguish libyt Fields and yt Fields

### libyt Fields and yt Fields
- **libyt fields** are fields loaded by `libyt`. They are fields defined inside [`yt_get_FieldsPtr`](../libyt-api/field/yt_get_fieldsptr.md#yt_get_fieldsptr) and [`yt_get_ParticlesPtr`](../libyt-api/yt_get_particlesptr.md#yt_get_particlesptr).
  Specify [`frontend`](../libyt-api/yt_set_parameters.md#yt_param_yt) and use `("frontend", "<field_name>")` to call libyt field.
- **yt fields** are fields defined in field information class (class `XXXFieldInfo`) in a yt frontend and yt built-in derived fields. `XXX` is frontend name defined in [`frontend`](../libyt-api/yt_set_parameters.md#yt_param_yt).
    
> {octicon}`info;1em;sd-text-info;` We can use both **libyt fields** and **yt fields** in in situ analysis Python script. All of them are two-component tuple, specify the whole tuple when using it in Python script. 

> {octicon}`info;1em;sd-text-info;` As a side note, we can use yt API to look up fields:
> ```python
> >>> ds = yt_libyt.libytDataset()
> >>> ds.field_list          # prints a list of libyt fields and field information class in a frontend
> >>> ds.derived_field_list  # prints a list of yt derived field
> ```

### Naming and Field Information
libyt inherits field information (ex: units, name aliases) defined in yt frontend, and it can access yt built-in derived fields.

**Ranking the priority of the field/particle information used by `libyt` from high to low**:
 - yt frontend ([`frontend`](../libyt-api/yt_set_parameters.md#yt_param_yt) set by [`yt_set_Parameters`](../libyt-api/yt_set_parameters.md#yt_set_parameters)) has the highest priority
 - Fields/particles defined through [`yt_get_FieldsPtr`](../libyt-api/field/yt_get_fieldsptr.md#yt_get_fieldsptr)/[`yt_get_ParticlesPtr`](../libyt-api/yt_get_particlesptr.md#yt_get_particlesptr), 
 - yt built-in derived fields.

**The Rule is based on**:
1. If field name `"A"` is both defined in [`yt_get_FieldsPtr`](../libyt-api/field/yt_get_fieldsptr.md#yt_get_fieldsptr)/[`yt_get_ParticlesPtr`](../libyt-api/yt_get_particlesptr.md#yt_get_particlesptr) and yt [`frontend`](../libyt-api/yt_set_parameters.md#yt_param_yt), then `yt` uses the field information (ex: units, name alias) defined in yt [`frontend`](../libyt-api/yt_set_parameters.md#yt_param_yt). (It also adds name alias defined through libyt API to this field information if there is.)
2. If field name `"B"` is only defined in [`yt_get_FieldsPtr`](../libyt-api/field/yt_get_fieldsptr.md#yt_get_fieldsptr)/[`yt_get_ParticlesPtr`](../libyt-api/yt_get_particlesptr.md#yt_get_particlesptr), then `yt` uses the information defined through libyt API.
3. If field name `"C"` defined in [`yt_get_FieldsPtr`](../libyt-api/field/yt_get_fieldsptr.md#yt_get_fieldsptr)/[`yt_get_ParticlesPtr`](../libyt-api/yt_get_particlesptr.md#yt_get_particlesptr) overlapped with yt built-in derived field (`"C"` and yt derived field have the same name), then `yt` uses `"C"` defined through libyt API. Namely, it overwrites yt derived field.

## FAQs

- [Why does my program hang and how do I solve it?](../FAQs.md#why-does-my-program-hang-and-how-do-i-solve-it)


[^1]: `annotate_quiver`, `annotate_cquiver`, `annotate_velocity`, `annotate_line_integral_convolution`, `annotate_magnetic_field`, and `annotate_particles`
