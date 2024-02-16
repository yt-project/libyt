# Limitations

## Limitations in MPI Related Python Tasks
Please avoid using MPI in such a way that reaches a dead end.

For example, this causes the program to hang, which is fatal, because root process is blocking and waiting for the other processes to join.
```python
from mpi4py import MPI
if MPI.COMM_WORLD.Get_rank() == 0:
    MPI.COMM_WORLD.Barrier()
else:
    pass
```

> {octicon}`info;1em;sd-text-info;` When accessing simulation data, `libyt` requires every process to participate (See [`libyt.get_field_remote`](./libyt-python-module.md#get-field-remote) and [`libyt.get_particle_remote`](./libyt-python-module.md#get-particle-remote).
