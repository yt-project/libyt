---
layout: default
title: Limitations
parent: In Situ Python Analysis
nav_order: 7
---
# Limitations
{: .no_toc }
<details open markdown="block">
  <summary>
    Table of contents
  </summary>
  {: .text-delta }
- TOC
{:toc}
</details>
---

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

> :information_source: When accessing simulation data, `libyt` requires every process to participate (See [`libyt.get_field_remote`]({% link InSituPythonAnalysis/libytPythonModule.md %}#get_field_remotefname_list--list-fname_list_len--int-prepare_list--list-prepare_list_len--int-fetch_gid_list--list-fetch_process_list--list-fetch_gid_list_len--int---dict) and [`libyt.get_particle_remote`]({% link InSituPythonAnalysis/libytPythonModule.md %}#get_particle_remotepar_dict--dict-par_dict--dict_keys-prepare_list--list-prepare_list_len--int-fetch_gid_list--list-fetch_process_list--list-fetch_gid_list_len--int---dict).