---
layout: default
title: FAQs
parent: In Situ Python Analysis
nav_order: 7
---
# Frequently Asked Questions
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

## How Does libyt Run Python Script?
`libyt` runs Python script synchronously, which means every MPI process runs the same piece of Python code. 
They do the job together under the process space of MPI tasks using [`mpi4py`](https://mpi4py.readthedocs.io/en/stable/index.html). 

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

> :information_source: When accessing simulation data, `libyt` requires every process to participate in this.
