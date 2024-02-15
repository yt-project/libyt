# FAQs

## How Does libyt Run Python Script?
`libyt` runs Python script synchronously, which means every MPI process runs the same piece of Python code. 
They do the job together under the process space of MPI tasks using [`mpi4py`](https://mpi4py.readthedocs.io/en/stable/index.html). 
