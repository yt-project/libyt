# FAQs

## How Does libyt Run Python Script?
`libyt` runs Python script synchronously, which means every MPI process runs the same piece of Python code. 
They do the job together under the process space of MPI tasks using [`mpi4py`](https://mpi4py.readthedocs.io/en/stable/index.html). 

## Have Problems when Running libyt in Parallel

If we happen to get errors related to MPI, for example:
```text
*** An error occurred in MPI_Win_attach
*** reported by process [3353411585,1]
*** on win rdma window 3
*** MPI_ERR_RMA_ATTACH: Could not attach RMA segment
*** MPI_ERRORS_ARE_FATAL (processes in this win will now abort,
***    and potentially your MPI job)
3 more processes have sent help message help-mpi-errors.txt / mpi_errors_are_fatal
Set MCA parameter "orte_base_help_aggregate" to 0 to see all help / error messages
```
Remember to set `OMPI_MCA_osc=sm,pt2pt` before running the application. It is for one-sided MPI communication. For example:

```bash
OMPI_MCA_osc=sm,pt2pt mpirun -np 3 ./example
```