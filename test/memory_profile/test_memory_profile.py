import libyt

try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    my_rank = comm.Get_rank()
    my_size = comm.Get_size()

except ImportError:
    my_rank = 0
    my_size = 1

def yt_inline():
    start_id = my_rank * int(libyt.param_yt["num_grids"] / my_size)
    end_id = (my_rank + 1) * int(libyt.param_yt["num_grids"] / my_size)
    if my_rank == my_size - 1:
        end_id = libyt.param_yt["num_grids"]

    query_data = list(range(start_id, start_id + 6000))

    for i in query_data:
        data = libyt.derived_func(i, "DerivedOnes")