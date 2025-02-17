import libyt

try:
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    my_rank = comm.Get_rank()
    my_size = comm.Get_size()
except ImportError:
    my_rank = 0
    my_size = 1


def test_derived_function():
    start_id = my_rank * int(libyt.param_yt["num_grids"] / my_size)
    end_id = (my_rank + 1) * int(libyt.param_yt["num_grids"] / my_size)
    if my_rank == my_size - 1:
        end_id = libyt.param_yt["num_grids"]

    query_data = list(range(start_id, start_id + 4000))

    for i in query_data:
        data = libyt.derived_func(i, "DerivedOnes")


def test_rma_function():
    start_id_list = [r * int(libyt.param_yt["num_grids"] / my_size) for r in range(my_size)]

    data_len = 3000
    my_data = list(range(start_id_list[my_rank], start_id_list[my_rank] + data_len))
    query_data = []
    data_location = []
    for r in range(my_size):
        if r != my_rank:
            query_data += list(range(start_id_list[r], start_id_list[r] + data_len))
            data_location += [r] * data_len

    for i in range(100):
        data = libyt.get_field_remote(["CCTwos".encode(encoding="UTF-8", errors="strict")], len(my_data), my_data,
                                      len(query_data), query_data, data_location,
                                      len(data_location))
