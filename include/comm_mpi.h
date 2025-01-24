#ifndef LIBYT_PROJECT_INCLUDE_COMM_MPI_H_
#define LIBYT_PROJECT_INCLUDE_COMM_MPI_H_

#include <mpi.h>

#include <map>
#include <string>
#include <vector>

class CommMpi {
public:
    static int mpi_rank_;
    static int mpi_size_;
    static int mpi_root_;

    // TODO: probably should move initialization of these to somewhere else, or move it inside the map
    static MPI_Datatype yt_long_mpi_type_;

    static void InitializeYtLongMpiDataType();

    static void InitializeInfo(int mpi_root = 0);
    static void SetAllNumGridsLocal(int* all_num_grids_local, int num_grids_local);
    static int CheckAllStates(int local_state, int desired_state, int success_value, int failure_value);
    static void SetStringUsingValueOnRank(std::string& sync_string, int src_mpi_rank);
    static void GatherAllStringsToRank(std::vector<std::string>& all_strings, const std::string& src_string,
                                       int dest_mpi_rank);
    static void GatherAllIntsToRank(std::vector<int>& all_ints, const int src_int, int dest_mpi_rank);
};

#endif  // LIBYT_PROJECT_INCLUDE_COMM_MPI_H_
