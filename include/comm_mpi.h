#ifndef COMM_MPI_H
#define COMM_MPI_H

#include <mpi.h>

class CommMPI {
public:
    static int mpi_rank_;
    static int mpi_size_;
    static int mpi_root_;

    static MPI_Datatype yt_long_mpi_type_;
    static MPI_Datatype yt_hierarchy_mpi_type_;
    static MPI_Datatype mpi_rma_amr_data_array_3d_info_mpi_type_;
    static MPI_Datatype yt_rma_grid_info_mpi_type_;
    static MPI_Datatype yt_rma_particle_info_mpi_type_;

    static void InitializeInfo(int mpi_root = 0);
    static void SetAllNumGridsLocal(int* all_num_grids_local, int num_grids_local);

    // TODO: probably should move initialization of these to somewhere else, or move it inside the map
    static void InitializeYtLongMpiDataType();
    static void InitializeYtHierarchyMpiDataType();
    static void InitializeMPiRmaAmrDataArray3dInfoMpiDataType();
    static void InitializeYtRmaGridInfoMpiDataType();
    static void InitializeYtRmaParticleInfoMpiDataType();
};

#endif  // COMM_MPI_H
