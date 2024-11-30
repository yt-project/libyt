#ifndef COMM_MPI_H
#define COMM_MPI_H

#include <mpi.h>

class CommMPI {
public:
    static MPI_Datatype yt_long_mpi_type_;
    static MPI_Datatype yt_hierarchy_mpi_type_;
    static MPI_Datatype yt_rma_grid_info_mpi_type_;
    static MPI_Datatype yt_rma_particle_info_mpi_type_;

    static void InitializeYtLongMpiDataType();
    static void InitializeYtHierarchyMpiDataType();
    static void InitializeYtRmaGridInfoMpiDataType();
    static void InitializeYtRmaParticleInfoMpiDataType();
};

#endif  // COMM_MPI_H
