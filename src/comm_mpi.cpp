#ifndef SERIAL_MODE
#include "comm_mpi.h"

#include "timer.h"

int CommMPI::mpi_rank_ = 0;
int CommMPI::mpi_size_ = 1;
int CommMPI::mpi_root_ = 0;

MPI_Datatype CommMPI::yt_long_mpi_type_;
MPI_Datatype CommMPI::yt_hierarchy_mpi_type_;
MPI_Datatype CommMPI::yt_rma_grid_info_mpi_type_;
MPI_Datatype CommMPI::yt_rma_particle_info_mpi_type_;

void CommMPI::InitializeInfo(int mpi_root) {
    SET_TIMER(__PRETTY_FUNCTION__);

    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank_);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size_);
    mpi_root_ = mpi_root;
}

void CommMPI::InitializeYtLongMpiDataType() {
    SET_TIMER(__PRETTY_FUNCTION__);

    int length[1] = {1};
    const MPI_Aint displacements[1] = {0};
    MPI_Datatype types[1] = {MPI_LONG};
    MPI_Type_create_struct(1, length, displacements, types, &yt_long_mpi_type_);
    MPI_Type_commit(&yt_long_mpi_type_);
}

void CommMPI::InitializeYtHierarchyMpiDataType() {
    SET_TIMER(__PRETTY_FUNCTION__);

    int lengths[7] = {3, 3, 1, 1, 3, 1, 1};
    const MPI_Aint displacements[7] = {0,
                                       3 * sizeof(double),
                                       6 * sizeof(double),
                                       6 * sizeof(double) + sizeof(long),
                                       6 * sizeof(double) + 2 * sizeof(long),
                                       6 * sizeof(double) + 2 * sizeof(long) + 3 * sizeof(int),
                                       6 * sizeof(double) + 2 * sizeof(long) + 4 * sizeof(int)};
    MPI_Datatype types[7] = {MPI_DOUBLE, MPI_DOUBLE, MPI_LONG, MPI_LONG, MPI_INT, MPI_INT, MPI_INT};
    MPI_Type_create_struct(7, lengths, displacements, types, &yt_hierarchy_mpi_type_);
    MPI_Type_commit(&yt_hierarchy_mpi_type_);
}

void CommMPI::InitializeYtRmaGridInfoMpiDataType() {
    SET_TIMER(__PRETTY_FUNCTION__);

    int lengths[5] = {1, 1, 1, 1, 3};
    const MPI_Aint displacements[5] = {0, 1 * sizeof(long), 1 * sizeof(long) + 1 * sizeof(MPI_Aint),
                                       1 * sizeof(long) + 1 * sizeof(MPI_Aint) + 1 * sizeof(int),
                                       1 * sizeof(long) + 1 * sizeof(MPI_Aint) + 2 * sizeof(int)};
    MPI_Datatype types[5] = {MPI_LONG, MPI_AINT, MPI_INT, MPI_INT, MPI_INT};
    MPI_Type_create_struct(5, lengths, displacements, types, &yt_rma_grid_info_mpi_type_);
    MPI_Type_commit(&yt_rma_grid_info_mpi_type_);
}

void CommMPI::InitializeYtRmaParticleInfoMpiDataType() {
    SET_TIMER(__PRETTY_FUNCTION__);

    int lengths[4] = {1, 1, 1, 1};
    const MPI_Aint displacements[4] = {0, 1 * sizeof(long), 1 * sizeof(long) + 1 * sizeof(MPI_Aint),
                                       2 * sizeof(long) + 1 * sizeof(MPI_Aint)};
    MPI_Datatype types[4] = {MPI_LONG, MPI_AINT, MPI_LONG, MPI_INT};
    MPI_Type_create_struct(4, lengths, displacements, types, &yt_rma_particle_info_mpi_type_);
    MPI_Type_commit(&yt_rma_particle_info_mpi_type_);
}

#endif
