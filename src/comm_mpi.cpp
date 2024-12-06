#ifndef SERIAL_MODE
#include "comm_mpi.h"

#include "data_structure_amr.h"
#include "timer.h"

int CommMpi::mpi_rank_ = 0;
int CommMpi::mpi_size_ = 1;
int CommMpi::mpi_root_ = 0;

std::map<std::string, MPI_Datatype*> CommMpi::mpi_custom_type_map_;
MPI_Datatype CommMpi::yt_long_mpi_type_;
MPI_Datatype CommMpi::yt_hierarchy_mpi_type_;
MPI_Datatype CommMpi::mpi_rma_address_mpi_type_;
MPI_Datatype CommMpi::amr_data_array_3d_mpi_type_;
MPI_Datatype CommMpi::amr_data_array_1d_mpi_type_;
MPI_Datatype CommMpi::yt_rma_grid_info_mpi_type_;
MPI_Datatype CommMpi::yt_rma_particle_info_mpi_type_;

void CommMpi::InitializeInfo(int mpi_root) {
    SET_TIMER(__PRETTY_FUNCTION__);

    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank_);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size_);
    mpi_root_ = mpi_root;
}

void CommMpi::InitializeYtLongMpiDataType() {
    SET_TIMER(__PRETTY_FUNCTION__);

    int length[1] = {1};
    const MPI_Aint displacements[1] = {0};
    MPI_Datatype types[1] = {MPI_LONG};
    MPI_Type_create_struct(1, length, displacements, types, &yt_long_mpi_type_);
    MPI_Type_commit(&yt_long_mpi_type_);
}

void CommMpi::InitializeYtHierarchyMpiDataType() {
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

void CommMpi::InitializeMpiRmaAddressMpiDataType() {
    SET_TIMER(__PRETTY_FUNCTION__);

    int lengths[2] = {1, 1};
    const MPI_Aint displacements[2] = {0, 1 * sizeof(MPI_Aint)};
    MPI_Datatype types[2] = {MPI_AINT, MPI_INT};
    MPI_Type_create_struct(2, lengths, displacements, types, &mpi_rma_address_mpi_type_);
    MPI_Type_commit(&mpi_rma_address_mpi_type_);
}

void CommMpi::InitializeAmrDataArray3DMpiDataType() {
    SET_TIMER(__PRETTY_FUNCTION__);

    // Though we won't use the void* pointer in other rank, we still pass it for consistency.
    // Since it is passed as struct, we need to set the memory layout that covers the void* pointer size
    static_assert(sizeof(void*) == sizeof(long), "sizeof(void*) and sizeof(long) have difference size");

    int lengths[5] = {1, 1, 3, 1, 1};
    const MPI_Aint displacements[5] = {
        0,
        1 * sizeof(long),
        1 * sizeof(long) + 1 * sizeof(int),
        1 * sizeof(long) + 4 * sizeof(int),
        2 * sizeof(long) + 4 * sizeof(int),
    };
    MPI_Datatype types[5] = {MPI_LONG, MPI_INT, MPI_INT, MPI_LONG, MPI_CXX_BOOL};
    MPI_Type_create_struct(5, lengths, displacements, types, &amr_data_array_3d_mpi_type_);
    MPI_Type_commit(&amr_data_array_3d_mpi_type_);

    mpi_custom_type_map_["amr_grid"] = &amr_data_array_3d_mpi_type_;
}

void CommMpi::InitializeAmrDataArray1DMpiDataType() {
    SET_TIMER(__PRETTY_FUNCTION__);

    AmrDataArray1D amr_1d{};

    int lengths[4] = {1, 1, 1, 1};
    MPI_Aint displacements[4];
    displacements[0] = offsetof(AmrDataArray1D, id);
    displacements[1] = offsetof(AmrDataArray1D, data_dtype);
    displacements[2] = offsetof(AmrDataArray1D, data_ptr);
    displacements[3] = offsetof(AmrDataArray1D, data_len);
    MPI_Datatype types[4] = {MPI_LONG, MPI_INT, MPI_AINT, MPI_LONG};
    MPI_Type_create_struct(4, lengths, displacements, types, &amr_data_array_1d_mpi_type_);
    MPI_Type_commit(&amr_data_array_1d_mpi_type_);

    mpi_custom_type_map_["amr_particle"] = &amr_data_array_1d_mpi_type_;
}

void CommMpi::InitializeYtRmaGridInfoMpiDataType() {
    SET_TIMER(__PRETTY_FUNCTION__);

    int lengths[5] = {1, 1, 1, 1, 3};
    const MPI_Aint displacements[5] = {0, 1 * sizeof(long), 1 * sizeof(long) + 1 * sizeof(MPI_Aint),
                                       1 * sizeof(long) + 1 * sizeof(MPI_Aint) + 1 * sizeof(int),
                                       1 * sizeof(long) + 1 * sizeof(MPI_Aint) + 2 * sizeof(int)};
    MPI_Datatype types[5] = {MPI_LONG, MPI_AINT, MPI_INT, MPI_INT, MPI_INT};
    MPI_Type_create_struct(5, lengths, displacements, types, &yt_rma_grid_info_mpi_type_);
    MPI_Type_commit(&yt_rma_grid_info_mpi_type_);
}

void CommMpi::InitializeYtRmaParticleInfoMpiDataType() {
    SET_TIMER(__PRETTY_FUNCTION__);

    int lengths[4] = {1, 1, 1, 1};
    const MPI_Aint displacements[4] = {0, 1 * sizeof(long), 1 * sizeof(long) + 1 * sizeof(MPI_Aint),
                                       2 * sizeof(long) + 1 * sizeof(MPI_Aint)};
    MPI_Datatype types[4] = {MPI_LONG, MPI_AINT, MPI_LONG, MPI_INT};
    MPI_Type_create_struct(4, lengths, displacements, types, &yt_rma_particle_info_mpi_type_);
    MPI_Type_commit(&yt_rma_particle_info_mpi_type_);
}

void CommMpi::SetAllNumGridsLocal(int* all_num_grids_local, int num_grids_local) {
    SET_TIMER(__PRETTY_FUNCTION__);

    MPI_Allgather(&num_grids_local, 1, MPI_INT, all_num_grids_local, 1, MPI_INT, MPI_COMM_WORLD);
}

#endif
