#ifndef COMM_MPI_H
#define COMM_MPI_H

#include <mpi.h>

#include <map>
#include <string>

class CommMpi {
public:
    static int mpi_rank_;
    static int mpi_size_;
    static int mpi_root_;

    // TODO: probably should move initialization of these to somewhere else, or move it inside the map
    static std::map<std::string, MPI_Datatype*> mpi_custom_type_map_;
    static MPI_Datatype yt_long_mpi_type_;
    static MPI_Datatype yt_hierarchy_mpi_type_;
    static MPI_Datatype mpi_rma_address_mpi_type_;
    static MPI_Datatype amr_data_array_3d_info_mpi_type_;
    static MPI_Datatype yt_rma_grid_info_mpi_type_;
    static MPI_Datatype yt_rma_particle_info_mpi_type_;

    static void InitializeYtLongMpiDataType();
    static void InitializeYtHierarchyMpiDataType();
    static void InitializeMpiRmaAddressMpiDataType();
    static void InitializeAmrDataArray3DInfoMpiDataType();
    static void InitializeYtRmaGridInfoMpiDataType();
    static void InitializeYtRmaParticleInfoMpiDataType();

    static void InitializeInfo(int mpi_root = 0);
    static void SetAllNumGridsLocal(int* all_num_grids_local, int num_grids_local);
};

#endif  // COMM_MPI_H
