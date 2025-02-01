#include <gtest/gtest.h>

#include "data_structure_amr.h"

class DsAmrFixture : public testing::Test {
private:
    // Though mpi_ prefix is used, in serial mode, it will be rank 0 and size 1.
    int mpi_rank_ = 0;
    int mpi_size_ = 1;

protected:
    int GetMpiRank() const { return mpi_rank_; }
    int GetMpiSize() const { return mpi_size_; }
    void GenerateLocalHierarchy(long num_grids, int index_offset, yt_grid* grids_local, int num_grids_local,
                                int num_par_types) {
        // Calculate range based on mpi rank
        long start_i = GetMpiRank() * (num_grids / GetMpiSize());

        // Generate local hierarchy
        for (int i = 0; i < num_grids_local; i++) {
            long gid = start_i + i + index_offset;
            grids_local[i].id = gid;
            GetGridHierarchy(gid, index_offset, &grids_local[i].parent_id, &grids_local[i].level,
                             grids_local[i].grid_dimensions, grids_local[i].left_edge, grids_local[i].right_edge,
                             num_grids, num_par_types, grids_local[i].par_count_list, nullptr);
        }
    }
    void GetGridHierarchy(long gid, int index_offset, long* parent_id_ptr, int* level_ptr, int* grid_dim_ptr,
                          double* grid_left_edge_ptr, double* grid_right_edge_ptr, long num_grids, int num_par_types,
                          long* par_count_list = nullptr, int* proc_num_ptr = nullptr) {
        // Info for creating a grid hierarchy based on gid
        int grid_dim[3] = {10, 1, 1};
        double dx_grid = 1.0;
        double domain_left_edge[3] = {0.0, 0.0, 0.0};
        double domain_right_edge[3] = {dx_grid * (double)num_grids, dx_grid * (double)num_grids,
                                       dx_grid * (double)num_grids};

        // Generate and assign to input parameters
        *parent_id_ptr = -1;
        *level_ptr = 0;
        for (int d = 0; d < 3; d++) {
            grid_dim_ptr[d] = grid_dim[d];
            grid_left_edge_ptr[d] = domain_left_edge[d] + dx_grid * ((double)gid - index_offset);
            grid_right_edge_ptr[d] = domain_left_edge[d] + dx_grid * ((double)gid - index_offset + 1.0);
        }

        if (par_count_list != nullptr) {
            for (int p = 0; p < num_par_types; p++) {
                par_count_list[p] = 10;
            }
        }

        if (proc_num_ptr != nullptr) {
            *proc_num_ptr = ((int)gid - index_offset) / (num_grids / GetMpiSize());
            if (*proc_num_ptr == GetMpiSize()) {
                *proc_num_ptr = GetMpiSize() - 1;
            }
        }
    }
};

int main(int argc, char** argv) {
    int result = 0;
    ::testing::InitGoogleTest(&argc, argv);
    result = RUN_ALL_TESTS();
    return result;
}