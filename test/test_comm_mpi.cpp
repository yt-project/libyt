#include <gtest/gtest.h>
#include <limits.h>

#include "amr_grid.h"
#include "big_mpi.h"
#include "comm_mpi.h"
#include "comm_mpi_rma.h"

class CommMPIFixture : public testing::Test {
protected:
    static void SplitArray(const long total_len, const int mpi_size, const int mpi_rank, int* count_in_each_rank,
                           int* displacement) {
        for (int r = 0; r < mpi_size - 1; r++) {
            count_in_each_rank[r] = total_len / mpi_size;
            if (mpi_rank > r) {
                *displacement += count_in_each_rank[r];
            }
        }
        count_in_each_rank[mpi_size - 1] = total_len - (total_len / mpi_size) * (mpi_size - 1);
    }

private:
    void SetUp() override {
        CommMPI::InitializeInfo(0);
        CommMPI::InitializeYtLongMpiDataType();
        CommMPI::InitializeYtHierarchyMpiDataType();
        CommMPI::InitializeYtRmaGridInfoMpiDataType();
        CommMPI::InitializeYtRmaParticleInfoMpiDataType();
    }
};

class TestBigMPI : public CommMPIFixture {};

TEST_F(TestBigMPI, Big_MPI_Gatherv_with_yt_long) {
    // Arrange
    int mpi_size = CommMPI::mpi_size_;
    int mpi_rank = CommMPI::mpi_rank_;
    int mpi_root = CommMPI::mpi_root_;
    std::cout << "mpi_size = " << mpi_size << ", " << "mpi_rank = " << mpi_rank << std::endl;
    MPI_Datatype mpi_datatype = CommMPI::yt_long_mpi_type_;

    int* send_count_in_each_rank = new int[mpi_size];
    long total_send_counts = 1000;  // TODO: make this a test parameter
    int displacement = 0;
    SplitArray(total_send_counts, mpi_size, mpi_rank, send_count_in_each_rank, &displacement);

    long* send_buffer = new long[send_count_in_each_rank[mpi_rank]];
    for (int i = 0; i < send_count_in_each_rank[mpi_rank]; i++) {
        send_buffer[i] = displacement + i;
    }

    long* recv_buffer = nullptr;
    if (mpi_rank == mpi_root) {
        recv_buffer = new long[total_send_counts];
    }

    // Act
    const int result =
        big_MPI_Gatherv<long>(mpi_root, send_count_in_each_rank, (void*)send_buffer, &mpi_datatype, (void*)recv_buffer);

    // Assert
    EXPECT_EQ(result, YT_SUCCESS);
    if (mpi_rank == mpi_root) {
        for (long i = 0; i < total_send_counts; i++) {
            EXPECT_EQ(recv_buffer[i], i);
        }
    }

    // Clean up
    delete[] send_count_in_each_rank;
    delete[] send_buffer;
    delete[] recv_buffer;
}

TEST_F(TestBigMPI, Big_MPI_Gatherv_with_yt_hierarchy) {
    // Arrange
    int mpi_size = CommMPI::mpi_size_;
    int mpi_rank = CommMPI::mpi_rank_;
    int mpi_root = CommMPI::mpi_root_;
    std::cout << "mpi_size = " << mpi_size << ", " << "mpi_rank = " << mpi_rank << std::endl;
    MPI_Datatype mpi_datatype = CommMPI::yt_hierarchy_mpi_type_;

    int* send_count_in_each_rank = new int[mpi_size];
    long total_send_counts = 1000;  // TODO: make this a test parameter
    int displacement = 0;
    SplitArray(total_send_counts, mpi_size, mpi_rank, send_count_in_each_rank, &displacement);

    yt_hierarchy* send_buffer = new yt_hierarchy[send_count_in_each_rank[mpi_rank]];
    for (int i = 0; i < send_count_in_each_rank[mpi_rank]; i++) {
        send_buffer[i].id = displacement + i;
        send_buffer[i].parent_id = displacement + i;
        send_buffer[i].level = displacement + i;
        send_buffer[i].proc_num = displacement + i;
        for (int d = 0; d < 3; d++) {
            send_buffer[i].left_edge[d] = d;
            send_buffer[i].right_edge[d] = d;
            send_buffer[i].dimensions[d] = d;
        }
    }

    yt_hierarchy* recv_buffer = nullptr;
    if (mpi_rank == mpi_root) {
        recv_buffer = new yt_hierarchy[total_send_counts];
    }

    // Act
    const int result = big_MPI_Gatherv<yt_hierarchy>(mpi_root, send_count_in_each_rank, (void*)send_buffer,
                                                     &mpi_datatype, (void*)recv_buffer);

    // Assert
    EXPECT_EQ(result, YT_SUCCESS);
    if (mpi_rank == mpi_root) {
        for (long i = 0; i < total_send_counts; i++) {
            EXPECT_EQ(recv_buffer[i].id, i);
            EXPECT_EQ(recv_buffer[i].parent_id, i);
            EXPECT_EQ(recv_buffer[i].level, i);
            EXPECT_EQ(recv_buffer[i].proc_num, i);
            for (int d = 0; d < 3; d++) {
                EXPECT_EQ(recv_buffer[i].left_edge[d], d);
                EXPECT_EQ(recv_buffer[i].right_edge[d], d);
                EXPECT_EQ(recv_buffer[i].dimensions[d], d);
            }
        }
    }

    // Clean up
    delete[] send_count_in_each_rank;
    delete[] send_buffer;
    delete[] recv_buffer;
}

TEST_F(TestBigMPI, Big_MPI_Bcast_with_yt_long) {
    // Arrange
    int mpi_size = CommMPI::mpi_size_;
    int mpi_rank = CommMPI::mpi_rank_;
    int mpi_root = CommMPI::mpi_root_;
    std::cout << "mpi_size = " << mpi_size << ", " << "mpi_rank = " << mpi_rank << std::endl;
    MPI_Datatype mpi_datatype = CommMPI::yt_long_mpi_type_;

    const long total_send_counts = 1000;  // TODO: make this a test parameter
    long* send_buffer = new long[total_send_counts];
    if (mpi_rank == mpi_root) {
        for (int i = 0; i < total_send_counts; i++) {
            send_buffer[i] = i;
        }
    }

    // Act
    const int result = big_MPI_Bcast<long>(mpi_root, total_send_counts, (void*)send_buffer, &mpi_datatype);

    // Assert
    EXPECT_EQ(result, YT_SUCCESS);
    for (long i = 0; i < total_send_counts; i++) {
        EXPECT_EQ(send_buffer[i], i);
    }

    // Clean up
    delete[] send_buffer;
}

TEST_F(TestBigMPI, Big_MPI_Bcast_with_yt_hierarchy) {
    // Arrange
    int mpi_size = CommMPI::mpi_size_;
    int mpi_rank = CommMPI::mpi_rank_;
    int mpi_root = CommMPI::mpi_root_;
    std::cout << "mpi_size = " << mpi_size << ", " << "mpi_rank = " << mpi_rank << std::endl;
    MPI_Datatype mpi_datatype = CommMPI::yt_hierarchy_mpi_type_;

    const long total_send_counts = 1000;  // TODO: make this a test parameter
    yt_hierarchy* send_buffer = new yt_hierarchy[total_send_counts];
    if (mpi_rank == mpi_root) {
        for (int i = 0; i < total_send_counts; i++) {
            send_buffer[i].id = i;
            send_buffer[i].parent_id = i;
            send_buffer[i].level = i;
            send_buffer[i].proc_num = i;
            for (int d = 0; d < 3; d++) {
                send_buffer[i].left_edge[d] = d;
                send_buffer[i].right_edge[d] = d;
                send_buffer[i].dimensions[d] = d;
            }
        }
    }

    // Act
    const int result = big_MPI_Bcast<yt_hierarchy>(mpi_root, total_send_counts, (void*)send_buffer, &mpi_datatype);

    // Assert
    EXPECT_EQ(result, YT_SUCCESS);
    for (int i = 0; i < total_send_counts; i++) {
        EXPECT_EQ(send_buffer[i].id, i);
        EXPECT_EQ(send_buffer[i].parent_id, i);
        EXPECT_EQ(send_buffer[i].level, i);
        EXPECT_EQ(send_buffer[i].proc_num, i);
        for (int d = 0; d < 3; d++) {
            EXPECT_EQ(send_buffer[i].left_edge[d], d);
            EXPECT_EQ(send_buffer[i].right_edge[d], d);
            EXPECT_EQ(send_buffer[i].dimensions[d], d);
        }
    }

    // Clean up
    delete[] send_buffer;
}

TEST(Function, SetAllNumGridsLocal_can_work) {
    // Arrange
    int num_grids_local = CommMPI::mpi_rank_;
    int* all_num_grids_local = new int[CommMPI::mpi_size_];

    // Act
    CommMPI::SetAllNumGridsLocal(all_num_grids_local, num_grids_local);

    // Assert
    for (int r = 0; r < CommMPI::mpi_size_; r++) {
        EXPECT_EQ(all_num_grids_local[r], r);
    }

    // Clean up
    delete[] all_num_grids_local;
}

TEST(RMA, CommMPIRma_can_work) {
    CommMPIRma<AMRFieldDataArray3DInfo, AMRFieldDataArray3D> comm_mpi_rma("test");
    std::pair<CommMPIRmaStatus, const std::vector<AMRFieldDataArray3D>&> result =
        comm_mpi_rma.GetRemoteData(std::vector<AMRFieldDataArray3D>(), std::vector<FetchedFromInfo>());

    EXPECT_EQ(result.first, CommMPIRmaStatus::kMPISuccess);
}

int main(int argc, char* argv[]) {
    int result = 0;

    ::testing::InitGoogleTest(&argc, argv);
    MPI_Init(&argc, &argv);
    result = RUN_ALL_TESTS();
    MPI_Finalize();

    return result;
}