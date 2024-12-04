#include <gtest/gtest.h>
#include <limits.h>

#include "big_mpi.h"
#include "comm_mpi.h"
#include "comm_mpi_rma.h"
#include "data_structure_amr.h"

class CommMpiFixture : public testing::Test {
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
        CommMpi::InitializeInfo(0);
        CommMpi::InitializeYtLongMpiDataType();
        CommMpi::InitializeYtHierarchyMpiDataType();
        CommMpi::InitializeMpiRmaAddressMpiDataType();
        CommMpi::InitializeAmrDataArray3DInfoMpiDataType();
        CommMpi::InitializeYtRmaGridInfoMpiDataType();
        CommMpi::InitializeYtRmaParticleInfoMpiDataType();
    }
};

class TestBigMpi : public CommMpiFixture {};
class TestRma : public CommMpiFixture {};

TEST_F(TestBigMpi, Big_MPI_Gatherv_with_yt_long) {
    // Arrange
    int mpi_size = CommMpi::mpi_size_;
    int mpi_rank = CommMpi::mpi_rank_;
    int mpi_root = CommMpi::mpi_root_;
    std::cout << "mpi_size = " << mpi_size << ", " << "mpi_rank = " << mpi_rank << std::endl;
    MPI_Datatype mpi_datatype = CommMpi::yt_long_mpi_type_;

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

TEST_F(TestBigMpi, Big_MPI_Gatherv_with_yt_hierarchy) {
    // Arrange
    int mpi_size = CommMpi::mpi_size_;
    int mpi_rank = CommMpi::mpi_rank_;
    int mpi_root = CommMpi::mpi_root_;
    std::cout << "mpi_size = " << mpi_size << ", " << "mpi_rank = " << mpi_rank << std::endl;
    MPI_Datatype mpi_datatype = CommMpi::yt_hierarchy_mpi_type_;

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

TEST_F(TestBigMpi, big_MPI_Gatherv_with_AmrDataArray3DInfo) {
    // Arrange
    int mpi_size = CommMpi::mpi_size_;
    int mpi_rank = CommMpi::mpi_rank_;
    int mpi_root = CommMpi::mpi_root_;
    std::cout << "mpi_size = " << mpi_size << ", " << "mpi_rank = " << mpi_rank << std::endl;
    MPI_Datatype mpi_datatype = CommMpi::amr_data_array_3d_info_mpi_type_;

    int* send_count_in_each_rank = new int[mpi_size];
    long total_send_counts = 1000;  // TODO: make this a test parameter
    int displacement = 0;
    SplitArray(total_send_counts, mpi_size, mpi_rank, send_count_in_each_rank, &displacement);

    AmrDataArray3DInfo* send_buffer = new AmrDataArray3DInfo[send_count_in_each_rank[mpi_rank]];
    for (int i = 0; i < send_count_in_each_rank[mpi_rank]; i++) {
        send_buffer[i].id = displacement + i;
        send_buffer[i].data_dtype = YT_INT;
        send_buffer[i].swap_axes = true;
        for (int d = 0; d < 3; d++) {
            send_buffer[i].data_dim[d] = d;
        }
    }

    AmrDataArray3DInfo* recv_buffer = nullptr;
    if (mpi_rank == mpi_root) {
        recv_buffer = new AmrDataArray3DInfo[total_send_counts];
    }

    // Act
    const int result = big_MPI_Gatherv<AmrDataArray3DInfo>(mpi_root, send_count_in_each_rank, (void*)send_buffer,
                                                           &mpi_datatype, (void*)recv_buffer);

    // Assert
    EXPECT_EQ(result, YT_SUCCESS);
    if (mpi_rank == mpi_root) {
        for (long i = 0; i < total_send_counts; i++) {
            EXPECT_EQ(recv_buffer[i].id, i);
            EXPECT_EQ(recv_buffer[i].data_dtype, YT_INT);
            EXPECT_EQ(recv_buffer[i].swap_axes, true);
            for (int d = 0; d < 3; d++) {
                EXPECT_EQ(recv_buffer[i].data_dim[d], d);
            }
        }
    }

    // Clean up
    delete[] send_count_in_each_rank;
    delete[] send_buffer;
    delete[] recv_buffer;
}

TEST_F(TestBigMpi, big_MPI_Gatherv_with_MpiRmaAddress) {
    // Arrange
    int mpi_size = CommMpi::mpi_size_;
    int mpi_rank = CommMpi::mpi_rank_;
    int mpi_root = CommMpi::mpi_root_;
    std::cout << "mpi_size = " << mpi_size << ", " << "mpi_rank = " << mpi_rank << std::endl;
    MPI_Datatype mpi_datatype = CommMpi::mpi_rma_address_mpi_type_;

    int* send_count_in_each_rank = new int[mpi_size];
    long total_send_counts = 1000;  // TODO: make this a test parameter
    int displacement = 0;
    SplitArray(total_send_counts, mpi_size, mpi_rank, send_count_in_each_rank, &displacement);

    MpiRmaAddress* send_buffer = new MpiRmaAddress[send_count_in_each_rank[mpi_rank]];
    for (int i = 0; i < send_count_in_each_rank[mpi_rank]; i++) {
        send_buffer[i].mpi_rank = displacement + i;
        send_buffer[i].mpi_address = displacement + i + total_send_counts;
    }

    MpiRmaAddress* recv_buffer = nullptr;
    if (mpi_rank == mpi_root) {
        recv_buffer = new MpiRmaAddress[total_send_counts];
    }

    // Act
    const int result = big_MPI_Gatherv<MpiRmaAddress>(mpi_root, send_count_in_each_rank, (void*)send_buffer,
                                                      &mpi_datatype, (void*)recv_buffer);

    // Assert
    EXPECT_EQ(result, YT_SUCCESS);
    if (mpi_rank == mpi_root) {
        for (long i = 0; i < total_send_counts; i++) {
            EXPECT_EQ(recv_buffer[i].mpi_rank, i);
            EXPECT_EQ(recv_buffer[i].mpi_address, i + total_send_counts);
        }
    }

    // Clean up
    delete[] send_count_in_each_rank;
    delete[] send_buffer;
    delete[] recv_buffer;
}

TEST_F(TestBigMpi, Big_MPI_Bcast_with_yt_long) {
    // Arrange
    int mpi_size = CommMpi::mpi_size_;
    int mpi_rank = CommMpi::mpi_rank_;
    int mpi_root = CommMpi::mpi_root_;
    std::cout << "mpi_size = " << mpi_size << ", " << "mpi_rank = " << mpi_rank << std::endl;
    MPI_Datatype mpi_datatype = CommMpi::yt_long_mpi_type_;

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

TEST_F(TestBigMpi, Big_MPI_Bcast_with_yt_hierarchy) {
    // Arrange
    int mpi_size = CommMpi::mpi_size_;
    int mpi_rank = CommMpi::mpi_rank_;
    int mpi_root = CommMpi::mpi_root_;
    std::cout << "mpi_size = " << mpi_size << ", " << "mpi_rank = " << mpi_rank << std::endl;
    MPI_Datatype mpi_datatype = CommMpi::yt_hierarchy_mpi_type_;

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

TEST_F(TestBigMpi, big_MPI_Bcast_with_AmrDataArray3DInfo) {
    // Arrange
    int mpi_size = CommMpi::mpi_size_;
    int mpi_rank = CommMpi::mpi_rank_;
    int mpi_root = CommMpi::mpi_root_;
    std::cout << "mpi_size = " << mpi_size << ", " << "mpi_rank = " << mpi_rank << std::endl;
    MPI_Datatype mpi_datatype = CommMpi::amr_data_array_3d_info_mpi_type_;

    const long total_send_counts = 1000;  // TODO: make this a test parameter
    AmrDataArray3DInfo* send_buffer = new AmrDataArray3DInfo[total_send_counts];
    if (mpi_rank == mpi_root) {
        for (int i = 0; i < total_send_counts; i++) {
            send_buffer[i].id = i;
            send_buffer[i].data_dtype = YT_INT;
            send_buffer[i].swap_axes = true;
            for (int d = 0; d < 3; d++) {
                send_buffer[i].data_dim[d] = d;
            }
        }
    }

    // Act
    const int result =
        big_MPI_Bcast<AmrDataArray3DInfo>(mpi_root, total_send_counts, (void*)send_buffer, &mpi_datatype);

    // Assert
    EXPECT_EQ(result, YT_SUCCESS);
    for (int i = 0; i < total_send_counts; i++) {
        EXPECT_EQ(send_buffer[i].id, i);
        EXPECT_EQ(send_buffer[i].data_dtype, YT_INT);
        EXPECT_EQ(send_buffer[i].swap_axes, true);
        for (int d = 0; d < 3; d++) {
            EXPECT_EQ(send_buffer[i].data_dim[d], d);
        }
    }

    // Clean up
    delete[] send_buffer;
}

TEST_F(TestBigMpi, big_MPI_Bcast_with_MpiRmaAddress) {
    // Arrange
    int mpi_size = CommMpi::mpi_size_;
    int mpi_rank = CommMpi::mpi_rank_;
    int mpi_root = CommMpi::mpi_root_;
    std::cout << "mpi_size = " << mpi_size << ", " << "mpi_rank = " << mpi_rank << std::endl;
    MPI_Datatype mpi_datatype = CommMpi::mpi_rma_address_mpi_type_;

    const long total_send_counts = 1000;  // TODO: make this a test parameter
    MpiRmaAddress* send_buffer = new MpiRmaAddress[total_send_counts];
    if (mpi_rank == mpi_root) {
        for (int i = 0; i < total_send_counts; i++) {
            send_buffer[i].mpi_rank = i;
            send_buffer[i].mpi_address = i + total_send_counts;
        }
    }

    // Act
    const int result = big_MPI_Bcast<MpiRmaAddress>(mpi_root, total_send_counts, (void*)send_buffer, &mpi_datatype);

    // Assert
    EXPECT_EQ(result, YT_SUCCESS);
    for (int i = 0; i < total_send_counts; i++) {
        EXPECT_EQ(send_buffer[i].mpi_rank, i);
        EXPECT_EQ(send_buffer[i].mpi_address, i + total_send_counts);
    }

    // Clean up
    delete[] send_buffer;
}

TEST(Function, SetAllNumGridsLocal_can_work) {
    std::cout << "mpi_size = " << CommMpi::mpi_size_ << ", " << "mpi_rank = " << CommMpi::mpi_rank_ << std::endl;
    // Arrange
    int num_grids_local = CommMpi::mpi_rank_;
    int* all_num_grids_local = new int[CommMpi::mpi_size_];

    // Act
    CommMpi::SetAllNumGridsLocal(all_num_grids_local, num_grids_local);

    // Assert
    for (int r = 0; r < CommMpi::mpi_size_; r++) {
        EXPECT_EQ(all_num_grids_local[r], r);
    }

    // Clean up
    delete[] all_num_grids_local;
}

TEST_F(TestRma, CommMpiRma_with_AmrDataArray3D_can_work) {
    std::cout << "mpi_size = " << CommMpi::mpi_size_ << ", " << "mpi_rank = " << CommMpi::mpi_rank_ << std::endl;
    // Arrange
    std::vector<AmrDataArray3D> prepared_data_list;
    std::vector<FetchedFromInfo> fetch_id_list;

    // Create data buffer with array values and id equal to mpi rank
    int* data_buffer = new int[10];
    for (int i = 0; i < 10; i++) {
        data_buffer[i] = CommMpi::mpi_rank_;
    }
    prepared_data_list.emplace_back(AmrDataArray3D{CommMpi::mpi_rank_, YT_INT, {10, 1, 1}, false, data_buffer});

    // Create fetch id list which gets the other mpi rank's data
    for (int r = 0; r < CommMpi::mpi_size_; r++) {
        if (r != CommMpi::mpi_rank_) {
            fetch_id_list.emplace_back(FetchedFromInfo{r, r});
        }
    }

    // This fails hard. (edge case)
    // prepared_data_list.emplace_back(AMRFieldDataArray3D{CommMPI::mpi_rank_, YT_INT, {1, 1, 1}, false, nullptr});

    // Act
    CommMpiRma<AmrDataArray3DInfo, AmrDataArray3D> comm_mpi_rma("test", "amr_grid");
    CommMpiRmaReturn<AmrDataArray3D> result = comm_mpi_rma.GetRemoteData(prepared_data_list, fetch_id_list);

    // Assert
    EXPECT_EQ(result.status, CommMpiRmaStatus::kMpiSuccess) << "Error: " << comm_mpi_rma.GetErrorStr();
    for (const AmrDataArray3D& fetched_data : result.data_list) {
        int dim[3] = {fetched_data.data_dim[0], fetched_data.data_dim[1], fetched_data.data_dim[2]};
        EXPECT_EQ(((int*)fetched_data.data_ptr)[dim[0] * dim[1] * dim[2] - 1], fetched_data.id);
    }

    // Clean up
    delete[] data_buffer;
    for (const AmrDataArray3D& fetched_data : result.data_list) {
        delete[] fetched_data.data_ptr;
    }
}

int main(int argc, char* argv[]) {
    int result = 0;

    ::testing::InitGoogleTest(&argc, argv);
    MPI_Init(&argc, &argv);
    result = RUN_ALL_TESTS();
    MPI_Finalize();

    return result;
}