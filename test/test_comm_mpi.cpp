#include <gtest/gtest.h>
#include <limits.h>

#include "big_mpi.h"
#include "comm_mpi.h"

class CommMPIFixture : public testing::Test {
protected:
    CommMPI comm_mpi_;
    int mpi_size_ = 1;
    int mpi_rank_ = 0;

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
        MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank_);
        MPI_Comm_size(MPI_COMM_WORLD, &mpi_size_);
        CommMPI::InitializeYtLongMpiDataType();
        CommMPI::InitializeYtHierarchyMpiDataType();
        CommMPI::InitializeYtRmaGridInfoMpiDataType();
        CommMPI::InitializeYtRmaParticleInfoMpiDataType();
    }
};

class TestBigMPI : public CommMPIFixture {};

TEST_F(TestBigMPI, Big_MPI_Gatherv_with_yt_long) {
    // Arrange
    std::cout << "mpi_size_ = " << mpi_size_ << ", " << "mpi_rank = " << mpi_rank_ << std::endl;
    int mpi_root = 0;
    MPI_Datatype mpi_datatype = CommMPI::yt_long_mpi_type_;

    int* send_count_in_each_rank = new int[mpi_size_];
    long total_send_counts = 1000;  // TODO: make this a test parameter
    int displacement = 0;
    SplitArray(total_send_counts, mpi_size_, mpi_rank_, send_count_in_each_rank, &displacement);

    long* send_buffer = new long[send_count_in_each_rank[mpi_rank_]];
    for (int i = 0; i < send_count_in_each_rank[mpi_rank_]; i++) {
        send_buffer[i] = displacement + i;
    }

    long* recv_buffer = nullptr;
    if (mpi_rank_ == mpi_root) {
        recv_buffer = new long[total_send_counts];
    }

    // Act
    const int result =
        big_MPI_Gatherv<long>(mpi_root, send_count_in_each_rank, (void*)send_buffer, &mpi_datatype, (void*)recv_buffer);

    // Assert
    EXPECT_EQ(result, YT_SUCCESS);
    if (mpi_rank_ == mpi_root) {
        for (long i = 0; i < total_send_counts; i++) {
            EXPECT_EQ(recv_buffer[i], i);
        }
    }

    // Clean up
    delete[] send_count_in_each_rank;
    delete[] send_buffer;
    delete[] recv_buffer;
}

TEST_F(TestBigMPI, Big_MPI_Bcast_with_yt_long) {
    // Arrange
    std::cout << "mpi_size_ = " << mpi_size_ << ", " << "mpi_rank = " << mpi_rank_ << std::endl;
    int mpi_root = 0;
    MPI_Datatype mpi_datatype = CommMPI::yt_long_mpi_type_;

    const long total_send_counts = 1000;  // TODO: make this a test parameter
    long* send_buffer = new long[total_send_counts];
    if (mpi_rank_ == mpi_root) {
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

int main(int argc, char* argv[]) {
    int result = 0;

    ::testing::InitGoogleTest(&argc, argv);
    MPI_Init(&argc, &argv);
    result = RUN_ALL_TESTS();
    MPI_Finalize();

    return result;
}