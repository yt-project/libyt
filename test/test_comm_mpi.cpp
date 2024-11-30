#include <gtest/gtest.h>
#include <limits.h>

#include "big_mpi.h"
#include "comm_mpi.h"

class CommMPIFixture : public testing::Test {
protected:
    CommMPI comm_mpi_;
    int mpi_size_ = 1;
    int mpi_rank_ = 0;

    template<typename T>
    static void PrepareArray(T* array, const int array_len, T value_start, T value_step) {
        for (int i = 0; i < array_len; i++) {
            array[i] = value_start + i * value_step;
        }
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
    for (int r = 0; r < mpi_size_ - 1; r++) {
        send_count_in_each_rank[r] = total_send_counts / mpi_size_;
        if (mpi_rank_ > r) {
            displacement += send_count_in_each_rank[r];
        }
    }
    send_count_in_each_rank[mpi_size_ - 1] = total_send_counts - (total_send_counts / mpi_size_) * (mpi_size_ - 1);
    long* send_buffer = new long[send_count_in_each_rank[mpi_rank_]];
    long* recv_buffer = new long[total_send_counts];
    PrepareArray<long>(send_buffer, send_count_in_each_rank[mpi_rank_], displacement, 1.0);

    // Act
    const int result =
        big_MPI_Gatherv<long>(mpi_root, send_count_in_each_rank, (void*)send_buffer, &mpi_datatype, (void*)recv_buffer);

    // Assert
    EXPECT_EQ(result, YT_SUCCESS);
    if (mpi_rank_ == mpi_root) {
        for (int i = 0; i < total_send_counts; i++) {
            EXPECT_EQ(recv_buffer[i], static_cast<long>(i));
        }
    }

    // Clean up
    delete[] send_count_in_each_rank;
    delete[] send_buffer;
    delete[] recv_buffer;
}

int main(int argc, char* argv[]) {
    int result = 0;

    ::testing::InitGoogleTest(&argc, argv);
    MPI_Init(&argc, &argv);
    result = RUN_ALL_TESTS();
    MPI_Finalize();

    return result;
}