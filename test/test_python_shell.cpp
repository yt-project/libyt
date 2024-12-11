#include <gtest/gtest.h>
#ifndef SERIAL_MODE
#include <mpi.h>
#endif

#include "libyt_python_shell.h"

class PythonFixture : public testing::Test {
private:
    // Though mpi_ prefix is used, in serial mode, it will be rank 0 and size 1.
    int mpi_rank_ = 0;
    int mpi_size_ = 1;
    void SetUp() override {
#ifndef SERIAL_MODE
        MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank_);
        MPI_Comm_size(MPI_COMM_WORLD, &mpi_size_);
#endif
    }

protected:
    int GetRank() const { return mpi_rank_; }
    int GetSize() const { return mpi_size_; }
};

class TestPythonShell : public PythonFixture {};

TEST_F(TestPythonShell, Python_can_run) {
    std::cout << "mpi_size = " << GetSize() << ", " << "mpi_rank = " << GetRank() << std::endl;
    PyRun_SimpleString("print('Hello, Python!')");
}

int main(int argc, char* argv[]) {
    int result = 0;

    ::testing::InitGoogleTest(&argc, argv);
#ifndef SERIAL_MODE
    MPI_Init(&argc, &argv);
#endif
    // initialize python (todo: should call libyt api later)
    wchar_t* program = Py_DecodeLocale(argv[0], NULL);
    if (program == NULL) {
        fprintf(stderr, "Fatal error: cannot decode argv[0]\n");
        exit(1);
    }
    Py_SetProgramName(program);
    Py_Initialize();

    result = RUN_ALL_TESTS();

    // finalize python (todo: should call libyt api later)
    if (Py_FinalizeEx() < 0) {
        exit(120);
    }
    PyMem_RawFree(program);
#ifndef SERIAL_MODE
    MPI_Finalize();
#endif

    return result;
}