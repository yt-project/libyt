#include <gtest/gtest.h>

#include "numpy_controller.h"

class PythonFixture : public testing::Test {
private:
    // Though mpi_ prefix is used, in serial mode, it will be rank 0 and size 1.
    int mpi_rank_ = 0;
    int mpi_size_ = 1;

    void SetUp() override { numpy_controller::InitializeNumPy(); }
};

class TestNumPyController : public PythonFixture {};

TEST_F(TestNumPyController, Can_create_numpy_array_from_existing_pointer_and_get_array_info) {
    // Arrange
    const int kNdim = 3;
    npy_intp dims[kNdim] = {4, 5, 6};
    int* array = new int[dims[0] * dims[1] * dims[2]];
    yt_dtype data_dtype = YT_INT;

    // Act
    PyObject* np_array = numpy_controller::ArrayToNumPyArray(3, dims, data_dtype, array, false, false);
    NumPyArray np_array_info = numpy_controller::GetNumPyArrayInfo(np_array);

    // Assert
    EXPECT_EQ(np_array_info.data_dtype, data_dtype);
    EXPECT_EQ(np_array_info.ndim, kNdim);
    for (int d = 0; d < kNdim; d++) {
        EXPECT_EQ(np_array_info.data_dims[d], dims[d]);
    }
    EXPECT_EQ(np_array_info.data_ptr, reinterpret_cast<void*>(array));

    // Clean up
    Py_DECREF(np_array);
    delete[] array;
}

int main(int argc, char** argv) {
    int result = 0;

    ::testing::InitGoogleTest(&argc, argv);

    // initialize python
    wchar_t* program = Py_DecodeLocale(argv[0], NULL);
    if (program == NULL) {
        fprintf(stderr, "Fatal error: cannot decode argv[0]\n");
        exit(1);
    }
    Py_SetProgramName(program);
    Py_Initialize();

    result = RUN_ALL_TESTS();

    // finalize python
    if (Py_FinalizeEx() < 0) {
        exit(120);
    }
    PyMem_RawFree(program);

    return result;
}