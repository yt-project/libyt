#include <gtest/gtest.h>
#ifndef SERIAL_MODE
#include "comm_mpi.h"
#endif

#include "data_structure_amr.h"

class PythonFixture : public testing::Test {
private:
    // Though mpi_ prefix is used, in serial mode, it will be rank 0 and size 1.
    int mpi_rank_ = 0;
    int mpi_size_ = 1;
    PyObject* py_template_dict_storage_ = nullptr;
    PyObject* py_hierarchy_ = nullptr;
    PyObject* py_grid_data_ = nullptr;
    PyObject* py_particle_data_ = nullptr;

    void SetUp() override {
#ifndef SERIAL_MODE
        MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank_);
        MPI_Comm_size(MPI_COMM_WORLD, &mpi_size_);
        CommMpi::InitializeInfo(0);
#endif
        DataStructureAmr::SetMpiInfo(mpi_size_, 0, mpi_rank_);

        // Initialize
        std::cout << "Initialize Python dictionary hierarchy, grid_data, particle_data ... done" << std::endl;
        InitializeTemplateDictStorage();
        py_hierarchy_ = CreateTemplateDictStorage("hierarchy");
        py_grid_data_ = CreateTemplateDictStorage("grid_data");
        py_particle_data_ = CreateTemplateDictStorage("particle_data");
    }

    void TearDown() override { PyRun_SimpleString("del sys"); }
    void InitializeTemplateDictStorage() {
        if (py_template_dict_storage_ != nullptr) {
            return;
        }
        PyRun_SimpleString("import sys; sys.TEMPLATE_DICT_STORAGE = dict()");
        PyObject* py_sys = PyImport_ImportModule("sys");
        py_template_dict_storage_ = PyObject_GetAttrString(py_sys, "TEMPLATE_DICT_STORAGE");
        Py_DECREF(py_sys);
        Py_DECREF(py_template_dict_storage_);
    }

    PyObject* CreateTemplateDictStorage(const std::string& key) {
        PyObject* py_dict = PyDict_New();
        PyDict_SetItemString(py_template_dict_storage_, key.c_str(), py_dict);
        Py_DECREF(py_dict);
        return py_dict;
    }

protected:
    int GetMpiRank() const { return mpi_rank_; }
    int GetMpiSize() const { return mpi_size_; }
    PyObject* GetPyHierarchy() const { return py_hierarchy_; }
    PyObject* GetPyGridData() const { return py_grid_data_; }
    PyObject* GetPyParticleData() const { return py_particle_data_; }
};

class TestDataStructureAmr : public PythonFixture {};

TEST_F(TestDataStructureAmr, Can_gather_local_hierarchy_and_bind_all_hierarchy_to_Python) {
    std::cout << "mpi_size = " << GetMpiSize() << ", mpi_rank = " << GetMpiRank() << std::endl;

    // Arrange
}

int main(int argc, char** argv) {
    int result = 0;

    ::testing::InitGoogleTest(&argc, argv);
#ifndef SERIAL_MODE
    MPI_Init(&argc, &argv);
#endif
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
#ifndef SERIAL_MODE
    MPI_Finalize();
#endif

    return result;
}