#include <gtest/gtest.h>
#ifndef SERIAL_MODE
#include "comm_mpi.h"
#endif
#include <Python.h>

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
        CommMpi::InitializeYtLongMpiDataType();
        CommMpi::InitializeYtHierarchyMpiDataType();
#endif
        DataStructureAmr::SetMpiInfo(mpi_size_, 0, mpi_rank_);
        DataStructureAmr::InitializeNumPy();

        // Initialize
        InitializeTemplateDictStorage();
        py_hierarchy_ = CreateTemplateDictStorage("hierarchy");
        py_grid_data_ = CreateTemplateDictStorage("grid_data");
        py_particle_data_ = CreateTemplateDictStorage("particle_data");
        std::cout << "Initialize Python dictionary hierarchy, grid_data, particle_data ... done" << std::endl;
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
    void GenerateLocalHierarchy(long num_grids, int index_offset, yt_grid* grids_local, int num_grids_local) {
        // Calculate range based on mpi rank
        long start_i = GetMpiRank() * (num_grids / GetMpiSize());

        // Domain dimensions
        int grid_dim[3] = {10, 1, 1};
        double dx_grid = 1.0;
        double domain_left_edge[3] = {0.0, 0.0, 0.0};
        double domain_right_edge[3] = {dx_grid * (double)num_grids, dx_grid * (double)num_grids,
                                       dx_grid * (double)num_grids};

        for (int i = 0; i < num_grids_local; i++) {
            long gid = start_i + i + index_offset;
            grids_local[i].id = gid;
            grids_local[i].parent_id = -1;
            grids_local[i].level = 0;
            for (int d = 0; d < 3; d++) {
                grids_local[i].grid_dimensions[d] = grid_dim[d];
                grids_local[i].left_edge[d] = domain_left_edge[d] + dx_grid * (double)(start_i + i);
                grids_local[i].right_edge[d] = domain_left_edge[d] + dx_grid * (double)(start_i + i + 1);
            }
            for (int p = 0; p < 2; p++) {
                grids_local[i].par_count_list[p] = gid;
            }
        }
    }
};

class TestDataStructureAmrHierarchy : public PythonFixture, public testing::WithParamInterface<int> {};

TEST_P(TestDataStructureAmrHierarchy, Can_gather_local_hierarchy_and_bind_all_hierarchy_to_Python) {
    std::cout << "mpi_size = " << GetMpiSize() << ", mpi_rank = " << GetMpiRank() << std::endl;

    // Arrange
    DataStructureAmr ds_amr;
    ds_amr.SetPythonBindings(GetPyHierarchy(), GetPyGridData(), GetPyParticleData());

    int mpi_root = 0;
    int index_offset = GetParam();
    bool check_data = false;
    long num_grids = 2400;
    int num_grids_local = (int)num_grids / GetMpiSize();
    int num_par_types = 2;
    yt_par_type par_type_list[2];
    par_type_list[0].par_type = "dark_matter";
    par_type_list[1].par_type = "star";
    par_type_list[0].num_attr = 2;
    par_type_list[1].num_attr = 2;
    if (GetMpiRank() == GetMpiSize() - 1) {
        num_grids_local = (int)num_grids - num_grids_local * (GetMpiSize() - 1);
    }
    std::cout << "(mpi_root, index_offset, num_grids, num_grids_local, check_data) = (" << mpi_root << ", "
              << index_offset << ", " << num_grids << ", " << num_grids_local << ", " << check_data << ")" << std::endl;
    ds_amr.AllocateStorage(num_grids, num_grids_local, 0, num_par_types, par_type_list, index_offset, check_data);
    GenerateLocalHierarchy(num_grids, index_offset, ds_amr.GetGridsLocal(), num_grids_local);

    // Act
    DataStructureOutput status = ds_amr.BindAllHierarchyToPython(mpi_root);

    // Assert it gets full hierarchy
    EXPECT_EQ(status.status, DataStructureStatus::kDataStructureSuccess) << status.error;
    for (int gid = index_offset; gid < num_grids + index_offset; gid++) {
        int grid_dims[3];
        status = ds_amr.GetPythonBoundFullHierarchyGridDimensions(gid, grid_dims);
        EXPECT_EQ(status.status, DataStructureStatus::kDataStructureSuccess) << status.error;
        EXPECT_EQ(grid_dims[0], 10);
        EXPECT_EQ(grid_dims[1], 1);
        EXPECT_EQ(grid_dims[2], 1);

        double grid_left_edge[3], grid_right_edge[3];
        status = ds_amr.GetPythonBoundFullHierarchyGridLeftEdge(gid, grid_left_edge);
        EXPECT_EQ(status.status, DataStructureStatus::kDataStructureSuccess) << status.error;
        for (int d = 0; d < 3; d++) {
            EXPECT_EQ(grid_left_edge[d], (double)gid - index_offset);
        }
        status = ds_amr.GetPythonBoundFullHierarchyGridRightEdge(gid, grid_right_edge);
        EXPECT_EQ(status.status, DataStructureStatus::kDataStructureSuccess) << status.error;
        for (int d = 0; d < 3; d++) {
            EXPECT_EQ(grid_right_edge[d], (double)gid - index_offset + 1.0);
        }

        long parent_id = -2;
        status = ds_amr.GetPythonBoundFullHierarchyGridParentId(gid, &parent_id);
        EXPECT_EQ(status.status, DataStructureStatus::kDataStructureSuccess) << status.error;
        EXPECT_EQ(parent_id, -1);

        int level = -2;
        status = ds_amr.GetPythonBoundFullHierarchyGridLevel(gid, &level);
        EXPECT_EQ(status.status, DataStructureStatus::kDataStructureSuccess) << status.error;
        EXPECT_EQ(level, 0);

        int proc_num = -2;
        status = ds_amr.GetPythonBoundFullHierarchyGridProcNum(gid, &proc_num);
        EXPECT_EQ(status.status, DataStructureStatus::kDataStructureSuccess) << status.error;
        int ans_proc_num = ((int)gid - index_offset) / (num_grids / GetMpiSize());
        if (ans_proc_num == GetMpiSize()) {
            ans_proc_num = GetMpiSize() - 1;
        }
        EXPECT_EQ(proc_num, ans_proc_num);

        long par_count = -2;
        status = ds_amr.GetPythonBoundFullHierarchyGridParticleCount(gid, "dark_matter", &par_count);
        EXPECT_EQ(status.status, DataStructureStatus::kDataStructureSuccess) << status.error;
        EXPECT_EQ(par_count, gid);
        status = ds_amr.GetPythonBoundFullHierarchyGridParticleCount(gid, "star", &par_count);
        EXPECT_EQ(status.status, DataStructureStatus::kDataStructureSuccess) << status.error;
        EXPECT_EQ(par_count, gid);
    }

    // Clean up
    ds_amr.CleanUp();
}

INSTANTIATE_TEST_SUITE_P(TestDataStructureAmrHierarchyInstantiation, TestDataStructureAmrHierarchy,
                         testing::Values(0, 1));

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