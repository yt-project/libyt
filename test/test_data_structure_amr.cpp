#include <gtest/gtest.h>
#ifndef SERIAL_MODE
#include "comm_mpi.h"
#endif
#include <Python.h>

#include "data_structure_amr.h"
#include "numpy_controller.h"

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
        numpy_controller::InitializeNumPy();
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
    PyObject* GetPyTemplateDictStorage() const { return py_template_dict_storage_; }
    PyObject* GetPyHierarchy() const { return py_hierarchy_; }
    PyObject* GetPyGridData() const { return py_grid_data_; }
    PyObject* GetPyParticleData() const { return py_particle_data_; }
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

class TestDataStructureAmrBindFieldParticleInfo : public PythonFixture {};
class TestDataStructureAmrBindHierarchy : public PythonFixture, public testing::WithParamInterface<int> {};
class TestDataStructureAmrBindLocalData : public PythonFixture, public testing::WithParamInterface<int> {};
class TestDataStructureAmrGenerateLocalData : public PythonFixture, public testing::WithParamInterface<int> {};

TEST_F(TestDataStructureAmrBindFieldParticleInfo, Can_bind_field_info_to_Python) {
    std::cout << "mpi_size = " << GetMpiSize() << ", mpi_rank = " << GetMpiRank() << std::endl;

    // Arrange
    DataStructureAmr ds_amr;
    ds_amr.SetPythonBindings(GetPyHierarchy(), GetPyGridData(), GetPyParticleData());

    int index_offset = 0;
    bool check_data = false;
    long num_grids = 2400;
    int num_grids_local = (int)num_grids / GetMpiSize();
    int num_fields = 2;
    if (GetMpiRank() == GetMpiSize() - 1) {
        num_grids_local = (int)num_grids - num_grids_local * (GetMpiSize() - 1);
    }
    std::cout << "(num_fields) = (" << num_fields << ")" << std::endl;
    ds_amr.AllocateStorage(num_grids, num_grids_local, num_fields, 0, nullptr, index_offset, check_data);

    yt_field* field_list = ds_amr.GetFieldList();
    field_list[0].field_name = "Field1";
    field_list[0].field_dtype = YT_DOUBLE;
    field_list[1].field_name = "Field2";
    field_list[1].field_dtype = YT_FLOAT;
    field_list[1].field_unit = "g/cm^3";
    field_list[1].num_field_name_alias = 2;
    const char* field2_alias[2] = {"Field2_alias1", "Field2_alias2"};
    field_list[1].field_name_alias = field2_alias;
    field_list[1].field_display_name = "Field2_display_name";

    // Act
    DataStructureOutput status = ds_amr.BindInfoToPython("sys.TEMPLATE_DICT_STORAGE", GetPyTemplateDictStorage());

    // Assert
    EXPECT_EQ(status.status, DataStructureStatus::kDataStructureSuccess) << status.error;

    // Print field_list in Python
    // (Field list in Python is printed here instead of checking its value in Python is because it takes lots of effort
    //  to retrieve, so I just print it for simplicity. Maybe can add a check later if necessary.)
    PyRun_SimpleString("import pprint;pprint.pprint(sys.TEMPLATE_DICT_STORAGE['field_list'])");

    // Clean up
    ds_amr.CleanUp();
}

TEST_F(TestDataStructureAmrBindFieldParticleInfo, Can_bind_particle_info_to_Python) {
    std::cout << "mpi_size = " << GetMpiSize() << ", mpi_rank = " << GetMpiRank() << std::endl;

    // Arrange
    DataStructureAmr ds_amr;
    ds_amr.SetPythonBindings(GetPyHierarchy(), GetPyGridData(), GetPyParticleData());

    int index_offset = 0;
    bool check_data = false;
    long num_grids = 2400;
    int num_grids_local = (int)num_grids / GetMpiSize();
    int num_par_types = 2;
    yt_par_type par_type_list[2];
    par_type_list[0].par_type = "Par1";
    par_type_list[1].par_type = "Par2";
    par_type_list[0].num_attr = 4;
    par_type_list[1].num_attr = 4;
    if (GetMpiRank() == GetMpiSize() - 1) {
        num_grids_local = (int)num_grids - num_grids_local * (GetMpiSize() - 1);
    }
    std::cout << "(num_par_types) = (" << num_par_types << ")" << std::endl;
    ds_amr.AllocateStorage(num_grids, num_grids_local, 0, num_par_types, par_type_list, index_offset, check_data);

    yt_particle* particle_list = ds_amr.GetParticleList();
    const char* attr_name_list[4] = {"PosX", "PosY", "PosZ", "Attr"};
    const char* attr3_alias[2] = {"Attr_alias1", "Attr_alias2"};
    for (int p = 0; p < num_par_types; p++) {
        for (int a = 0; a < particle_list[0].num_attr - 1; a++) {
            particle_list[p].attr_list[a].attr_name = attr_name_list[a];
            particle_list[p].attr_list[a].attr_dtype = YT_DOUBLE;
            particle_list[p].attr_list[a].attr_unit = "kpc";
        }
        particle_list[p].attr_list[3].attr_name = attr_name_list[3];
        particle_list[p].attr_list[3].attr_dtype = YT_INT;
        particle_list[p].attr_list[3].attr_unit = "unit";
        particle_list[p].attr_list[3].num_attr_name_alias = 2;
        particle_list[p].attr_list[3].attr_name_alias = attr3_alias;
        particle_list[p].attr_list[3].attr_display_name = "Attr_display_name";

        particle_list[p].coor_x = "PosX";
        particle_list[p].coor_y = "PosY";
        particle_list[p].coor_z = "PosZ";
    }

    // Act
    DataStructureOutput status = ds_amr.BindInfoToPython("sys.TEMPLATE_DICT_STORAGE", GetPyTemplateDictStorage());

    // Assert
    EXPECT_EQ(status.status, DataStructureStatus::kDataStructureSuccess) << status.error;

    // Print particle_list in Python
    // (Particle list in Python is printed here instead of checking its value in Python is because it takes
    //  lots of effort to retrieve, so I just print it for simplicity. Maybe can add a check later if necessary.)
    PyRun_SimpleString("import pprint;pprint.pprint(sys.TEMPLATE_DICT_STORAGE['particle_list'])");

    // Clean up
    ds_amr.CleanUp();
}

TEST_P(TestDataStructureAmrBindHierarchy, Can_gather_local_hierarchy_and_bind_all_hierarchy_to_Python) {
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
    GenerateLocalHierarchy(num_grids, index_offset, ds_amr.GetGridsLocal(), num_grids_local, num_par_types);

    // Act
    DataStructureOutput status = ds_amr.BindAllHierarchyToPython(mpi_root);

    // Assert it can look up full hierarchy
    EXPECT_EQ(status.status, DataStructureStatus::kDataStructureSuccess) << status.error;
    for (int gid = index_offset; gid < num_grids + index_offset; gid++) {
        // Get grid hierarchy
        int ans_grid_dims[3];
        double ans_grid_left_edge[3], ans_grid_right_edge[3];
        long ans_parent_id;
        int ans_level, ans_proc_num;
        long ans_par_count[2];
        GetGridHierarchy(gid, index_offset, &ans_parent_id, &ans_level, ans_grid_dims, ans_grid_left_edge,
                         ans_grid_right_edge, num_grids, num_par_types, ans_par_count, &ans_proc_num);

        //
        int grid_dims[3];
        status = ds_amr.GetPythonBoundFullHierarchyGridDimensions(gid, grid_dims);
        EXPECT_EQ(status.status, DataStructureStatus::kDataStructureSuccess) << status.error;
        for (int d = 0; d < 3; d++) {
            EXPECT_EQ(grid_dims[d], ans_grid_dims[d]);
        }

        double grid_left_edge[3], grid_right_edge[3];
        status = ds_amr.GetPythonBoundFullHierarchyGridLeftEdge(gid, grid_left_edge);
        EXPECT_EQ(status.status, DataStructureStatus::kDataStructureSuccess) << status.error;
        for (int d = 0; d < 3; d++) {
            EXPECT_EQ(grid_left_edge[d], ans_grid_left_edge[d]);
        }
        status = ds_amr.GetPythonBoundFullHierarchyGridRightEdge(gid, grid_right_edge);
        EXPECT_EQ(status.status, DataStructureStatus::kDataStructureSuccess) << status.error;
        for (int d = 0; d < 3; d++) {
            EXPECT_EQ(grid_right_edge[d], ans_grid_right_edge[d]);
        }

        long parent_id = -2;
        status = ds_amr.GetPythonBoundFullHierarchyGridParentId(gid, &parent_id);
        EXPECT_EQ(status.status, DataStructureStatus::kDataStructureSuccess) << status.error;
        EXPECT_EQ(parent_id, ans_parent_id);

        int level = -2;
        status = ds_amr.GetPythonBoundFullHierarchyGridLevel(gid, &level);
        EXPECT_EQ(status.status, DataStructureStatus::kDataStructureSuccess) << status.error;
        EXPECT_EQ(level, ans_level);

        int proc_num = -2;
        status = ds_amr.GetPythonBoundFullHierarchyGridProcNum(gid, &proc_num);
        EXPECT_EQ(status.status, DataStructureStatus::kDataStructureSuccess) << status.error;
        EXPECT_EQ(proc_num, ans_proc_num);

        long par_count = -2;
        for (int p = 0; p < num_par_types; p++) {
            status = ds_amr.GetPythonBoundFullHierarchyGridParticleCount(gid, par_type_list[p].par_type, &par_count);
            EXPECT_EQ(status.status, DataStructureStatus::kDataStructureSuccess) << status.error;
            EXPECT_EQ(par_count, ans_par_count[p]);
        }
    }

    // Clean up
    ds_amr.CleanUp();
}

TEST_P(TestDataStructureAmrBindLocalData, Can_bind_local_field_data_to_Python) {
    std::cout << "mpi_size = " << GetMpiSize() << ", mpi_rank = " << GetMpiRank() << std::endl;

    // Arrange
    DataStructureAmr ds_amr;
    ds_amr.SetPythonBindings(GetPyHierarchy(), GetPyGridData(), GetPyParticleData());

    int index_offset = GetParam();
    bool check_data = false;
    int num_grids_local = 2;
    long num_grids = num_grids_local * GetMpiSize();
    int num_fields = 2;
    std::cout << "(index_offset, num_fields) = (" << index_offset << ", " << num_fields << ")" << std::endl;
    ds_amr.AllocateStorage(num_grids, num_grids_local, num_fields, 0, nullptr, index_offset, check_data);
    GenerateLocalHierarchy(num_grids, index_offset, ds_amr.GetGridsLocal(), num_grids_local, 0);

    // Set field info, default is cell-centered and no ghost cell
    yt_field* field_list = ds_amr.GetFieldList();
    field_list[0].field_name = "Field1";
    field_list[0].field_dtype = YT_DOUBLE;
    field_list[0].contiguous_in_x = false;
    field_list[1].field_name = "Field2";
    field_list[1].field_dtype = YT_INT;
    field_list[1].contiguous_in_x = false;

    // Set local field data
    yt_grid* grids_local = ds_amr.GetGridsLocal();
    long length =
        grids_local[0].grid_dimensions[0] * grids_local[0].grid_dimensions[1] * grids_local[0].grid_dimensions[2];
    int grid_dims[3] = {grids_local[0].grid_dimensions[0], grids_local[0].grid_dimensions[1],
                        grids_local[0].grid_dimensions[2]};
    std::vector<void*> field_data;
    double* field1_data = new double[length];
    int* field2_data = new int[length];
    for (long i = 0; i < length; i++) {
        field1_data[i] = 1.0;
        field2_data[i] = 2;
    }
    field_data.push_back(field1_data);
    field_data.push_back(field2_data);
    for (int lid = 0; lid < num_grids_local; lid++) {
        grids_local[lid].field_data[0].data_ptr = field1_data;
        grids_local[lid].field_data[1].data_ptr = field2_data;
    }

    // Act
    DataStructureOutput status = ds_amr.BindLocalDataToPython();

    // Assert
    EXPECT_EQ(status.status, DataStructureStatus::kDataStructureSuccess) << status.error;
    yt_data query_data;
    for (int i = 0; i < num_grids_local; i++) {
        long gid = num_grids_local * GetMpiRank() + i + index_offset;
        for (int v = 0; v < num_fields; v++) {
            status = ds_amr.GetPythonBoundLocalFieldData(gid, field_list[v].field_name, &query_data);
            EXPECT_EQ(status.status, DataStructureStatus::kDataStructureSuccess) << status.error;
            EXPECT_EQ(query_data.data_dtype, field_list[v].field_dtype);
            EXPECT_EQ(query_data.data_ptr, field_data[v]);
            for (int d = 0; d < 3; d++) {
                EXPECT_EQ(query_data.data_dimensions[d], grid_dims[d]);
            }
        }
    }
    PyRun_SimpleString("import pprint;pprint.pprint(sys.TEMPLATE_DICT_STORAGE['grid_data'])");

    // Clean up
    ds_amr.CleanUp();
    for (auto data : field_data) {
        free(data);
    }
}

TEST_P(TestDataStructureAmrBindLocalData, Can_bind_local_particle_data_to_Python) {
    std::cout << "mpi_size = " << GetMpiSize() << ", mpi_rank = " << GetMpiRank() << std::endl;

    // Arrange
    DataStructureAmr ds_amr;
    ds_amr.SetPythonBindings(GetPyHierarchy(), GetPyGridData(), GetPyParticleData());

    int index_offset = GetParam();
    bool check_data = false;
    int num_grids_local = 2;
    long num_grids = num_grids_local * GetMpiSize();
    int num_par_types = 2;
    yt_par_type par_type_list[2];
    par_type_list[0].par_type = "Par1";
    par_type_list[1].par_type = "Par2";
    par_type_list[0].num_attr = 4;
    par_type_list[1].num_attr = 4;
    std::cout << "(index_offset, num_par_types) = (" << index_offset << ", " << num_par_types << ")" << std::endl;
    ds_amr.AllocateStorage(num_grids, num_grids_local, 0, num_par_types, par_type_list, index_offset, check_data);
    GenerateLocalHierarchy(num_grids, index_offset, ds_amr.GetGridsLocal(), num_grids_local, num_par_types);

    // Set particle info
    yt_particle* particle_list = ds_amr.GetParticleList();
    const char* attr_name_list[4] = {"PosX", "PosY", "PosZ", "Attr"};
    for (int p = 0; p < num_par_types; p++) {
        for (int a = 0; a < particle_list[0].num_attr - 1; a++) {
            particle_list[p].attr_list[a].attr_name = attr_name_list[a];
            particle_list[p].attr_list[a].attr_dtype = YT_DOUBLE;
        }
        particle_list[p].attr_list[3].attr_name = attr_name_list[3];
        particle_list[p].attr_list[3].attr_dtype = YT_INT;
    }

    // Set local particle data
    yt_grid* grids_local = ds_amr.GetGridsLocal();
    std::vector<void*> particle_data;
    long length = grids_local[0].par_count_list[0];
    double* dummy_double = new double[length];
    int* dummy_int = new int[length];
    for (int i = 0; i < length; i++) {
        dummy_double[i] = 1.0;
        dummy_int[i] = 2;
    }
    particle_data.push_back(dummy_double);
    particle_data.push_back(dummy_int);
    for (int lid = 0; lid < num_grids_local; lid++) {
        for (int p = 0; p < num_par_types; p++) {
            grids_local[lid].particle_data[p][0].data_ptr = dummy_double;
            grids_local[lid].particle_data[p][1].data_ptr = dummy_double;
            grids_local[lid].particle_data[p][2].data_ptr = dummy_double;
            grids_local[lid].particle_data[p][3].data_ptr = dummy_int;
        }
    }

    // Act
    DataStructureOutput status = ds_amr.BindLocalDataToPython();

    // Assert
    EXPECT_EQ(status.status, DataStructureStatus::kDataStructureSuccess) << status.error;
    yt_data query_data;
    for (int i = 0; i < num_grids_local; i++) {
        long gid = num_grids_local * GetMpiRank() + i + index_offset;
        for (int p = 0; p < num_par_types; p++) {
            for (int a = 0; a < particle_list[0].num_attr; a++) {
                status = ds_amr.GetPythonBoundLocalParticleData(gid, particle_list[p].par_type,
                                                                particle_list[p].attr_list[a].attr_name, &query_data);
                EXPECT_EQ(status.status, DataStructureStatus::kDataStructureSuccess) << status.error;
                EXPECT_EQ(query_data.data_dtype, particle_list[p].attr_list[a].attr_dtype);
                EXPECT_EQ(query_data.data_dimensions[0], length);
                if (a == 3) {
                    EXPECT_EQ(query_data.data_ptr, dummy_int);
                } else {
                    EXPECT_EQ(query_data.data_ptr, dummy_double);
                }
            }
        }
    }
    PyRun_SimpleString("import pprint;pprint.pprint(sys.TEMPLATE_DICT_STORAGE['particle_data'])");

    // Clean up
    ds_amr.CleanUp();
    for (auto data : particle_data) {
        free(data);
    }
}

TEST_P(TestDataStructureAmrGenerateLocalData, Can_generate_derived_field_data) {
    std::cout << "mpi_size = " << GetMpiSize() << ", mpi_rank = " << GetMpiRank() << std::endl;

    // Arrange
    DataStructureAmr ds_amr;
    ds_amr.SetPythonBindings(GetPyHierarchy(), GetPyGridData(), GetPyParticleData());

    int mpi_root = 0;
    int index_offset = GetParam();
    bool check_data = false;
    int num_grids_local = 1;
    long num_grids = num_grids_local * GetMpiSize();
    long local_gid = num_grids_local * GetMpiRank() + index_offset;
    int num_fields = 1;
    std::cout << "(index_offset, num_fields) = (" << index_offset << ", " << num_fields << ")" << std::endl;
    ds_amr.AllocateStorage(num_grids, num_grids_local, num_fields, 0, nullptr, index_offset, check_data);
    GenerateLocalHierarchy(num_grids, index_offset, ds_amr.GetGridsLocal(), num_grids_local, 0);

    // Set field info
    yt_field* field_list = ds_amr.GetFieldList();
    field_list[0].field_name = "Field100";
    field_list[0].field_type = "derived_func";
    field_list[0].field_dtype = YT_DOUBLE;
    field_list[0].contiguous_in_x = true;
    field_list[0].derived_func = [](const int len, const long* gid_list, const char* field_name, yt_array* data) {
        for (int i = 0; i < len; i++) {
            for (int data_index = 0; data_index < data[i].data_length; data_index++) {
                ((double*)data[i].data_ptr)[data_index] = 100.0;
            }
        }
    };
    double value = 100.0;
    ds_amr.BindInfoToPython("sys.TEMPLATE_DICT_STORAGE", GetPyTemplateDictStorage());
    ds_amr.BindAllHierarchyToPython(mpi_root);

    // Act
    std::vector<AmrDataArray3D> storage;
    DataStructureOutput status = ds_amr.GenerateLocalFieldData({local_gid}, "Field100", storage);

    // Assert
    EXPECT_EQ(status.status, DataStructureStatus::kDataStructureSuccess) << status.error;
    EXPECT_EQ(storage.size(), 1);
    EXPECT_EQ(storage[0].id, local_gid);
    EXPECT_EQ(storage[0].data_dtype, YT_DOUBLE);
    EXPECT_EQ(storage[0].contiguous_in_x, field_list[0].contiguous_in_x);
    for (int i = 0; i < storage[0].data_dim[0] * storage[0].data_dim[1] * storage[0].data_dim[2]; i++) {
        EXPECT_EQ(((double*)storage[0].data_ptr)[i], value);
    }

    // Clean up
    ds_amr.CleanUp();
    for (const AmrDataArray3D& kData : storage) {
        free(kData.data_ptr);
    }
}

TEST_P(TestDataStructureAmrGenerateLocalData, Can_generate_particle_data) {
    std::cout << "mpi_size = " << GetMpiSize() << ", mpi_rank = " << GetMpiRank() << std::endl;

    // Arrange
    DataStructureAmr ds_amr;
    ds_amr.SetPythonBindings(GetPyHierarchy(), GetPyGridData(), GetPyParticleData());

    int mpi_root = 0;
    int index_offset = GetParam();
    bool check_data = false;
    int num_grids_local = 1;
    long num_grids = num_grids_local * GetMpiSize();
    long local_gid = num_grids_local * GetMpiRank() + index_offset;
    int num_par_types = 1;
    yt_par_type par_type_list[1];
    par_type_list[0].par_type = "Par100";
    par_type_list[0].num_attr = 4;
    std::cout << "(index_offset, num_par_types) = (" << index_offset << ", " << num_par_types << ")" << std::endl;
    ds_amr.AllocateStorage(num_grids, num_grids_local, 0, num_par_types, par_type_list, index_offset, check_data);
    GenerateLocalHierarchy(num_grids, index_offset, ds_amr.GetGridsLocal(), num_grids_local, num_par_types);

    // Set particle info
    yt_particle* particle_list = ds_amr.GetParticleList();
    const char* attr_name_list[4] = {"PosX", "PosY", "PosZ", "Attr"};
    for (int a = 0; a < particle_list[0].num_attr; a++) {
        particle_list[0].attr_list[a].attr_name = attr_name_list[a];
        particle_list[0].attr_list[a].attr_dtype = YT_DOUBLE;
    }
    particle_list[0].coor_x = "PosX";
    particle_list[0].coor_y = "PosY";
    particle_list[0].coor_z = "PosZ";
    particle_list[0].get_par_attr = [](const int len, const long* gid_list, const char* ptype, const char* attr,
                                       yt_array* data) {
        for (int i = 0; i < len; i++) {
            for (int data_index = 0; data_index < data[i].data_length; data_index++) {
                ((double*)data[i].data_ptr)[data_index] = 100.0;
            }
        }
    };
    double value = 100.0;
    ds_amr.BindInfoToPython("sys.TEMPLATE_DICT_STORAGE", GetPyTemplateDictStorage());
    ds_amr.BindAllHierarchyToPython(mpi_root);

    // Act
    std::vector<AmrDataArray1D> storage;
    DataStructureOutput status = ds_amr.GenerateLocalParticleData({local_gid}, "Par100", "Attr", storage);

    // Assert
    EXPECT_EQ(status.status, DataStructureStatus::kDataStructureSuccess) << status.error;
    EXPECT_EQ(storage.size(), 1);
    EXPECT_EQ(storage[0].id, local_gid);
    EXPECT_EQ(storage[0].data_dtype, YT_DOUBLE);
    EXPECT_EQ(storage[0].data_len, 10);
    for (int i = 0; i < storage[0].data_len; i++) {
        EXPECT_EQ(((double*)storage[0].data_ptr)[i], value);
    }

    // Clean up
    ds_amr.CleanUp();
    for (const AmrDataArray1D& kData : storage) {
        free(kData.data_ptr);
    }
}

INSTANTIATE_TEST_SUITE_P(DifferentIndexOffset, TestDataStructureAmrBindHierarchy, testing::Values(0, 1));
INSTANTIATE_TEST_SUITE_P(DifferentIndexOffset, TestDataStructureAmrBindLocalData, testing::Values(0, 1));
INSTANTIATE_TEST_SUITE_P(DifferentIndexOffset, TestDataStructureAmrGenerateLocalData, testing::Values(0, 1));

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