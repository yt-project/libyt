#ifndef LIBYT_PROJECT_INCLUDE_DATA_STRUCTURE_AMR_H_
#define LIBYT_PROJECT_INCLUDE_DATA_STRUCTURE_AMR_H_

#include <Python.h>

#include <string>

#include "yt_type.h"

//-------------------------------------------------------------------------------------------------------
// Structure   :  yt_hierarchy
// Description :  Data structure for pass hierarchy of the grid in MPI process, it is meant to be temporary.
//       Notes :  1. We don't deal with particle count in each ptype here.
//
// Data Member :  dimensions     : Number of cells along each direction
//                left_edge      : Grid left  edge in code units
//                right_edge     : Grid right edge in code units
//                level          : AMR level (0 for the root level)
//                proc_num       : An array of MPI rank that the grid belongs
//                id             : Grid ID
//                parent_id      : Parent grid ID
//                proc_num       : Process number, grid belong to which MPI rank
//-------------------------------------------------------------------------------------------------------
struct yt_hierarchy {
    double left_edge[3]{-1.0, -1.0, -1.0};
    double right_edge[3]{-1.0, -1.0, -1.0};
    long id = -1;
    long parent_id = -2;
    int dimensions[3]{-1, -1, -1};
    int level = -1;
    int proc_num = -1;
};

enum class DataStructureStatus : int { kDataStructureFailed = 0, kDataStructureSuccess = 1 };

struct DataStructureOutput {
    DataStructureStatus status;
    std::string error;
};

class DataStructureAmr {
private:
    bool has_particle_;
    bool check_data_;

    // Hierarchy
    long num_grids_;
    int num_fields_;
    int num_par_types_;
    int num_grids_local_;
    int index_offset_;

    double* grid_left_edge_;
    double* grid_right_edge_;
    int* grid_dimensions_;
    long* grid_parent_id_;
    int* grid_levels_;
    int* proc_num_;
    long* par_count_list_;

private:
    void AllocateFieldList();
    void AllocateParticleList(yt_par_type* par_type_list);
    void AllocateGridsLocal();
    void AllocateAllHierarchyStorageForPython();
    void GatherAllHierarchy(int mpi_root, yt_hierarchy** full_hierarchy_ptr, long*** full_particle_count_ptr) const;
    DataStructureOutput BindLocalFieldDataToPython(const yt_grid& grid) const;
    DataStructureOutput BindLocalParticleDataToPython(const yt_grid& grid) const;

    // TODO: Provide check data method
    // (1) check_sum_num_grids_local
    // (2) check the hierarchy

    void CleanUpFieldList();
    void CleanUpParticleList();
    void CleanUpAllHierarchyStorageForPython();
    void CleanUpLocalDataPythonBindings() const;

public:
    // MPI
    static int mpi_size_;
    static int mpi_root_;
    static int mpi_rank_;

    // AMR data structure to data
    yt_field* field_list_;
    yt_particle* particle_list_;
    yt_grid* grids_local_;

    // Python Bindings
    PyObject* py_hierarchy_;      // TODO: should be private
    PyObject* py_grid_data_;      // TODO: should be private
    PyObject* py_particle_data_;  // TODO: make it private after remove yt_rma_particle

public:
    // Initialize
    static void SetMpiInfo(const int mpi_size, const int mpi_root, const int mpi_rank) {
        mpi_size_ = mpi_size;
        mpi_root_ = mpi_root;
        mpi_rank_ = mpi_rank;
    }
    DataStructureAmr();
    void SetPythonBindings(PyObject* py_hierarchy, PyObject* py_grid_data, PyObject* py_particle_data);

    // Process of setting up the data structure
    DataStructureOutput AllocateStorage(long num_grids, int num_grids_local, int num_fields, int num_par_types,
                                        yt_par_type* par_type_list, int index_offset, bool check_data);
    void BindAllHierarchyToPython(int mpi_root);
    void BindLocalDataToPython() const;
    void CleanUpGridsLocal();  // This method is public due to bad API design :(
    void CleanUp();

    // Look up field/particle info method.
    int GetFieldIndex(const char* field_name) const;
    int GetParticleIndex(const char* particle_type) const;
    int GetParticleAttributeIndex(int particle_type_index, const char* attr_name) const;

    // Look up full hierarchy methods
    DataStructureOutput GetPythonBoundFullHierarchyGridDimensions(long gid, int* dimensions) const;
    DataStructureOutput GetPythonBoundFullHierarchyGridLeftEdge(long gid, double* left_edge) const;
    DataStructureOutput GetPythonBoundFullHierarchyGridRightEdge(long gid, double* right_edge) const;
    DataStructureOutput GetPythonBoundFullHierarchyGridParentId(long gid, long* parent_id) const;
    DataStructureOutput GetPythonBoundFullHierarchyGridLevel(long gid, int* level) const;
    DataStructureOutput GetPythonBoundFullHierarchyGridProcNum(long gid, int* proc_num) const;
    DataStructureOutput GetPythonBoundFullHierarchyGridParticleCount(long gid, const char* ptype,
                                                                     long* par_count) const;

    // Look up data methods
    DataStructureOutput GetPythonBoundLocalFieldData(long gid, const char* field_name, yt_data* field_data) const;
    DataStructureOutput GetPythonBoundLocalParticleData(long gid, const char* ptype, const char* attr,
                                                        yt_data* par_data) const;
};

#endif  // LIBYT_PROJECT_INCLUDE_DATA_STRUCTURE_AMR_H_
