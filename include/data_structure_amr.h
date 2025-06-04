#ifndef LIBYT_PROJECT_INCLUDE_DATA_STRUCTURE_AMR_H_
#define LIBYT_PROJECT_INCLUDE_DATA_STRUCTURE_AMR_H_

#ifndef SERIAL_MODE
#include <mpi.h>
#endif

#include <Python.h>

#include <string>
#include <vector>

#include "yt_type.h"

//-------------------------------------------------------------------------------------------------------
// Structure   :  yt_hierarchy
// Description :  Data structure for pass hierarchy of the grid in MPI process, it is
// meant to be temporary.
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

//-------------------------------------------------------------------------------------------------------
// Structure   :  AmrDataArray3D / AmrDataArray1D
// Description :  Data structure for 3d / 1d data array.
//
// Notes       :  1. Must have data_ptr.
//                2. TODO: do I even need contiguous_in_x in the array element?
//-------------------------------------------------------------------------------------------------------
struct AmrDataArray3D {
  long id = -1;
  yt_dtype data_dtype = YT_DTYPE_UNKNOWN;
  int data_dim[3]{0, 0, 0};
  void* data_ptr = nullptr;
  bool contiguous_in_x = false;
};

struct AmrDataArray2D {
  long id = -1;
  yt_dtype data_dtype = YT_DTYPE_UNKNOWN;
  int data_dim[2]{0, 0};
  void* data_ptr = nullptr;
  bool contiguous_in_x = false;
};

struct AmrDataArray1D {
  long id = -1;
  yt_dtype data_dtype = YT_DTYPE_UNKNOWN;
  long data_dim[1] = {0};
  void* data_ptr = nullptr;
  bool contiguous_in_x = true;  // not in use, keep it only for consistency with 3D/2D
};

enum class DataStructureStatus : int {
  kDataStructureFailed = 0,
  kDataStructureNotImplemented = 1,
  kDataStructureSuccess = 2
};

struct DataStructureOutput {
  DataStructureStatus status;
  std::string error;
};

class DataStructureAmr {
 private:
#ifndef SERIAL_MODE
  static MPI_Datatype mpi_hierarchy_data_type_;
#endif

  bool check_data_;

  // AMR data structure to data
  yt_field* field_list_;
  yt_particle* particle_list_;
  yt_grid* grids_local_;

  // Python Bindings
  PyObject* py_hierarchy_;
  PyObject* py_grid_data_;
  PyObject* py_particle_data_;

  // Hierarchy
  long num_grids_;
  int num_fields_;
  int num_par_types_;
  int num_grids_local_;
  int num_grids_local_field_data_;  // This is for tracking field data in grids_local, we
                                    // need it due to bad Api
  int num_grids_local_par_data_;  // This is for tracking particle data in grids_local, we
                                  // need it due to bad Api
  bool has_particle_;  // This is for tracking particle count column num in hierarchy
                       // Python binding.
  int index_offset_;
  int dimensionality_;  // Dimensionality of the simulation

  double* grid_left_edge_;
  double* grid_right_edge_;
  int* grid_dimensions_;
  long* grid_parent_id_;
  int* grid_levels_;
  int* proc_num_;
  long* par_count_list_;

 private:
  // Initializations
  static void InitializeMpiHierarchyDataType();

  // Allocate storage
  DataStructureOutput AllocateFieldList(int num_fields);
  DataStructureOutput AllocateParticleList(int num_par_types, yt_par_type* par_type_list);
  DataStructureOutput AllocateGridsLocal(int num_grids_local, int num_fields,
                                         int num_par_types, yt_par_type* par_type_list);
  DataStructureOutput AllocateFullHierarchyStorageForPython(long num_grids,
                                                            int num_par_types);

  // Clean up
  void CleanUpFieldList();
  void CleanUpParticleList();
  void CleanUpFullHierarchyStorageForPython();
  void CleanUpLocalDataPythonBindings() const;

  // Sub operations
  DataStructureOutput GatherAllHierarchy(int mpi_root, yt_hierarchy** full_hierarchy_ptr,
                                         long*** full_particle_count_ptr) const;
  DataStructureOutput BindFieldListToPython(PyObject* py_dict,
                                            const std::string& py_dict_name) const;
  DataStructureOutput BindParticleListToPython(PyObject* py_dict,
                                               const std::string& py_dict_name) const;
  DataStructureOutput BindLocalFieldDataToPython(const yt_grid& grid) const;
  DataStructureOutput BindLocalParticleDataToPython(const yt_grid& grid) const;

  // Check data method
#ifndef SERIAL_MODE
  DataStructureOutput CheckHierarchyIsValid(yt_hierarchy* hierarchy) const;
#else
  DataStructureOutput CheckHierarchyIsValid(yt_grid* hierarchy) const;
#endif
  DataStructureOutput CheckFieldList() const;
  DataStructureOutput CheckField(const yt_field& field) const;
  DataStructureOutput CheckParticleList() const;
  DataStructureOutput CheckParticle(yt_particle& particle) const;
  DataStructureOutput CheckParticleAttribute(yt_attribute& attr) const;
  DataStructureOutput CheckGridsLocal() const;
  DataStructureOutput CheckGrid(yt_grid& grid) const;

 public:
  // MPI
  static int mpi_size_;
  static int mpi_root_;
  static int mpi_rank_;

 public:
  // Initialize
  DataStructureAmr();
  static void SetMpiInfo(int mpi_size, int mpi_root, int mpi_rank);
  void SetPythonBindings(PyObject* py_hierarchy, PyObject* py_grid_data,
                         PyObject* py_particle_data);
#ifndef SERIAL_MODE
  MPI_Datatype& GetMpiHierarchyDataType() { return mpi_hierarchy_data_type_; }
#endif

  // Process of setting up the data structure
  DataStructureOutput AllocateStorage(long num_grids, int num_grids_local, int num_fields,
                                      int num_par_types, yt_par_type* par_type_list,
                                      int index_offset, int dimensionality,
                                      bool check_data);
  DataStructureOutput BindInfoToPython(const std::string& py_dict_name,
                                       PyObject* py_dict);
  DataStructureOutput BindAllHierarchyToPython(int mpi_root);
  DataStructureOutput BindLocalDataToPython() const;
  void CleanUpGridsLocal();  // This method is public due to bad API design :(
  void CleanUp();

  // Get basic info
  int GetDimensionality() const { return dimensionality_; }

  // Look up field/particle info method.
  yt_grid* GetGridsLocal() const { return grids_local_; }
  yt_field* GetFieldList() const { return field_list_; }
  yt_particle* GetParticleList() const { return particle_list_; }
  int GetFieldIndex(const char* field_name) const;
  int GetParticleIndex(const char* particle_type) const;
  int GetParticleAttributeIndex(int particle_type_index, const char* attr_name) const;

  // Generate data array
  template<typename DataClass>
  DataStructureOutput GenerateLocalFieldData(const std::vector<long>& gid_list,
                                             const char* field_name,
                                             std::vector<DataClass>& storage) const;
  DataStructureOutput GenerateLocalParticleData(
      const std::vector<long>& gid_list, const char* ptype, const char* attr,
      std::vector<AmrDataArray1D>& storage) const;

  // Look up full hierarchy methods
  DataStructureOutput GetPythonBoundFullHierarchyGridDimensions(long gid,
                                                                int* dimensions) const;
  DataStructureOutput GetPythonBoundFullHierarchyGridLeftEdge(long gid,
                                                              double* left_edge) const;
  DataStructureOutput GetPythonBoundFullHierarchyGridRightEdge(long gid,
                                                               double* right_edge) const;
  DataStructureOutput GetPythonBoundFullHierarchyGridParentId(long gid,
                                                              long* parent_id) const;
  DataStructureOutput GetPythonBoundFullHierarchyGridLevel(long gid, int* level) const;
  DataStructureOutput GetPythonBoundFullHierarchyGridProcNum(long gid,
                                                             int* proc_num) const;
  DataStructureOutput GetPythonBoundFullHierarchyGridParticleCount(long gid,
                                                                   const char* ptype,
                                                                   long* par_count) const;

  // Look up data methods
  DataStructureOutput GetPythonBoundLocalFieldData(long gid, const char* field_name,
                                                   yt_data* field_data) const;
  DataStructureOutput GetPythonBoundLocalParticleData(long gid, const char* ptype,
                                                      const char* attr,
                                                      yt_data* par_data) const;
};

#endif  // LIBYT_PROJECT_INCLUDE_DATA_STRUCTURE_AMR_H_
