#ifndef LIBYT_PROJECT_INCLUDE_LIBYT_H_
#define LIBYT_PROJECT_INCLUDE_LIBYT_H_

/*******************************************************************************
/
/  This is the API header to be included by simulation codes
/
********************************************************************************/

#define LIBYT_MAJOR_VERSION 0
#define LIBYT_MINOR_VERSION 1
#define LIBYT_MICRO_VERSION 0

// declare libyt data type
#include "yt_type.h"

#ifndef __cplusplus
#include <stdbool.h>
#endif

// declare libyt API
#ifdef __cplusplus
extern "C" {
#endif

// For libyt workflow
int yt_initialize(int argc, char* argv[], const yt_param_libyt* param_libyt);
int yt_finalize();
int yt_set_Parameters(yt_param_yt* input_param_yt);
int yt_get_FieldsPtr(yt_field** field_list);
int yt_get_ParticlesPtr(yt_particle** particle_list);
int yt_get_GridsPtr(yt_grid** grids_local);
int yt_set_UserParameterInt(const char* key, const int n, const int* input);
int yt_set_UserParameterLong(const char* key, const int n, const long* input);
int yt_set_UserParameterLongLong(const char* key, const int n, const long long* input);
int yt_set_UserParameterUint(const char* key, const int n, const unsigned int* input);
int yt_set_UserParameterUlong(const char* key, const int n, const unsigned long* input);
int yt_set_UserParameterFloat(const char* key, const int n, const float* input);
int yt_set_UserParameterDouble(const char* key, const int n, const double* input);
int yt_set_UserParameterString(const char* key, const char* input);
int yt_commit();
int yt_free();
int yt_run_FunctionArguments(const char* function_name, int argc, ...);
int yt_run_Function(const char* function_name);
int yt_run_InteractiveMode(const char* flag_file_name);
int yt_run_ReloadScript(const char* flag_file_name, const char* reload_file_name,
                        const char* script_name);
int yt_run_JupyterKernel(const char* flag_file_name, bool use_connection_file);

// For derived field function to get grid information by GID and by field_name.
int yt_getGridInfo_Dimensions(const long gid, int (*dimensions)[3]);
int yt_getGridInfo_LeftEdge(const long gid, double (*left_edge)[3]);
int yt_getGridInfo_RightEdge(const long gid, double (*right_edge)[3]);
int yt_getGridInfo_ParentId(const long gid, long* parent_id);
int yt_getGridInfo_Level(const long gid, int* level);
int yt_getGridInfo_ProcNum(const long gid, int* proc_num);
int yt_getGridInfo_ParticleCount(const long gid, const char* ptype, long* par_count);
int yt_getGridInfo_FieldData(const long gid, const char* field_name, yt_data* field_data);
int yt_getGridInfo_ParticleData(const long gid, const char* ptype, const char* attr,
                                yt_data* par_data);

#ifdef __cplusplus
}
#endif

#endif  // #ifndef __LIBYT_H__
