#ifndef LIBYT_PROJECT_INCLUDE_LIBYT_H_
#define LIBYT_PROJECT_INCLUDE_LIBYT_H_

/**
 * \file libyt.h
 * \brief This is the API header to be included by simulation codes
 */

#define LIBYT_MAJOR_VERSION 0
#define LIBYT_MINOR_VERSION 3
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
// clang-format off
int yt_initialize(int argc, char* argv[], const yt_param_libyt* param_libyt);             /*!< \ingroup api_yt_initialize */
int yt_finalize();                                                                        /*!< \ingroup api_yt_finalize */
int yt_set_Parameters(yt_param_yt* input_param_yt);                                       /*!< \ingroup api_yt_set_Parameters */
int yt_get_FieldsPtr(yt_field** field_list);                                              /*!< \ingroup api_yt_get_FieldsPtr */
int yt_get_ParticlesPtr(yt_particle** particle_list);                                     /*!< \ingroup api_yt_get_ParticlesPtr */
int yt_get_GridsPtr(yt_grid** grids_local);                                               /*!< \ingroup api_yt_get_GridsPtr */
int yt_set_UserParameterInt(const char* key, const int n, const int* input);              /*!< \ingroup api_yt_set_UserParameter */
int yt_set_UserParameterLong(const char* key, const int n, const long* input);            /*!< \ingroup api_yt_set_UserParameter */
int yt_set_UserParameterLongLong(const char* key, const int n, const long long* input);   /*!< \ingroup api_yt_set_UserParameter */
int yt_set_UserParameterUint(const char* key, const int n, const unsigned int* input);    /*!< \ingroup api_yt_set_UserParameter */
int yt_set_UserParameterUlong(const char* key, const int n, const unsigned long* input);  /*!< \ingroup api_yt_set_UserParameter */
int yt_set_UserParameterFloat(const char* key, const int n, const float* input);          /*!< \ingroup api_yt_set_UserParameter */
int yt_set_UserParameterDouble(const char* key, const int n, const double* input);        /*!< \ingroup api_yt_set_UserParameter */
int yt_set_UserParameterString(const char* key, const char* input);                       /*!< \ingroup api_yt_set_UserParameter */
int yt_commit();                                                                          /*!< \ingroup api_yt_commit */
int yt_free();                                                                            /*!< \ingroup api_yt_free */
int yt_run_FunctionArguments(const char* function_name, int argc, ...);                   /*!< \ingroup api_yt_run_Function */
int yt_run_Function(const char* function_name);                                           /*!< \ingroup api_yt_run_Function */
int yt_run_InteractiveMode(const char* flag_file_name);                                   /*!< \ingroup api_yt_run_InteractiveMode */
int yt_run_ReloadScript(const char* flag_file_name, const char* reload_file_name,
                        const char* script_name);                                         /*!< \ingroup api_yt_run_ReloadScript */
int yt_run_JupyterKernel(const char* flag_file_name, bool use_connection_file);           /*!< \ingroup api_yt_run_JupyterKernel */

// For derived field function to get grid information by GID and by field_name.
int yt_getGridInfo_Dimensions(const long gid, int (*dimensions)[3]);                        /*!< \ingroup api_yt_getGridInfo */
int yt_getGridInfo_LeftEdge(const long gid, double (*left_edge)[3]);                        /*!< \ingroup api_yt_getGridInfo */
int yt_getGridInfo_RightEdge(const long gid, double (*right_edge)[3]);                      /*!< \ingroup api_yt_getGridInfo */
int yt_getGridInfo_ParentId(const long gid, long* parent_id);                               /*!< \ingroup api_yt_getGridInfo */
int yt_getGridInfo_Level(const long gid, int* level);                                       /*!< \ingroup api_yt_getGridInfo */
int yt_getGridInfo_ProcNum(const long gid, int* proc_num);                                  /*!< \ingroup api_yt_getGridInfo */
int yt_getGridInfo_ParticleCount(const long gid, const char* ptype, long* par_count);       /*!< \ingroup api_yt_getGridInfo */
int yt_getGridInfo_FieldData(const long gid, const char* field_name, yt_data* field_data);  /*!< \ingroup api_yt_getGridInfo */
int yt_getGridInfo_ParticleData(const long gid, const char* ptype, const char* attr,
                                yt_data* par_data);                                         /*!< \ingroup api_yt_getGridInfo */
// clang-format on
#ifdef __cplusplus
}
#endif

#endif  // #ifndef __LIBYT_H__
