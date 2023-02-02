#ifndef __LIBYT_H__
#define __LIBYT_H__



/*******************************************************************************
/
/  This is the API header to be included by simulation codes
/
/  All API files must include it as well to prevent name mangling
/
********************************************************************************/


// include relevant headers
#include <stdio.h>
#include "yt_type.h"


// declare libyt API
#ifdef __cplusplus
extern "C" {
#endif

// For libyt workflow
int yt_initialize( int argc, char *argv[], const yt_param_libyt *param_libyt );
int yt_finalize();
int yt_set_Parameters  ( yt_param_yt *param_yt       );
int yt_get_FieldsPtr   ( yt_field    **field_list    );
int yt_get_ParticlesPtr( yt_particle **particle_list );
int yt_get_GridsPtr    ( yt_grid     **grids_local   );
int yt_set_UserParameterInt   ( const char *key, const int n, const int    *input );
int yt_set_UserParameterLong  ( const char *key, const int n, const long   *input );
int yt_set_UserParameterUint  ( const char *key, const int n, const uint   *input );
int yt_set_UserParameterUlong ( const char *key, const int n, const ulong  *input );
int yt_set_UserParameterFloat ( const char *key, const int n, const float  *input );
int yt_set_UserParameterDouble( const char *key, const int n, const double *input );
int yt_set_UserParameterString( const char *key,              const char   *input );
int yt_commit();
int yt_free();
int yt_run_FunctionArguments( const char *function_name, int argc, ... );
int yt_run_Function         ( const char *function_name );

// For derived field function to get grid information by GID and by field_name.
int yt_getGridInfo_Dimensions( const long gid, int (*dimensions)[3] );
int yt_getGridInfo_LeftEdge(const long, double (*)[3]);
int yt_getGridInfo_RightEdge(const long, double (*)[3]);
int yt_getGridInfo_ParentId(const long, long *);
int yt_getGridInfo_Level(const long, int *);
int yt_getGridInfo_ProcNum(const long, int *);
int yt_getGridInfo_ParticleCount(const long gid, const char *ptype, long *par_count);
int yt_getGridInfo_FieldData( const long gid, const char *field_name, yt_data *field_data);

#ifdef __cplusplus
}
#endif

#endif // #ifndef __LIBYT_H__
