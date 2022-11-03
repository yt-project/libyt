#ifndef __YT_H__
#define __YT_H__



/*******************************************************************************
/
/  This is the API header to be included by simulation codes
/
/  All API files (i.e., yt_init.cpp) must include it as well to prevent name mangling
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
int yt_init( int argc, char *argv[], const yt_param_libyt *param_libyt );
int yt_finalize();
int yt_set_parameter( yt_param_yt *param_yt );
int yt_get_fieldsPtr( yt_field **field_list );
int yt_get_particlesPtr( yt_particle **particle_list );
int yt_get_gridsPtr( yt_grid **grids_local );
int yt_add_user_parameter_int   ( const char *key, const int n, const int    *input );
int yt_add_user_parameter_long  ( const char *key, const int n, const long   *input );
int yt_add_user_parameter_uint  ( const char *key, const int n, const uint   *input );
int yt_add_user_parameter_ulong ( const char *key, const int n, const ulong  *input );
int yt_add_user_parameter_float ( const char *key, const int n, const float  *input );
int yt_add_user_parameter_double( const char *key, const int n, const double *input );
int yt_add_user_parameter_string( const char *key,              const char   *input );
int yt_commit_grids();
int yt_free_gridsPtr();
int yt_inline_argument( char *function_name, int argc, ... );
int yt_inline( char *function_name );

// For derived field function to get grid information by GID and by field_name.
int yt_getGridInfo_Dimensions( const long gid, int (*dimensions)[3] );
int yt_getGridInfo_LeftEdge(const long, double (*)[3]);
int yt_getGridInfo_RightEdge(const long, double (*)[3]);
int yt_getGridInfo_ParentId(const long, long *);
int yt_getGridInfo_Level(const long, int *);
int yt_getGridInfo_ProcNum(const long, int *);
int yt_getGridInfo_FieldData( const long gid, const char *field_name, yt_data *field_data);

#ifdef __cplusplus
}
#endif

#endif // #ifndef __YT_H__
