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

int yt_init( int argc, char *argv[], const yt_param_libyt *param_libyt );
int yt_finalize();
int yt_set_parameter( yt_param_yt *param_yt );
int yt_get_gridsPtr( yt_grid **grids_local );
int yt_add_user_parameter_int   ( const char *key, const int n, const int    *input );
int yt_add_user_parameter_long  ( const char *key, const int n, const long   *input );
int yt_add_user_parameter_uint  ( const char *key, const int n, const uint   *input );
int yt_add_user_parameter_ulong ( const char *key, const int n, const ulong  *input );
int yt_add_user_parameter_float ( const char *key, const int n, const float  *input );
int yt_add_user_parameter_double( const char *key, const int n, const double *input );
int yt_add_user_parameter_string( const char *key,              const char   *input );
int yt_add_grids();
int yt_inline();

#ifdef __cplusplus
}
#endif



#endif // #ifndef __YT_H__
