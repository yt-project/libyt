#ifndef __YT_H__
#define __YT_H__



/*******************************************************************************
/
/  This is the API header to be included by simulation codes
/
/  All API files (i.e., yt_init) must include it as well to prevent name mangling
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
int yt_inline();

#ifdef __cplusplus
}
#endif



#endif // #ifndef __YT_H__
