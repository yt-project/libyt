#ifndef __YT_H__
#define __YT_H__



// include relevant headers
#include "yt_type.h"


// declare libyt API
#ifdef __cplusplus
extern "C" {
#endif

int yt_init( int argc, char *argv[], const yt_param *param );
int yt_finalize();

#ifdef __cplusplus
}
#endif



#endif // #ifndef __YT_H__
