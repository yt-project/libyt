#ifndef __YT_H__
#define __YT_H__



// includ necessary headers
#include "yt_type.h"


// declare libyt API
#ifdef __cplusplus
extern "C" {
#endif

int yt_init( int argc, char *argv[], const yt_param *param );

#ifdef __cplusplus
}
#endif



#endif // #ifndef __YT_H__
