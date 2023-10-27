#ifndef __YT_TYPE_H__
#define __YT_TYPE_H__



/*******************************************************************************
/
/  Data types used by libyt
/
********************************************************************************/


// short names for unsigned types
typedef unsigned int       uint;
typedef unsigned long int  ulong;


// enumerate types
typedef enum yt_verbose { YT_VERBOSE_OFF=0, YT_VERBOSE_INFO=1, YT_VERBOSE_WARNING=2, YT_VERBOSE_DEBUG=3 } yt_verbose;
typedef enum yt_dtype {
    YT_FLOAT=0,           // float
    YT_DOUBLE,            // double
    YT_LONGDOUBLE,        // long double
    YT_INT,               // int
    YT_LONG,              // long
    YT_DTYPE_UNKNOWN
} yt_dtype;


// structures
#include "yt_type_param_libyt.h"
#include "yt_type_param_yt.h"
#include "yt_type_grid.h"
#include "yt_type_field.h"
#include "yt_type_particle.h"
#include "yt_type_array.h"


#endif // #ifndef __YT_TYPE_H__
