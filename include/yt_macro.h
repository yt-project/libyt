#ifndef __YT_MACRO_H__
#define __YT_MACRO_H__

#include <limits.h>
#include <float.h>

// include relevant headers/prototypes
void log_error( const char *format, ... );


#ifndef NULL
#define NULL               0
#endif

#define YT_SUCCESS         1
#define YT_FAIL            0

#define FLT_UNDEFINED      FLT_MAX
#define DBL_UNDEFINED      DBL_MAX
#define INT_UNDEFINED      INT_MAX
#define LNG_UNDEFINED      LONG_MAX

// convenient macro to deal with errors
#define YT_ABORT( ... )                                              \
{                                                                    \
   log_error( __VA_ARGS__ );                                         \
   fprintf( stderr, "%13s==> file <%s>, line <%d>, function <%s>\n", \
            "", __FILE__, __LINE__, __FUNCTION__ );                  \
   return YT_FAIL;                                                   \
}



#endif // #ifndef __YT_MACRO_H__
