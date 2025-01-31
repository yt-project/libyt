#ifndef LIBYT_PROJECT_INCLUDE_YT_MACRO_H_
#define LIBYT_PROJECT_INCLUDE_YT_MACRO_H_

#ifdef __cplusplus
#include <cfloat>
#include <climits>
#else
#include <float.h>
#include <limits.h>
#endif  // #ifdef __cplusplus

#ifndef NULL
#define NULL 0
#endif

#define YT_SUCCESS 1
#define YT_FAIL    0

#define FLT_UNDEFINED FLT_MIN
#define DBL_UNDEFINED DBL_MIN
#define INT_UNDEFINED INT_MIN
#define LNG_UNDEFINED LONG_MIN

#endif  // LIBYT_PROJECT_INCLUDE_YT_MACRO_H_
