#ifndef LIBYT_PROJECT_INCLUDE_YT_TYPE_H_
#define LIBYT_PROJECT_INCLUDE_YT_TYPE_H_

/**
 * \file yt_type.h
 */

// enumerate types
typedef enum yt_verbose {
  YT_VERBOSE_OFF = 0, /*!< Turn off log */
  YT_VERBOSE_INFO,    /*!< Log level info */
  YT_VERBOSE_WARNING, /*!< Log level warning */
  YT_VERBOSE_DEBUG    /*!< Log level debug */
} yt_verbose;

typedef enum yt_dtype {
  YT_FLOAT = 0,    /*!< float */
  YT_DOUBLE,       /*!< double */
  YT_LONGDOUBLE,   /*!< long double */
  YT_CHAR,         /*!< char */
  YT_UCHAR,        /*!< unsigned char */
  YT_SHORT,        /*!< short */
  YT_USHORT,       /*!< unsigned short */
  YT_INT,          /*!< int */
  YT_UINT,         /*!< unsigned int */
  YT_LONG,         /*!< long */
  YT_ULONG,        /*!< unsigned long */
  YT_LONGLONG,     /*!< long long */
  YT_ULONGLONG,    /*!< unsigned long long */
  YT_DTYPE_UNKNOWN /*!< unknown data type */
} yt_dtype;

// structures
#include "yt_type_array.h"
#include "yt_type_field.h"
#include "yt_type_grid.h"
#include "yt_type_param_libyt.h"
#include "yt_type_param_yt.h"
#include "yt_type_particle.h"

#endif  // LIBYT_PROJECT_INCLUDE_YT_TYPE_H_
