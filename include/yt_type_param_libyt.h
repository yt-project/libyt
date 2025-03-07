#ifndef LIBYT_PROJECT_INCLUDE_YT_TYPE_PARAM_LIBYT_H_
#define LIBYT_PROJECT_INCLUDE_YT_TYPE_PARAM_LIBYT_H_

#ifndef __cplusplus
#include <stdbool.h>
#endif

/**
 * \struct yt_param_libyt
 * \brief Data structure of libyt runtime parameters
 *
 * \rst
 * .. caution::
 *    The lifetime of ``script`` should cover the whole in situ process in libyt.
 * \endrst
 */
typedef struct yt_param_libyt {
  yt_verbose verbose; /*!< Verbose log level */
  const char* script; /*!< Script name _without_ the file extension `.py` */
  long counter;       /*!< Number of iteration doing in situ analysis */
  bool check_data;    /*!< Check the input data (e.g., hierarchy, grid information...) */

#ifdef __cplusplus
  yt_param_libyt() {
    verbose = YT_VERBOSE_WARNING;
    script = "yt_inline_script";
    counter = 0;
    check_data = true;
  }
#endif  // #ifdef __cplusplus

} yt_param_libyt;

#endif  // LIBYT_PROJECT_INCLUDE_YT_TYPE_PARAM_LIBYT_H_
