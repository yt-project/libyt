#include "libyt.h"
#include "libyt_process_control.h"
#include "logging.h"
#include "timer.h"

#ifdef USE_PYBIND11
#include "pybind11/embed.h"
#endif

/**
 * \defgroup api_yt_finalize libyt API: yt_finalize
 * \fn int yt_finalize()
 * \brief Finalize libyt workflow
 * \details
 * 1. Do not reinitialize libyt (i.e., calling yt_initialize()) after calling this
 *    function. Some extensions (e.g., NumPy) may not work properly.
 * 2. Make sure that the user has follow the full libyt workflow. Like calling
 *    \ref yt_free before calling this function.
 *
 * @return \ref YT_SUCCESS or \ref YT_FAIL
 */
int yt_finalize() {
  SET_TIMER(__PRETTY_FUNCTION__);

  logging::LogInfo("Exiting libyt ...\n");

  // check whether libyt has been initialized
  if (!LibytProcessControl::Get().libyt_initialized_) {
    YT_ABORT("Calling yt_finalize() before yt_initialize()!\n");
  }

  // check if all the libyt allocated resource are freed
  if (LibytProcessControl::Get().need_free_) {
    YT_ABORT("Please invoke yt_free() before calling yt_finalize().\n");
  }

#ifndef USE_PYBIND11
  Py_Finalize();
#else
  pybind11::finalize_interpreter();
#endif

  LibytProcessControl::Get().libyt_initialized_ = false;

  return YT_SUCCESS;

}  // FUNCTION : yt_finalize
