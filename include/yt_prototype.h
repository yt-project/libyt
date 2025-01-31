#ifndef LIBYT_PROJECT_INCLUDE_YT_PROTOTYPE_H_
#define LIBYT_PROJECT_INCLUDE_YT_PROTOTYPE_H_

// include relevant headers
#include <typeinfo>

#include "yt_type.h"

#ifdef USE_PYBIND11
// #include "pybind11/numpy.h"
#else
#include <Python.h>
#endif

// TODO: move this elsewhere
int check_yt_param_yt(const yt_param_yt& param_yt);
int print_yt_param_yt(const yt_param_yt& param_yt);
int print_yt_field(const yt_field& field);

#endif  // LIBYT_PROJECT_INCLUDE_YT_PROTOTYPE_H_
