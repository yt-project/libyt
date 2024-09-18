#ifdef USE_PYBIND11

#include "libyt.h"
#include "pybind11/embed.h"

//-------------------------------------------------------------------------------------------------------
// Description :  List of libyt C extension python methods built using Pybind11 API
//
// Note        :  1. List of python C extension methods functions.
//                2. These function will be called in python, so the parameters indicate python
//                   input type.
//
// Lists       :       Python Method         C Extension Function
//              .............................................................
//                     derived_func          libyt_field_derived_func
//                     get_particle          libyt_particle_get_particle
//                     get_field_remote      libyt_field_get_field_remote
//                     get_particle_remote   libyt_particle_get_particle_remote
//-------------------------------------------------------------------------------------------------------

PYBIND11_EMBEDDED_MODULE(libyt, m) {
#ifdef SUPPORT_TIMER
    SET_TIMER(__PRETTY_FUNCTION__, &timer_control);
#endif
    m.attr("param_yt") = pybind11::dict();
    m.attr("param_user") = pybind11::dict();
    m.attr("hierarchy") = pybind11::dict();
    m.attr("grid_data") = pybind11::dict();
    m.attr("particle_data") = pybind11::dict();
    m.attr("libyt_info") = pybind11::dict();
#if defined(INTERACTIVE_MODE) || defined(JUPYTER_KERNEL)
    m.attr("interactive_mode") = pybind11::dict();
#endif

    m.attr("libyt_info")["version"] =
        pybind11::make_tuple(LIBYT_MAJOR_VERSION, LIBYT_MINOR_VERSION, LIBYT_MICRO_VERSION);
    m.attr("libyt_info")["SERIAL_MODE"] = pybind11::bool_(true);
    m.attr("libyt_info")["INTERACTIVE_MODE"] = pybind11::bool_(false);
    m.attr("libyt_info")["JUPYTER_KERNEL"] = pybind11::bool_(false);
#ifdef SUPPORT_TIMER
    m.attr("libyt_info")["SUPPORT_TIMER"] = pybind11::bool_(true);
#else
    m.attr("libyt_info")["SUPPORT_TIMER"] = pybind11::bool_(false);
#endif
}

#endif  // #ifdef USE_PYBIND11