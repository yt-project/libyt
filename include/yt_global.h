#ifndef __YT_GLOBAL_H__
#define __YT_GLOBAL_H__



// include relevant headers
#include "yt_type.h"

extern bool      g_initialized;
extern yt_param  g_param;

#ifndef NO_PYTHON
extern PyObject *g_grid_data;
extern PyObject *g_hierarchy;
extern PyObject *g_parameter;
#endif



#endif // #ifndef __YT_GLOBAL_H__
