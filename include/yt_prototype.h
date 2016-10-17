#ifndef __YT_PROTOTYPE_H__
#define __YT_PROTOTYPE_H__



// include relevant headers
#include "yt_type.h"

void log_info   ( const char *Format, ... );
void log_warning( const char *format, ... );
void log_debug  ( const char *Format, ... );
void log_error  ( const char *format, ... );
int  init_python( int argc, char *argv[] );
int  init_libyt_module();
#ifndef NO_PYTHON
template <typename T>
int  add_dict_scalar( PyObject *dict, const char *key, const T value );
template <typename T>
int  add_dict_vector3( PyObject *dict, const char *key, const T *vector );
int  add_dict_string( PyObject *dict, const char *key, const char *string );
#endif



#endif // #ifndef __YT_PROTOTYPE_H__
