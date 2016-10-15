#ifndef __YT_PROTOTYPE_H__
#define __YT_PROTOTYPE_H__



// include relevant headers
#include "yt_type.h"

void log_info   ( const char *Format, ... );
void log_warning( const char *format, ... );
void log_debug  ( const char *Format, ... );
void log_error  ( const char *format, ... );
int  init_python( int argc, char *argv[] );



#endif // #ifndef __YT_PROTOTYPE_H__
