#ifndef __YT_PROTOTYPE_H__
#define __YT_PROTOTYPE_H__



void log_info( const char *Format, ... );
void log_warning( const char *format, ... );
void log_error( const char *format, ... );
int  init_python( int argc, char *argv[], const yt_param *param );



#endif // #ifndef __YT_PROTOTYPE_H__
