#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>




//-------------------------------------------------------------------------------------------------------
// Function    :  log_info
// Description :  Print out messages to standard output
//
// Note        :  1. Use the variable argument lists provided in "stdarg"
//                   --> It is equivalent to call "fprintf( stdout, format, ... );   fflush( Type );"
//
// Parameter   :  format : Output format
//                ...    : Arguments in vfprintf
//
// Return      :  None
//-------------------------------------------------------------------------------------------------------
void log_info( const char *format, ... )
{

// flush previous messages
   fflush( stdout );

   va_list arg;
   va_start( arg, format );

   vfprintf( stdout, format, arg );
   fflush( stdout );

   va_end( arg );

} // FUNCTION : log_info



//-------------------------------------------------------------------------------------------------------
// Function    :  log_warning
// Description :  Print out warning messages to standard error
//
// Note        :  1. Similar to log_info except that the messages are printed out to standard error with
//                   a prefix "Warning: "
//
// Parameter   :  format : Output format
//                ...    : Arguments in vfprintf
//
// Return      :  None
//-------------------------------------------------------------------------------------------------------
void log_warning( const char *format, ... )
{

// flush previous messages
   fflush( stderr );

   va_list arg;
   va_start( arg, format );

   fprintf( stderr, "Warning: " );
   vfprintf( stderr, format, arg );
   fflush( stderr );

   va_end( arg );

} // FUNCTION : log_warning



//-------------------------------------------------------------------------------------------------------
// Function    :  log_error
// Description :  Print out error messages to standard error
//
// Note        :  1. Similar to log_info except that the messages are printed out to standard error with
//                   a prefix "Error: "
//
// Parameter   :  format : Output format
//                ...    : Arguments in vfprintf
//
// Return      :  None
//-------------------------------------------------------------------------------------------------------
void log_error( const char *format, ... )
{

// flush previous messages
   fflush( stderr );

   va_list arg;
   va_start( arg, format );

   fprintf( stderr, "Error: " );
   vfprintf( stderr, format, arg );
   fflush( stderr );

   va_end( arg );

} // FUNCTION : log_error
