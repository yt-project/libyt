#include <stdarg.h>

#include "libyt_process_control.h"
#include "yt_combo.h"

// width of log prefix ==> [LogPrefixWidth] messages
static const int LogPrefixWidth = 10;

//-------------------------------------------------------------------------------------------------------
// Function    :  log_info
// Description :  Print out basic messages to standard output
//
// Note        :  1. Work only for verbose level >= YT_VERBOSE_INFO
//                   --> Rely on the global variable "LibytProcessControl::Get().param_libyt_"
//                2. Messages are printed out to standard output with a prefix "[YT_INFO] "
//                3. Use the variable argument lists provided in "stdarg"
//                   --> It is equivalent to call "fprintf( stdout, format, ... );   fflush( Type );"
//                4. Print INFO only in root rank.
//
// Parameter   :  format : Output format
//                ...    : Arguments in vfprintf
//
// Return      :  None
//-------------------------------------------------------------------------------------------------------
void log_info(const char* format, ...) {
    if (LibytProcessControl::Get().mpi_rank_ != 0) return;

    // work only for verbose level >= YT_VERBOSE_INFO
    if (LibytProcessControl::Get().param_libyt_.verbose < YT_VERBOSE_INFO) return;

    // flush previous messages
    fflush(stdout);

    // print messages
    va_list arg;
    va_start(arg, format);

    fprintf(stdout, "[%-*s] ", LogPrefixWidth, "YT_INFO");
    vfprintf(stdout, format, arg);
    fflush(stdout);

    va_end(arg);

}  // FUNCTION : log_info

//-------------------------------------------------------------------------------------------------------
// Function    :  log_warning
// Description :  Print out warning messages to standard error
//
// Note        :  1. Similar to log_info, excpet that it works only for verbose level >= YT_VERBOSE_WARNING
//                2. Messages are printed out to standard output with a prefix "[YT_WARNING] "
//
// Parameter   :  format : Output format
//                ...    : Arguments in vfprintf
//
// Return      :  None
//-------------------------------------------------------------------------------------------------------
void log_warning(const char* format, ...) {
    // work only for verbose level >= YT_VERBOSE_WARNING
    if (LibytProcessControl::Get().param_libyt_.verbose < YT_VERBOSE_WARNING) return;

    // flush previous messages
    fflush(stderr);

    // print messages
    va_list arg;
    va_start(arg, format);

    fprintf(stderr, "[%-*s] ", LogPrefixWidth, "YT_WARNING");
    vfprintf(stderr, format, arg);
    fflush(stderr);

    va_end(arg);

}  // FUNCTION : log_warning

//-------------------------------------------------------------------------------------------------------
// Function    :  log_debug
// Description :  Print out debug messages to standard output
//
// Note        :  1. Similar to log_info, excpet that it works only for verbose level >= YT_VERBOSE_DEBUG
//                2. Messages are printed out to standard output with a prefix "[YT_DEBUG] "
//
// Parameter   :  format : Output format
//                ...    : Arguments in vfprintf
//
// Return      :  None
//-------------------------------------------------------------------------------------------------------
void log_debug(const char* format, ...) {
    // work only for verbose level >= YT_VERBOSE_DEBUG
    if (LibytProcessControl::Get().param_libyt_.verbose < YT_VERBOSE_DEBUG) return;

    // flush previous messages
    fflush(stderr);

    // print messages
    va_list arg;
    va_start(arg, format);

    fprintf(stderr, "[%-*s] ", LogPrefixWidth, "YT_DEBUG");
    vfprintf(stderr, format, arg);
    fflush(stderr);

    va_end(arg);

}  // FUNCTION : log_debug

//-------------------------------------------------------------------------------------------------------
// Function    :  log_error
// Description :  Print out error messages to standard error
//
// Note        :  1. Similar to log_info, excpet that messages are always printed out regardless of the
//                   verbose level
//                2. Messages are printed out to standard error with a prefix "[YT_ERROR] "
//                3. A convenient macro "YT_ABORT" is defined in yt_macro.h, which calls log_error, print
//                   out the line number, and returns YT_FAIL
//
// Parameter   :  format : Output format
//                ...    : Arguments in vfprintf
//
// Return      :  None
//-------------------------------------------------------------------------------------------------------
void log_error(const char* format, ...) {
    // flush previous messages
    fflush(stderr);

    // print messages
    va_list arg;
    va_start(arg, format);

    fprintf(stderr, "[%-*s] ", LogPrefixWidth, "YT_ERROR");
    vfprintf(stderr, format, arg);
    fflush(stderr);

    va_end(arg);

}  // FUNCTION : log_error
