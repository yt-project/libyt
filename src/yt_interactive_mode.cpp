#ifdef INTERACTIVE_MODE

#include "yt_combo.h"
#include <readline.h>

//-------------------------------------------------------------------------------------------------------
// Function    :  yt_interactive_mode
// Description :  Enter libyt interactive mode.
//
// Note        :  1. TODO: Only enter this mode when inline functions have error or "flag_file_name" is detacted.
//                2. TODO: Display inline script execute result finished/failed, and show errors if have.
//                3. TODO: Enter interactive mode.
//                   (1) TODO: Python scripting
//                   (2) TODO: libyt command
//                4. TODO: Let user maintain what inline function to run in the follow process.
//
// Parameter   :  const char *flag_file_name : once this file is detacted, it enters yt_interactive_mode
//
// Return      :  YT_SUCCESS or YT_FAIL
//-------------------------------------------------------------------------------------------------------
int yt_interactive_mode(const char* flag_file_name){

}

#endif // #ifdef INTERACTIVE_MODE
