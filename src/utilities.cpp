#include <sys/stat.h>

#include "libyt_utilities.h"

namespace libyt_utilities {

//-------------------------------------------------------------------------------------------------------
// Function    :  DoesFileExist
// Description :  Does the file exist?
//
// Parameter   :  const char *flag_file : check if this file exist
//
// Return      :  true/false
//-------------------------------------------------------------------------------------------------------
bool DoesFileExist(const char* file_name) {
    struct stat buffer {};
    if (stat(file_name, &buffer) != 0) {
        return false;
    } else {
        return true;
    }
}

}  // namespace libyt_utilities
