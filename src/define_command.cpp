#ifdef INTERACTIVE_MODE

#include "define_command.h"
#include "yt_combo.h"
#include <sstream>
#include <vector>

//-------------------------------------------------------------------------------------------------------
// Class       :  define_command
// Method      :  run
//
// Notes       :  1. Parst m_Command, and call according method. m_Command does not contain spaces at
//                   the beginning.
//                2. stringstream is slow and string copying is slow, but ..., too lazy to do that.
//
// Arguments   :  None
//
// Return      : YT_SUCCESS or YT_FAIL
//-------------------------------------------------------------------------------------------------------
int define_command::run() {

    std::stringstream ss(m_Command);
    std::string arg;
    std::vector<std::string> arg_list;

    // get rid of %libyt, and start parsing from the second word.
    ss >> arg;
    while (ss >> arg) {
        arg_list.push_back(arg);
    }

    // call corresponding static method
    if (arg_list.size() == 2) {
        if      (arg_list[0].compare("load") == 0)    load_script(arg_list[1].c_str());
        else if (arg_list[0].compare("export") == 0)  export_script(arg_list[1].c_str());
    }
    else {
        YT_ABORT("Unkown libyt command : %s", m_Command.c_str());
    }

    return YT_SUCCESS;
}


//-------------------------------------------------------------------------------------------------------
// Class       :  define_command
// Method      :  is_exit
//
// Notes       :  1. Parse m_Command to see if it is exit.
//                2. Since we need to set variable in interactive mode while loop, we single this method
//                   out of run method.
//
// Arguments   :  None
//
// Return      : true or false
//-------------------------------------------------------------------------------------------------------
bool define_command::is_exit() {
    std::size_t start_pos = 0;
    std::size_t found = m_Command.find("exit", start_pos);
    if (found != std::string::npos) return true;
    else return false;
}


//-------------------------------------------------------------------------------------------------------
// Class         :  define_command
// Static Method :  load_script
//
// Notes         :  1. Reload all the variables and functions defined inside the script. It will erase
//                     the previous Python workspace originally defined in the ongoing inline analysis.
//                  2. Parse functions in script and add to g_func_status_list. If function name already
//                     exists in the list, the source code in libyt.interactive_mode["func_body"] will
//                     be rewritten.
//                  3. All
//
// Arguments     :  char *filename : file name to reload
//
// Return        : YT_SUCCESS or YT_FAIL
//-------------------------------------------------------------------------------------------------------
int define_command::load_script(const char *filename) {
    printf("Reloading script %s ...\n", filename);
    return YT_SUCCESS;
}


//-------------------------------------------------------------------------------------------------------
// Class         :  define_command
// Static Method :  export_script
//
// Notes         :  1.
//
// Arguments     :  char *filename : output file name
//
// Return        :  YT_SUCCESS or YT_FAIL
//-------------------------------------------------------------------------------------------------------
int define_command::export_script(const char *filename) {
    printf("Exporting script %s ...\n", filename);
    return YT_SUCCESS;
}
#endif // #ifdef INTERACTIVE_MODE
