#ifdef INTERACTIVE_MODE

#include "yt_macro.h"
#include <string.h>
#include "func_status_list.h"

int func_status_list::get_func_index(char *func_name) {
    int index = -1;

    for (int i=0; i<m_FuncStatusList.size(); i++) {
        if (strcmp(m_FuncStatusList[i].get_func_name(), func_name) == 0) {
            index = i;
            break;
        }
    }

    return index;
}

//-------------------------------------------------------------------------------------------------------
// Class       :  func_status_list
// Method      :  add_new_func
//
// Notes       :  1. Check if func_name is defined inside the vector, if not create one.
//
// Arguments   :  char   *func_name: inline function name
//
// Return      : YT_SUCCESS or YT_FAIL
//-------------------------------------------------------------------------------------------------------
int func_status_list::add_new_func(char *func_name) {
    // Check if func_name exist, return YT_SUCCESS if exist
    if (get_func_index(func_name) >= 0) return YT_SUCCESS;

    // add func_name
    func_status new_func(func_name);
    m_FuncStatusList.push_back(new_func);

    return YT_SUCCESS;
}

#endif // #ifdef INTERACTIVE_MODE
