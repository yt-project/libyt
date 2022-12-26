#ifdef INTERACTIVE_MODE

#include "yt_combo.h"
#include <string.h>
#include "func_status_list.h"

//-------------------------------------------------------------------------------------------------------
// Class       :  func_status_list
// Method      :  reset
//
// Notes       :  1. Reset every func_status data member m_Status in list to -1 (not run yet).
//
// Arguments   :  None
//
// Return      : YT_SUCCESS
//-------------------------------------------------------------------------------------------------------
int func_status_list::reset() {
    for (int i=0; i<m_FuncStatusList.size(); i++) {
        m_FuncStatusList[i].set_status(-1);
    }
    return YT_SUCCESS;
}


//-------------------------------------------------------------------------------------------------------
// Class       :  func_status_list
// Method      :  print_summary
//
// Notes       :  1. Print function status and error msg in func_status_list.
//                2. Every rank will call this function. Other ranks only need to output their error msg.
//                3. Will use element class's method print_error().
//                4. normal -> bold white
//                   idle   -> bold blue
//                   success-> bold green
//                   failed -> bold red
//
// Arguments   :  None
//
// Return      : YT_SUCCESS or YT_FAIL
//-------------------------------------------------------------------------------------------------------
int func_status_list::print_summary() {
    // make sure every rank has reach here, so that printing in other ranks are done
//    fflush(stdout);
//    fflush(stderr);
//    MPI_Barrier(MPI_COMM_WORLD);

    if (g_myrank != 0) return YT_SUCCESS; // todo

    printf("\033[1;37m"); // change to bold white
    printf("=====================================================================\n");
    printf("* Inline function execute status:\n");
    for (int i=0; i<m_FuncStatusList.size(); i++) {
        printf("  * %-40s ... ", m_FuncStatusList[i].get_func_name());
        bool run = m_FuncStatusList[i].get_run();
        short status = m_FuncStatusList[i].get_status();
        if (!run) {
            printf("\033[1;34m"); // bold blue;
            printf("idle");
            printf("\033[1;37m");
        }
        else {
            if (status == 0) {
                printf("\033[1;31m"); // bold red
                printf("failed");
                printf("\033[1;37m");
            }
            else if (status == 1) {
                printf("\033[1;32m"); // bold green
                printf("success");
                printf("\033[1;37m");
            }
            else if (status == -1) {
                printf("\033[1;33m"); // bold yellow
                printf("not run yet");
                printf("\033[1;37m");
            }
            else {
                printf("\033[0;37m"); // change to white
                YT_ABORT("Unkown status\n");
            }
        }
        printf("\n");
        fflush(stdout);
    }
    printf("=====================================================================\n");
    printf("\033[0;37m"); // change to white
    return YT_SUCCESS;
}


//-------------------------------------------------------------------------------------------------------
// Class       :  func_status_list
// Method      :  get_func_index
//
// Notes       :  1. Look up index of func_name in m_FuncStatusList.
//
// Arguments   :  char   *func_name: inline function name
//
// Return      :  index
//-------------------------------------------------------------------------------------------------------
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
// Return      : YT_SUCCESS
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
