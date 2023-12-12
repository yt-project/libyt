#ifdef INTERACTIVE_MODE

#include "func_status_list.h"

#include <string.h>

#include <iostream>

#include "yt_combo.h"

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
    SET_TIMER(__PRETTY_FUNCTION__);

    for (int i = 0; i < size(); i++) {
        m_FuncStatusList[i].set_status(-1);
    }
    return YT_SUCCESS;
}

//-------------------------------------------------------------------------------------------------------
// Class       :  func_status_list
// Method      :  print_summary
//
// Notes       :  1. Print function status and error msg in func_status_list.
//                2. Only root rank prints.
//                3. normal      -> bold white
//                   idle        -> bold blue
//                   not run yet -> bold yellow
//                   success     -> bold green
//                   failed      -> bold red
//                   MPI process -> bold cyan
//
// Arguments   :  None
//
// Return      : YT_SUCCESS or YT_FAIL
//-------------------------------------------------------------------------------------------------------
int func_status_list::print_summary() {
    SET_TIMER(__PRETTY_FUNCTION__);

    // make sure every rank has reach here, so that printing in other ranks are done
    fflush(stdout);
    fflush(stderr);

    if (g_myrank == g_myroot) {
        printf("\033[1;37m");
        printf("=====================================================================\n");
        printf("  %-40s     %-12s   %s\n", "Inline Function", "Status", "Run");
        printf("---------------------------------------------------------------------\n");
        for (int i = 0; i < size(); i++) {
            printf("\033[1;37m");  // change to bold white
            printf("  * %-43s", m_FuncStatusList[i].get_func_name());
            int run = m_FuncStatusList[i].get_run();
            int status = m_FuncStatusList[i].get_status();

            if (status == 0) {
                printf("\033[1;31m");  // bold red: failed
                printf("%-12s", "failed");
            } else if (status == 1) {
                printf("\033[1;32m");  // bold green: success
                printf("%-12s", "success");
            } else if (status == -1) {
                printf("\033[1;34m");  // bold blue: idle
                printf("%-12s", "idle");
            } else {
                printf("\033[0;37m");  // change to white
                printf("%-12s (%d)", "unknown status", status);
            }

            printf("\033[1;33m");  // bold yellow
            if (run == 1)
                printf("%5s\n", "V");
            else
                printf("%5s\n", "X");

            fflush(stdout);
        }
        printf("\033[1;37m");
        printf("=====================================================================\n");
        printf("\033[0;37m");  // change to white
    }

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
// Return      :  index : index of func_name in list, return -1 if doesn't exist.
//-------------------------------------------------------------------------------------------------------
int func_status_list::get_func_index(const char* func_name) {
    SET_TIMER(__PRETTY_FUNCTION__);

    int index = -1;
    for (int i = 0; i < size(); i++) {
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
//                2. Return function index.
//
// Arguments   :  char   *func_name: inline function name
//                int     run      : run in next inline analysis or not.
//
// Return      : Function index in list.
//-------------------------------------------------------------------------------------------------------
int func_status_list::add_new_func(const char* func_name, int run) {
    SET_TIMER(__PRETTY_FUNCTION__);

    // Check if func_name exist, return YT_SUCCESS if exist
    int index = get_func_index(func_name);
    if (index >= 0) return index;

    // add func_name, since it adds to the end, its index is equal to original size
    index = size();
    m_FuncStatusList.emplace_back(func_name, run);

    return index;
}

//-------------------------------------------------------------------------------------------------------
// Class       :  func_status_list
// Method      :  run_func
//
// Notes       :  1. This is a collective call. It executes new added functions that haven't run by
//                   yt_run_Function/yt_run_FunctionArguments yet.
//                2. How this method runs python function is identical to yt_run_Function*. It use
//                   PyRun_SimpleString.
//                3. libyt uses either """ or ''' to wrap the code to execute in exec(). It finds if the
//                   arguments are using triple quotes, if yes, it chooses using """ or '''.
//                4. When setting arguments in %libyt run func args, libyt will make sure user are only
//                   using either """ or ''', if they are using triple quotes.
//                5. Get input arguments from func_status.m_Args if it has.
//
// Arguments   :  None
//
// Return      :  YT_SUCCESS
//-------------------------------------------------------------------------------------------------------
int func_status_list::run_func() {
    SET_TIMER(__PRETTY_FUNCTION__);

    for (int i = 0; i < size(); i++) {
        int run = m_FuncStatusList[i].get_run();
        int status = m_FuncStatusList[i].get_status();
        if (run == 1 && status == -1) {
            // command
            const char* funcname = m_FuncStatusList[i].get_func_name();
            int command_width = 350 + strlen(g_param_libyt.script) + strlen(funcname) * 2;
            char* command = (char*)malloc(command_width * sizeof(char));
            const char* wrapped = m_FuncStatusList[i].get_wrapper() ? "\"\"\"" : "'''";
            sprintf(command,
                    "try:\n"
                    "    exec(%s%s(%s)%s, sys.modules[\"%s\"].__dict__)\n"
                    "except Exception as e:\n"
                    "    libyt.interactive_mode[\"func_err_msg\"][\"%s\"] = traceback.format_exc()\n",
                    wrapped, funcname, m_FuncStatusList[i].get_args().c_str(), wrapped, g_param_libyt.script, funcname);

            // run and update status
            log_info("Performing YT inline analysis %s(%s) ...\n", funcname, m_FuncStatusList[i].get_args().c_str());
            m_FuncStatusList[i].set_status(-2);
            if (PyRun_SimpleString(command) != 0) {
                m_FuncStatusList[i].set_status(0);
                free(command);
                YT_ABORT("Unexpected error occurred while executing %s(%s) in script's namespace.\n", funcname,
                         m_FuncStatusList[i].get_args().c_str());
            }
            m_FuncStatusList[i].get_status();
            log_info("Performing YT inline analysis %s(%s) ... done\n", funcname,
                     m_FuncStatusList[i].get_args().c_str());

            // clean up
            free(command);
        }
    }
    return YT_SUCCESS;
}

#endif  // #ifdef INTERACTIVE_MODE
