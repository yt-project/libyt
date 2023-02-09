#include "yt_combo.h"
#include <stdarg.h>
#include <iostream>
#include <string>
#include "libyt.h"


//-------------------------------------------------------------------------------------------------------
// Function    :  yt_run_FunctionArguments
// Description :  Call function with arguments in in situ process
//
// Note        :  1. Python script name, which is also its namespace's name is stored in "g_param_libyt.script"
//                2. This python script must contain function of <function_name> you called.
//                3. Must give argc (argument count), even if there are no arguments.
//                4. Under INTERACTIVE_MODE, function will be wrapped inside try/except. If there is error
//                   it will store under libyt.interactive_mode["func_err_msg"]["func_name"].
//
// Parameter   :  const char *function_name : function name in python script 
//                int  argc                 : input arguments count
//                ...                       : list of arguments, should be input as (char*)
//
// Return      :  YT_SUCCESS or YT_FAIL
//-------------------------------------------------------------------------------------------------------
int yt_run_FunctionArguments(const char *function_name, int argc, ...) {

#ifdef SUPPORT_TIMER
    g_timer->record_time(function_name, 0);
#endif

    // check if libyt has been initialized
    if (!g_param_libyt.libyt_initialized) {
        YT_ABORT("Please invoke yt_initialize() before calling %s()!\n", __FUNCTION__);
    }

#ifdef INTERACTIVE_MODE
    // always run m_Run = -1 function, and set 1.
    // always run unknown function and let Python generates error.
    int func_index = g_func_status_list.get_func_index(function_name);
    if (func_index != -1) {
        if (g_func_status_list[func_index].get_run() == 0) {
            log_info("YT inline function \"%s\" was set to idle ... idle\n", function_name);
            return YT_SUCCESS;
        }
        else if (g_func_status_list[func_index].get_run() == -1) g_func_status_list[func_index].set_run(1);
    }
    else {
        g_func_status_list.add_new_func(function_name, 1);
        func_index = g_func_status_list.get_func_index(function_name);
    }
    g_func_status_list[func_index].set_status(-2);
#endif

    // start running inline function when every rank come to this stage.
    MPI_Barrier(MPI_COMM_WORLD);

    // join function name and input arguments
    va_list Args;
    va_start(Args, argc);
    std::string str_function(std::string(function_name) + std::string("("));
    for (int i = 0; i < argc; i++) {
        if (i != 0) str_function += std::string(",");
        str_function += std::string(va_arg(Args, const char*));
    }
    str_function += std::string(")");
    va_end(Args);

    // join function and input arguments into string wrapped by exec()
    std::string str_CallYT(std::string("exec(\"") + str_function + std::string("\", sys.modules[\"") 
                           + std::string(g_param_libyt.script) + std::string("\"].__dict__)"));

    log_info("Performing YT inline analysis %s ...\n", str_function.c_str());

#ifdef INTERACTIVE_MODE
    std::string str_CallYT_TryExcept;
    str_CallYT_TryExcept = std::string("try:\n") +
                           std::string("    ") + str_CallYT + std::string("\n") +
                           std::string("except Exception as e:\n"
                                       "    libyt.interactive_mode[\"func_err_msg\"][\"") + 
                           std::string(function_name) + std::string("\"] = traceback.format_exc()\n");
#endif // #ifdef INTERACTIVE_MODE

#ifdef INTERACTIVE_MODE
    if (PyRun_SimpleString(str_CallYT_TryExcept.c_str()) != 0)
#else
    if (PyRun_SimpleString(str_CallYT.c_str()) != 0)
#endif // #ifdef INTERACTIVE_MODE
    {
#ifdef INTERACTIVE_MODE
        g_func_status_list[func_index].set_status(0);
#endif
        YT_ABORT("Invoking \"%s\" ... failed\n", str_function.c_str());
    }

#ifdef INTERACTIVE_MODE
    // update status in g_func_status_list
    g_func_status_list[func_index].get_status();
#endif

    log_info("Performing YT inline analysis %s ... done.\n", str_function.c_str());

#ifdef SUPPORT_TIMER
    g_timer->record_time(function_name, 1);
#endif

    return YT_SUCCESS;
}

//-------------------------------------------------------------------------------------------------------
// Function    :  yt_run_Function
// Description :  Execute the YT inline analysis script
//
// Note        :  1. Python script name, which is also its namespace's name is stored in "g_param_libyt.script"
//                2. This python script must contain function of <function_name> you called.
//                3. Must give argc (argument count), even if there are no arguments.
//                4. Under INTERACTIVE_MODE, function will be wrapped inside try/except. If there is error
//                   it will store under libyt.interactive_mode["func_err_msg"]["func_name"].
//
// Parameter   :  const char *function_name : function name in python script
//
// Return      :  YT_SUCCESS or YT_FAIL
//-------------------------------------------------------------------------------------------------------
int yt_run_Function(const char *function_name) {

#ifdef SUPPORT_TIMER
    g_timer->record_time(function_name, 0);
#endif

    // check if libyt has been initialized
    if (!g_param_libyt.libyt_initialized) {
        YT_ABORT("Please invoke yt_initialize() before calling %s()!\n", __FUNCTION__);
    }

#ifdef INTERACTIVE_MODE
    // always run m_Run = -1 function, and set 1.
    // always run unknown function and let Python generates error.
    int func_index = g_func_status_list.get_func_index(function_name);
    if (func_index != -1) {
        if (g_func_status_list[func_index].get_run() == 0) {
            log_info("YT inline function \"%s\" was set to idle ... idle\n", function_name);
            return YT_SUCCESS;
        }
        else if (g_func_status_list[func_index].get_run() == -1) g_func_status_list[func_index].set_run(1);
    }
    else {
        g_func_status_list.add_new_func(function_name, 1);
        func_index = g_func_status_list.get_func_index(function_name);
    }
    g_func_status_list[func_index].set_status(-2);
#endif

    // start running inline function when every rank come to this stage.
    MPI_Barrier(MPI_COMM_WORLD);

    // join function into string wrapped by exec()
    std::string str_function(std::string(function_name) + std::string("()"));
    std::string str_CallYT(std::string("exec(\"") + str_function + std::string("\", sys.modules[\"") 
                           + std::string(g_param_libyt.script) + std::string("\"].__dict__)"));

    log_info("Performing YT inline analysis %s ...\n", str_function.c_str());

#ifdef INTERACTIVE_MODE
    std::string str_CallYT_TryExcept;
    str_CallYT_TryExcept = std::string("try:\n") +
                           std::string("    ") + str_CallYT + std::string("\n") +
                           std::string("except Exception as e:\n"
                                       "    libyt.interactive_mode[\"func_err_msg\"][\"") + 
                           std::string(function_name) + std::string("\"] = traceback.format_exc()\n");
#endif // #ifdef INTERACTIVE_MODE

#ifdef INTERACTIVE_MODE
    if (PyRun_SimpleString(str_CallYT_TryExcept.c_str()) != 0)
#else
    if (PyRun_SimpleString(str_CallYT.c_str()) != 0)
#endif // #ifdef INTERACTIVE_MODE
    {
#ifdef INTERACTIVE_MODE
        g_func_status_list[func_index].set_status(0);
#endif
        YT_ABORT("Invoking \"%s\" ... failed\n", str_function.c_str());
    }

#ifdef INTERACTIVE_MODE
    // update status in g_func_status_list
    g_func_status_list[func_index].get_status();
#endif

    log_info("Performing YT inline analysis %s ... done.\n", str_function.c_str());

#ifdef SUPPORT_TIMER
    g_timer->record_time(function_name, 1);
#endif

    return YT_SUCCESS;
}