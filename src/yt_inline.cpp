#include "yt_combo.h"
#include <stdarg.h>
#include "libyt.h"


//-------------------------------------------------------------------------------------------------------
// Function    :  yt_inline_argument
// Description :  Execute the YT inline analysis script
//
// Note        :  1. Python script name is stored in "g_param_libyt.script"
//                2. This python script must contain function of <function_name> you called.
//                3. Must give argc (argument count), even if there are no arguments.
//                4. Under INTERACTIVE_MODE, function will be wrapped inside try/except. If there is error
//                   it will store under libyt.interactive_mode["func_err_msg"]["func_name"].
//
// Parameter   :  char *function_name : function name in python script 
//                int  argc           : input arguments count
//                ...                 : list of arguments, should be input as (char*)
//
// Return      :  YT_SUCCESS or YT_FAIL
//-------------------------------------------------------------------------------------------------------
int yt_inline_argument(char *function_name, int argc, ...) {

#ifdef SUPPORT_TIMER
    g_timer->record_time(function_name, 0);
#endif

    // check if libyt has been initialized
    if (!g_param_libyt.libyt_initialized) {
        YT_ABORT("Please invoke yt_init() before calling %s()!\n", __FUNCTION__);
    }

#ifdef INTERACTIVE_MODE
    // get index in g_func_status_list,
    // if it does not exist, add new one, and always run new function.
    // if it exists and get_run() return false, return directly.
    int func_index = g_func_status_list.get_func_index(function_name);
    if (func_index == -1) {
        g_func_status_list.add_new_func(function_name, 1);
        func_index = g_func_status_list.get_func_index(function_name);
    }
    else {
        if (g_func_status_list[func_index].get_run() == false) {
            log_info("YT inline function \"%s\" was set to idle ... idle\n", function_name);
            return YT_SUCCESS;
        }
    }
    g_func_status_list[func_index].set_status(-2);
#endif

    // start running inline function when every rank come to this stage.
    MPI_Barrier(MPI_COMM_WORLD);
    log_info("Performing YT inline analysis ...\n");

    va_list Args, Args_len;
    va_start(Args, argc);
    va_copy(Args_len, Args);

    // Count inline function width = .<function_name>() + '\0'
    int InlineFunctionWidth = strlen(function_name) + 4;
    for (int i = 0; i < argc; i++) {
        if (i != 0) InlineFunctionWidth++; // comma "," in called function
        InlineFunctionWidth = InlineFunctionWidth + strlen(va_arg(Args_len, char * ));
    }

    // Allocate command, and connect input arguments
    const int CallYT_CommandWidth = strlen(g_param_libyt.script) + InlineFunctionWidth;
    char *CallYT = (char *) malloc(CallYT_CommandWidth * sizeof(char));
    strcpy(CallYT, g_param_libyt.script);
    strcat(CallYT, ".");
    strcat(CallYT, function_name);
    strcat(CallYT, "(");
    for (int i = 0; i < argc; i++) {
        if (i != 0) strcat(CallYT, ",");
        strcat(CallYT, va_arg(Args, char * ));
    }
    strcat(CallYT, ")");

    va_end(Args_len);
    va_end(Args);

#ifdef INTERACTIVE_MODE
    // put command in try ... except
    // 101 -> try: ... except: ...
    //   4 -> newline
    // scipt_name.func_name(...) -> strlen(script_name) + InlineFunctionWidth(contain '\0')
    // func_name -> strlen(function_name)
    char *CallYT_TryExcept = (char *) malloc((101 + 4 + strlen(g_param_libyt.script) + strlen(function_name) + InlineFunctionWidth)
                                             * sizeof(char));
    sprintf(CallYT_TryExcept, "try:\n"
                              "    %s\n"
                              "except Exception as e:\n"
                              "    libyt.interactive_mode[\"func_err_msg\"][\"%s\"] = traceback.format_exc()\n",
                              CallYT, function_name);
#endif // #ifdef INTERACTIVE_MODE

#ifdef INTERACTIVE_MODE
    if (PyRun_SimpleString(CallYT_TryExcept) == 0) log_debug("Invoking \"%s\" in interactive mode ... done\n", CallYT);
#else
    if (PyRun_SimpleString(CallYT) == 0)           log_debug("Invoking \"%s\" ... done\n", CallYT);
#endif // #ifdef INTERACTIVE_MODE
    else{
#ifdef INTERACTIVE_MODE
        g_func_status_list[func_index].set_status(0);
#endif
        YT_ABORT("Invoking \"%s\" ... failed\n", CallYT);
    }

#ifdef INTERACTIVE_MODE
    // update status in g_func_status_list
    g_func_status_list[func_index].get_status();
#endif
    log_info("Performing YT inline analysis <%s> ... done.\n", CallYT);

    free(CallYT);
#ifdef INTERACTIVE_MODE
    free(CallYT_TryExcept);
#endif

#ifdef SUPPORT_TIMER
    g_timer->record_time(function_name, 1);
#endif

    return YT_SUCCESS;
}

//-------------------------------------------------------------------------------------------------------
// Function    :  yt_inline
// Description :  Execute the YT inline analysis script
//
// Note        :  1. Python script name is stored in "g_param_libyt.script"
//                2. This python script must contain function of <function_name> you called.
//                3. This python function must not contain input arguments.
//                4. Under INTERACTIVE_MODE, function will be wrapped inside try/except. If there is error
//                   it will store under libyt.interactive_mode["func_err_msg"]["func_name"].
//
// Parameter   :  char *function_name : function name in python script
//
// Return      :  YT_SUCCESS or YT_FAIL
//-------------------------------------------------------------------------------------------------------
int yt_inline(char *function_name) {
#ifdef SUPPORT_TIMER
    g_timer->record_time(function_name, 0);
#endif

    // check if libyt has been initialized
    if (!g_param_libyt.libyt_initialized) {
        YT_ABORT("Please invoke yt_init() before calling %s()!\n", __FUNCTION__);
    }

#ifdef INTERACTIVE_MODE
    // get index in g_func_status_list,
    // if it does not exist, add new one, and always run new function.
    // if it exists and get_run() return false, return directly.
    int func_index = g_func_status_list.get_func_index(function_name);
    if (func_index == -1) {
        g_func_status_list.add_new_func(function_name, 1);
        func_index = g_func_status_list.get_func_index(function_name);
    }
    else {
        if (g_func_status_list[func_index].get_run() == false) {
            log_info("YT inline function \"%s\" was set to idle ... idle\n", function_name);
            return YT_SUCCESS;
        }
    }
    g_func_status_list[func_index].set_status(-2);
#endif

    // start running inline function when every rank come to this stage.
    MPI_Barrier(MPI_COMM_WORLD);
    log_info("Performing YT inline analysis ...\n");

    int InlineFunctionWidth = strlen(function_name) + 4; // width = .<function_name>() + '\0'
    const int CallYT_CommandWidth = strlen(g_param_libyt.script) + InlineFunctionWidth;
    char *CallYT = (char *) malloc(CallYT_CommandWidth * sizeof(char));
    sprintf(CallYT, "%s.%s()", g_param_libyt.script, function_name);

#ifdef INTERACTIVE_MODE
    // put command in try ... except
    // 101 -> try: ... except: ...
    //   4 -> newline
    // scipt_name.func_name() -> strlen(script_name) + InlineFunctionWidth(contain '\0')
    // func_name -> strlen(function_name)
    char *CallYT_TryExcept = (char *) malloc((101 + 4 + strlen(g_param_libyt.script) + strlen(function_name) + InlineFunctionWidth)
                                             * sizeof(char));
    sprintf(CallYT_TryExcept, "try:\n"
                              "    %s\n"
                              "except Exception as e:\n"
                              "    libyt.interactive_mode[\"func_err_msg\"][\"%s\"] = traceback.format_exc()\n",
                              CallYT, function_name);
#endif // #ifdef INTERACTIVE_MODE

#ifdef INTERACTIVE_MODE
    if (PyRun_SimpleString(CallYT_TryExcept) == 0) log_debug("Invoking \"%s\" in interactive mode ... done\n", CallYT);
#else
    if (PyRun_SimpleString(CallYT) == 0)           log_debug("Invoking \"%s\" ... done\n", CallYT);
#endif // #ifdef INTERACTIVE_MODE
    else{
#ifdef INTERACTIVE_MODE
        g_func_status_list[func_index].set_status(0);
#endif
        YT_ABORT("Invoking \"%s\" ... failed\n", CallYT);
    }

#ifdef INTERACTIVE_MODE
    // update status in g_func_status_list
    g_func_status_list[func_index].get_status();
#endif
    log_info("Performing YT inline analysis <%s> ... done.\n", CallYT);

    free(CallYT);
#ifdef INTERACTIVE_MODE
    free(CallYT_TryExcept);
#endif

#ifdef SUPPORT_TIMER
    g_timer->record_time(function_name, 1);
#endif

    return YT_SUCCESS;
}