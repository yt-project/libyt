#include "libyt.h"
#include "yt_combo.h"

#ifdef INTERACTIVE_MODE
#include <readline/readline.h>
#include <sys/stat.h>

#include <cctype>
#include <chrono>
#include <fstream>
#include <iostream>
#include <string>
#include <thread>

#include "LibytProcessControl.h"
#include "define_command.h"

static bool detect_file(const char* flag_file);
#endif

//-------------------------------------------------------------------------------------------------------
// Function    :  yt_run_ReloadScript
// Description :  Reload script in interactive mode
//
// Note        :  1. Basically, this API and yt_run_InteractiveMode has the same feature, but this one
//                   uses file base to interact with the prompt. This is a minimalist API for reloading script.
//                2. Instead of input each line through readline, it simply loads each line in a file line
//                   by-line.
//                3. It stops and enters the API if flag file is detected or there is an error in in situ
//                   analysis. In the latter case, it will generate the flag file to indicate it enters
//                   the mode, and will remove the flag file once it exits the API.
//                4. Using <reload> and <reload>_EXIT to control.
//
// Parameter   :  const char *flag_file_name : indicates if it will enter reload script mode or not (see 3.)
//                const char *reload         : a series of flag file for reload (ex: <reload>_EXIT/SUCCESS/FAILED)
//                const char *script_name    : full script name to reload
//
// Return      :  YT_SUCCESS or YT_FAIL
//-------------------------------------------------------------------------------------------------------
int yt_run_ReloadScript(const char* flag_file_name, const char* reload, const char* script_name) {
    SET_TIMER(__PRETTY_FUNCTION__);

#ifndef INTERACTIVE_MODE
    log_error("Cannot reload script. Please compile libyt with -DINTERACTIVE_MODE.\n");
    return YT_FAIL;
#else
    // check if libyt has been initialized
    if (!LibytProcessControl::Get().libyt_initialized) {
        YT_ABORT("Please invoke yt_initialize() before calling %s()!\n", __FUNCTION__);
    }

    fflush(stdout);
    fflush(stderr);

    // run new added function and output func_status summary
    if (g_func_status_list.run_func() != YT_SUCCESS)
        YT_ABORT("Something went wrong when running new added functions\n");
    if (g_func_status_list.print_summary() != YT_SUCCESS)
        YT_ABORT("Something went wrong when summarizing inline function status\n");

    // check if we need to enter reload script phase
    bool remove_flag_file = false;
    if (!detect_file(flag_file_name)) {
        bool enter_reload = false;
        for (int i = 0; i < g_func_status_list.size(); i++) {
            if ((g_func_status_list[i].get_run() == 1) && (g_func_status_list[i].get_status() == 0)) {
                enter_reload = true;
                break;
            }
        }

        // if every function works fine, leave reloading script mode,
        // otherwise create flag file to indicate it enters the mode
        if (!enter_reload) {
            log_info("No failed inline functions ... leaving reload script mode\n");
            return YT_SUCCESS;
        } else {
            std::ofstream generate_flag_file(flag_file_name);
            generate_flag_file.close();
            remove_flag_file = true;
            log_info("Generating '%s' because there are errors in inline functions ... entering reload script mode\n",
                     flag_file_name);
        }
    } else {
        log_info("Flag file '%s' is detected ... entering reload script mode\n", flag_file_name);
    }

    // make sure every process has reached here
    fflush(stdout);
    fflush(stderr);
#ifndef SERIAL_MODE
    MPI_Barrier(MPI_COMM_WORLD);
#endif

    // setting up reading file
    bool done = false;
    int root = g_myroot;
    std::string reload_exit = std::string(reload) + std::string("_EXIT");

    // enter reloading loop
    while (!done) {
        // block and detect <reload> or <reload>_EXIT every 2 sec
        bool get_reload_state = false;
        while (!get_reload_state) {
            if (detect_file(reload)) {
                get_reload_state = true;
            } else if (detect_file(reload_exit.c_str())) {
                get_reload_state = true;
                done = true;
            }

            log_info("Create '%s' file to reload script, or create '%s' file to exit.\n", reload, reload_exit.c_str());
            std::this_thread::sleep_for(std::chrono::seconds(2));
        }

        if (done) {
            log_info("Detect '%s' file ... exiting reload script\n", reload_exit.c_str());
            if (detect_file(reload_exit.c_str())) {
                std::remove(reload_exit.c_str());
            }
            break;
        } else {
            log_info("Detect '%s' file ... reloading '%s' script\n", reload, script_name);
        }

        // reloading file
        if (g_myrank == root) {
        }
#ifndef SERIAL_MODE
        else {
        }
#endif

        // remove reload flag file when done
        if (detect_file(reload)) {
            std::remove(reload);
        }
    }

    // remove flag file if it is generated by libyt because of error occurred in inline functions
    if (remove_flag_file && detect_file(flag_file_name)) {
        std::remove(flag_file_name);
    }

    log_info("Exit reload script\n");

    return YT_SUCCESS;
#endif  // #ifndef INTERACTIVE_MODE
}

//-------------------------------------------------------------------------------------------------------
// Function    :  detect_file
// Description :  A private function for detecting if file exists.
//
// Parameter   :  const char *flag_file : check if this file exist
//
// Return      :  true/false
//-------------------------------------------------------------------------------------------------------
static bool detect_file(const char* flag_file) {
    struct stat buffer {};
    if (stat(flag_file, &buffer) != 0) {
        return false;
    } else {
        return true;
    }
}
