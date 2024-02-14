#include "libyt.h"
#include "yt_combo.h"

#ifdef INTERACTIVE_MODE
#include <readline/readline.h>
#include <sys/stat.h>

#include <chrono>
#include <fstream>
#include <iostream>
#include <sstream>
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
//                4. Using <reload_file_name>_EXIT/SUCCESS/FAILED for a series of instructions.
//                   Try to avoid these instruction file names.
//                5. Run reload script in file, forcing every libyt command needs to be commented and put at
//                   the end and use #LIBYT_COMMANDS at the start.
//                   Ex:
//                     # python code ...
//                     #LIBYT_COMMANDS
//                     # %libyt status
//
// Parameter   :  const char *flag_file_name  : indicates if it will enter reload script mode or not (see 3.)
//                const char *reload_file_name: a series of flag file for reload instructions
//                const char *script_name     : full script name to reload
//
// Return      :  YT_SUCCESS or YT_FAIL
//-------------------------------------------------------------------------------------------------------
int yt_run_ReloadScript(const char* flag_file_name, const char* reload_file_name, const char* script_name) {
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
            if (g_myrank == g_myroot) {
                std::ofstream generate_flag_file(flag_file_name);
                generate_flag_file.close();
            }
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
    std::string reload_exit_filename = std::string(reload_file_name) + std::string("_EXIT");
    std::string reload_success_filename = std::string(reload_file_name) + std::string("_SUCCESS");
    std::string reload_failed_filename = std::string(reload_file_name) + std::string("_FAILED");
    std::string reloading_filename = std::string(".") + std::string(reload_file_name) + std::string("_RELOADING");

    // enter reloading loop
    while (!done) {
        // responsible for getting reload instruction and broadcast to non-root rank
        if (g_myrank == g_myroot) {
            // block and detect <reload_file_name> or <reload_file_name>_EXIT every 2 sec
            log_info("Create '%s' file to reload script, or create '%s' file to exit.\n", reload_file_name,
                     reload_exit_filename.c_str());
            bool get_reload_state = false;
            while (!get_reload_state) {
                if (detect_file(reload_file_name)) {
                    get_reload_state = true;
                } else if (detect_file(reload_exit_filename.c_str())) {
                    get_reload_state = true;
                    done = true;
                }
                std::this_thread::sleep_for(std::chrono::seconds(2));
            }

            // broadcast reload/exit instruction
            if (done) {
#ifndef SERIAL_MODE
                int indicator = -1;
                MPI_Bcast(&indicator, 1, MPI_INT, g_myroot, MPI_COMM_WORLD);
#endif
                g_libyt_python_shell.clear_prompt_history();
                log_info("Detect '%s' file ... exiting reload script\n", reload_exit_filename.c_str());
                if (detect_file(reload_exit_filename.c_str())) {
                    std::remove(reload_exit_filename.c_str());
                }
                break;
            } else {
                log_info("Detect '%s' file ... reloading '%s' script\n", reload_file_name, script_name);
            }

            // create new dumped results temporary file for reloading file
            bool reload_success = true;
            std::ofstream reload_result_file(reloading_filename.c_str());
            reload_result_file.close();

            // make sure file exists then reload
            if (!detect_file(script_name)) {
                reload_success = false;
                reload_result_file.open(reloading_filename.c_str(), std::ostream::out | std::ostream::app);
                reload_result_file << "Unable to find file '" << script_name << "' ... reload failed" << std::endl;
                reload_result_file.close();
            } else {
                // read file line-by-line
                std::ifstream reload_stream;
                reload_stream.open(script_name);
                std::string line;
                std::stringstream python_code_buffer, libyt_command_buffer;
                bool libyt_command_block = false;
                while (getline(reload_stream, line)) {
                    python_code_buffer << line << "\n";

                    if (line.find("#LIBYT_COMMANDS") != std::string::npos) {
                        libyt_command_block = true;
                    }
                    if (libyt_command_block) {
                        // store in libyt_command_buffer if it is libyt command and ignore pound keys
                        std::size_t found_first_char = line.find_first_not_of("\t\n\v\f\r ");
                        if (found_first_char != std::string::npos && line.at(found_first_char) == '#') {
                            std::size_t found_second_char = line.find_first_not_of("\t\n\v\f\r ", found_first_char + 1);
                            if (found_second_char != std::string::npos &&
                                line.substr(found_second_char, 6) == "%libyt") {
                                libyt_command_buffer
                                    << line.substr(found_second_char, line.length() - found_second_char) << "\n";
                            }
                        }
                    }
                }
                reload_stream.close();

                // check code validity then load python code
                CodeValidity code_validity =
                    LibytPythonShell::check_code_validity(python_code_buffer.str(), false, script_name);
                if (code_validity.is_valid.compare("complete") == 0) {
#ifndef SERIAL_MODE
                    int indicator = 1;
                    MPI_Bcast(&indicator, 1, MPI_INT, g_myroot, MPI_COMM_WORLD);
#endif
                    std::array<AccumulatedOutputString, 2> output =
                        LibytPythonShell::execute_file(python_code_buffer.str(), script_name);
                    reload_result_file.open(reloading_filename.c_str(), std::ostream::out | std::ostream::app);
                    for (int i = 0; i < 2; i++) {
                        if (output[i].output_string.length() > 0) {
                            int offset = 0;
                            for (int r = 0; r < g_mysize; r++) {
                                reload_result_file << "====== MPI Process " << r << (i == 0 ? "" : " - ErrorMsg")
                                                   << " ======\n";
                                if (output[i].output_length[r] == 0) {
                                    reload_result_file << "(None)\n";
                                }
                                reload_result_file
                                    << output[i].output_string.substr(offset, output[i].output_length[r]).c_str()
                                    << "\n";
                                offset += output[i].output_length[r];
                            }

                            // error output[1] length > 0, which means there is an error
                            if (i == 1) {
                                reload_success = false;
                            }
                        } else {
                            // error output[1] length == 0, which means there is no error msg
                            if (i == 1) {
                                reload_success = true;
                                LibytPythonShell::load_file_func_body(script_name);
                            }
                        }
                    }
                    reload_result_file.close();
                } else {
                    reload_success = false;
                    reload_result_file.open(reloading_filename.c_str(), std::ostream::out | std::ostream::app);
                    reload_result_file << code_validity.error_msg.c_str() << std::endl;
                    reload_result_file.close();
                }

                // Loading libyt commands, continue loading even if one of the command failed,
                // because they are independent.
                define_command command(reloading_filename);
                while (std::getline(libyt_command_buffer, line, '\n')) {
#ifndef SERIAL_MODE
                    int indicator = 0;
                    MPI_Bcast(&indicator, 1, MPI_INT, g_myroot, MPI_COMM_WORLD);
#endif
                    std::array<bool, 2> command_result = command.run(line);
                    reload_success = reload_success & command_result[1];
                }
            }

            // remove previous <reload_file_name>_SUCCESS or <reload_file_name>_FAILED and
            if (detect_file(reload_success_filename.c_str())) std::remove(reload_success_filename.c_str());
            if (detect_file(reload_failed_filename.c_str())) std::remove(reload_failed_filename.c_str());

            // rename dumped temporary file to either of them based on success or failed results.
            if (detect_file(reloading_filename.c_str())) {
                if (reload_success) {
                    std::rename(reloading_filename.c_str(), reload_success_filename.c_str());
                } else {
                    std::rename(reloading_filename.c_str(), reload_failed_filename.c_str());
                }
            } else {
                std::ofstream dump_result_file(reload_failed_filename.c_str());
                dump_result_file << "Unable to store the output when reloading the script." << std::endl;
                dump_result_file.close();

                log_info("Reloading script '%s' ... failed\n", script_name);
                log_info("See '%s' log\n", reload_failed_filename.c_str());
            }

            // remove reload_file_name flag file when done
            if (detect_file(reload_file_name)) {
                std::remove(reload_file_name);
            }
        }
#ifndef SERIAL_MODE
        else {
            // TODO: (this is a bad practice.) Get code for further instructions
            int indicator = -2;
            MPI_Bcast(&indicator, 1, MPI_INT, g_myroot, MPI_COMM_WORLD);

            switch (indicator) {
                case -1: {
                    done = true;
                    g_libyt_python_shell.clear_prompt_history();
                    break;
                }
                case 0: {
                    define_command command(reloading_filename);
                    std::array<bool, 2> command_result = command.run();
                    break;
                }
                case 1: {
                    std::array<AccumulatedOutputString, 2> output = LibytPythonShell::execute_file();
                    if (output[1].output_string.length() <= 0) {
                        LibytPythonShell::load_file_func_body(script_name);
                    }
                    break;
                }
            }
        }
#endif
    }

    // remove flag file if it is generated by libyt because of error occurred in inline functions
    if (g_myrank == g_myroot && remove_flag_file && detect_file(flag_file_name)) {
        std::remove(flag_file_name);
    }

    log_info("Exit reloading script\n");

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
