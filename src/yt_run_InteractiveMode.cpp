#include "libyt.h"
#include "yt_combo.h"

#ifdef INTERACTIVE_MODE
#include <readline/readline.h>
#include <sys/stat.h>

#include <cctype>
#include <iostream>
#include <string>

#include "LibytProcessControl.h"
#include "function_info.h"
#include "magic_command.h"
#endif

//-------------------------------------------------------------------------------------------------------
// Function    :  yt_run_InteractiveMode
// Description :  Enter libyt interactive mode.
//
// Note        :  1. Only enter this mode flag_file_name is detected.
//                2. Display inline script execute result success/failed.
//                3. Enter interactive mode, user will be operating in inline script's name space.
//                   (1) Python scripting
//                   (2) libyt command
//                   (3) Execute characters should be less than INT_MAX.
//                4. Let user add and decide what inline function to run in the follow process.
//                5. This API deals with user inputs, and direct jobs to other functions.
//
// Parameter   :  const char *flag_file_name : once this file is detected, it enters interactive mode
//
// Return      :  YT_SUCCESS or YT_FAIL
//-------------------------------------------------------------------------------------------------------
int yt_run_InteractiveMode(const char* flag_file_name) {
    SET_TIMER(__PRETTY_FUNCTION__);

#ifndef INTERACTIVE_MODE
    log_error("Cannot enter interactive prompt. "
              "Please compile libyt with -DINTERACTIVE_MODE.\n");
    return YT_FAIL;
#else
    // check if libyt has been initialized
    if (!LibytProcessControl::Get().libyt_initialized) {
        YT_ABORT("Please invoke yt_initialize() before calling %s()!\n", __FUNCTION__);
    }

    fflush(stdout);
    fflush(stderr);

    int mpi_root = LibytProcessControl::Get().mpi_root_;
    int mpi_rank = LibytProcessControl::Get().mpi_rank_;
    int mpi_size = LibytProcessControl::Get().mpi_size_;

    // run new added function and output func_status summary
    g_func_status_list.RunEveryFunction();
    MagicCommand command(MagicCommand::EntryPoint::kLibytInteractiveMode);
    MagicCommandOutput command_result = command.Run("%libyt status");
    if (mpi_root == mpi_rank) {
        std::cout << command_result.output << std::endl;
    }

    // enter interactive mode only when flag file is detected
    struct stat buffer;
    if (stat(flag_file_name, &buffer) != 0) {
        log_info("Flag file '%s' is not detected ... leaving interactive mode\n", flag_file_name);
        return YT_SUCCESS;
    } else {
        log_info("Flag file '%s' is detected ... entering interactive mode\n", flag_file_name);
    }

    // create prompt interface
    const char* ps1 = ">>> ";
    const char* ps2 = "... ";
    const char* prompt = ps1;
    bool done = false;
    int root = mpi_root;

    // input line and error msg
    int input_len, code_len;
    char *input_line, *code = NULL;

    // make sure every rank has reach here
    fflush(stdout);
    fflush(stderr);
#ifndef SERIAL_MODE
    MPI_Barrier(MPI_COMM_WORLD);
#endif

    // enter interactive loop
    while (!done) {
        // root: prompt and input
        if (mpi_rank == root) {
            input_line = readline(prompt);
            if (input_line == NULL) continue;

            if (prompt == ps1) {
                // check if it contains spaces only or null line if prompt >>>, otherwise python will count as
                // not finished yet.
                long first_char = -1;
                for (long i = 0; i < (long)strlen(input_line); i++) {
                    if (isspace(input_line[i]) == 0) {
                        first_char = i;
                        break;
                    }
                }
                if (first_char == -1) {
                    free(input_line);
                    continue;
                }

                // assume it was libyt defined command if the first char is '%'
                if (input_line[first_char] == '%') {
#ifndef SERIAL_MODE
                    // Send call libyt define command code (indicator = 0)
                    int indicator = 0;
                    MPI_Bcast(&indicator, 1, MPI_INT, root, MPI_COMM_WORLD);
#endif
                    // run libyt command
                    command_result = command.Run(&(input_line[first_char]));
                    std::cout << command_result.output << std::endl;
                    std::cout << command_result.error << std::endl;
                    done = command_result.exit_entry_point;
#ifndef SERIAL_MODE
                    MPI_Barrier(MPI_COMM_WORLD);
#endif

                    // clean up
                    free(input_line);
                    continue;
                }
            }

            // dealing with Python input
            input_len = strlen(input_line);
            if (code == NULL) {
                code_len = 0;
            } else {
                code_len = strlen(code);
            }

            // append input to code, additional 2 for '\n' and '\0', no longer need input_line
            code = (char*)realloc(code, input_len + code_len + 2);
            if (code_len == 0) code[0] = '\0';
            strncat(code, input_line, input_len);
            code[code_len + input_len] = '\n';
            code[code_len + input_len + 1] = '\0';
            free(input_line);

            // check validity
            CodeValidity code_validity = LibytPythonShell::check_code_validity(std::string(code), true);
            if (code_validity.is_valid.compare("complete") == 0) {
                // is complete and is a single-line statement or second \n for multi-line statement
                if (prompt == ps1 || code[code_len + input_len - 1] == '\n') {
#ifndef SERIAL_MODE
                    // Send call libyt execute code (indicator = 1)
                    int indicator = 1;
                    MPI_Bcast(&indicator, 1, MPI_INT, root, MPI_COMM_WORLD);
#endif

                    // Execute code and print result
                    std::array<AccumulatedOutputString, 2> output = LibytPythonShell::execute_prompt(std::string(code));
                    for (int i = 0; i < 2; i++) {
                        if (output[i].output_string.length() > 0) {
                            int offset = 0;
                            for (int r = 0; r < mpi_size; r++) {
                                printf("\033[1;34m[MPI Process %d]\033[0;37m\n", r);
                                if (output[i].output_length[r] == 0) {
                                    printf("(None)\n");
                                }
                                printf("%s\n",
                                       output[i].output_string.substr(offset, output[i].output_length[r]).c_str());
                                offset += output[i].output_length[r];
                            }
                        }
                    }

                    // Reset
                    free(code);
                    code = NULL;
                    prompt = ps1;
                    fflush(stdout);
                    fflush(stderr);
#ifndef SERIAL_MODE
                    MPI_Barrier(MPI_COMM_WORLD);
#endif
                }
            } else if (code_validity.is_valid.compare("incomplete") == 0) {
                prompt = ps2;
            } else {
                // Print error
                printf("%s\n", code_validity.error_msg.c_str());

                // Reset
                free(code);
                code = NULL;
                prompt = ps1;
            }
        }
#ifndef SERIAL_MODE
        // other ranks: get and execute python line from root
        else {
            // TODO: (this is a bad practice.) Get code for further instructions
            int indicator = -1;
            MPI_Bcast(&indicator, 1, MPI_INT, root, MPI_COMM_WORLD);

            if (indicator == 0) {
                // call libyt command, if indicator is 0
                command_result = command.Run();
                done = command_result.exit_entry_point;
            } else {
                // Execute code, the code must be a vaild code and successfully compile now
                std::array<AccumulatedOutputString, 2> temp_output = LibytPythonShell::execute_prompt();
            }

            // clean up and wait
            fflush(stdout);
            fflush(stderr);
            MPI_Barrier(MPI_COMM_WORLD);
        }
#endif  // #ifndef SERIAL_MODE
    }

    return YT_SUCCESS;
#endif  // #ifndef INTERACTIVE_MODE
}
