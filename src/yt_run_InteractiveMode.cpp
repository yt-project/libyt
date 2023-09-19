#include "yt_combo.h"
#include "libyt.h"

#ifdef INTERACTIVE_MODE
#include <ctype.h>
#include "define_command.h"
#include <sys/stat.h>
#include <readline/readline.h>
#endif

//-------------------------------------------------------------------------------------------------------
// Function    :  yt_run_InteractiveMode
// Description :  Enter libyt interactive mode.
//
// Note        :  1. Only enter this mode when executed inline functions have errors or flag_file_name
//                   is detacted.
//                2. Display inline script execute result success/failed, and show errors if have.
//                3. Enter interactive mode, user will be operating in inline script's name space.
//                   (1) Python scripting
//                   (2) libyt command
//                   (3) Execute charactars should be less than INT_MAX.
//                4. Let user add and decide what inline function to run in the follow process.
//
// Parameter   :  const char *flag_file_name : once this file is detacted, it enters interactive mode
//
// Return      :  YT_SUCCESS or YT_FAIL
//-------------------------------------------------------------------------------------------------------
int yt_run_InteractiveMode(const char* flag_file_name) {

    SET_TIMER(__PRETTY_FUNCTION__);

#ifndef INTERACTIVE_MODE
    log_error("Cannot enter interactive prompt. "
              "Please compile libyt with -DINTERACTIVE_MODE\n");
    return YT_FAIL;
#else
    fflush(stdout);
    fflush(stderr);

    // run new added function and output func_status summary
    if (g_func_status_list.run_func() != YT_SUCCESS) YT_ABORT("Something went wrong when running new added functions\n");
    if (g_func_status_list.print_summary() != YT_SUCCESS) YT_ABORT("Something went wrong when summarizing inline function status\n");

    // check if we need to enter interactive prompt
    struct stat buffer;
    if (stat(flag_file_name, &buffer) != 0) {
        bool enter_interactive_mode = false;
        for (int i=0; i<g_func_status_list.size(); i++) {
            if ((g_func_status_list[i].get_run() == 1) && (g_func_status_list[i].get_status() == 0)) {
                enter_interactive_mode = true;
                break;
            }
        }

        if (!enter_interactive_mode) {
            log_info("No failed inline function and no stop file %s detected ... leaving interactive mode\n", flag_file_name);
            return YT_SUCCESS;
        }
    }

    // create prompt interface
    const char *ps1 = ">>> ";
    const char *ps2 = "... ";
    const char *prompt = ps1;
    bool done = false;
    int root = 0;

    // input line and error msg
    int input_len, code_len;
    char *input_line, *code = NULL;

    // get inline script's namespace, globals and locals are the same.
    PyObject *global_var;
    global_var = PyDict_GetItemString(g_py_interactive_mode, "script_globals");

    // python object for interactive loop, parsing syntax error for code not yet done
    PyObject *src, *dum;

    // make sure every rank has reach here
    fflush(stdout);
    fflush(stderr);
    MPI_Barrier(MPI_COMM_WORLD);

    // enter interactive loop
    while (!done) {

        // root: prompt and input
        if (g_myrank == root) {
            input_line = readline(prompt);
            if (input_line == NULL) continue;

            if (prompt == ps1) {
                // check if it contains spaces only or null line if prompt >>>, otherwise python will counted as
                // not finished yet.
                long first_char = -1;
                for (long i=0; i<(long)strlen(input_line); i++) {
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
                    // tell other ranks no matter if it is valid, even though not all libyt command are collective
                    input_len = (int) strlen(input_line) - first_char;
                    MPI_Bcast(&input_len, 1, MPI_INT, root, MPI_COMM_WORLD);
                    MPI_Bcast(&(input_line[first_char]), input_len, MPI_CHAR, root, MPI_COMM_WORLD);

                    // run libyt command
                    define_command command(&(input_line[first_char]));
                    done = command.run();
                    MPI_Barrier(MPI_COMM_WORLD);

                    // clean up
                    free(input_line);
                    continue;
                }
            }

            // dealing with Python input
            input_len = strlen(input_line);
            if (code == NULL) code_len = 0;
            else              code_len = strlen(code);

            // append input to code, additional 2 for '\n' and '\0'
            code = (char*) realloc(code, input_len + code_len + 2);
            if (code_len == 0) code[0] = '\0';
            strncat(code, input_line, input_len);
            code[code_len + input_len] = '\n';
            code[code_len + input_len + 1] = '\0';

            // compile and check if this code is complete yet
            src = Py_CompileString(code, "<libyt-stdin>", Py_single_input);
            // case 1: it works fine, ready to run
            if (src != NULL) {
                if (prompt == ps1 || code[code_len + input_len - 1] == '\n') {
                    // broadcast to other ranks, code character num no longer than INT_MAX
                    int temp = (int) strlen(code);
                    MPI_Bcast(&temp, 1, MPI_INT, root, MPI_COMM_WORLD);
                    MPI_Bcast(code, strlen(code), MPI_CHAR, root, MPI_COMM_WORLD);

                    // run code
                    dum = PyEval_EvalCode(src, global_var, global_var);
                    PyRun_SimpleString("sys.stdout.flush()");
                    if (PyErr_Occurred()) {
                        PyErr_Print();
                        PyRun_SimpleString("sys.stderr.flush()");
                    }
                    else {
                        // if it worked successfully, write to prompt history (only on root)
                        g_func_status_list.update_prompt_history(std::string(code));
                    }

                    // detect callables and their function definition
                    func_status_list::load_input_func_body(code);

                    // clean up
                    Py_XDECREF(dum);
                    free(code);
                    code = NULL;
                    prompt = ps1;

                    // wait till every rank is done
                    fflush(stdout);
                    fflush(stderr);
                    MPI_Barrier(MPI_COMM_WORLD);
                }
            }
            // case 2: not finish yet
            else if (func_status_list::is_not_done_err_msg()) {
                // code not complete yet, switch prompt to ps2
                prompt = ps2;
            }
            // case 3: real errors in code
            else{
                PyErr_Print();
                PyRun_SimpleString("sys.stderr.flush()");

                // clean up
                free(code);
                code = NULL;
                prompt = ps1;
            }

            // clean up
            Py_XDECREF(src);
            free(input_line);
        }
        // other ranks: get and execute python line from root
        else {
            // get code; additional 1 for '\0'
            MPI_Bcast(&code_len, 1, MPI_INT, root, MPI_COMM_WORLD);
            code = (char*) malloc((code_len + 1) * sizeof(char));
            MPI_Bcast(code, code_len, MPI_CHAR, root, MPI_COMM_WORLD);
            code[code_len] = '\0';

            // call libyt command, if the first char is '%'
            if (code[0] == '%') {
                define_command command(code);
                done = command.run();
            }
            else {
                // compile and execute code, and detect functors.
                src = Py_CompileString(code, "<libyt-stdin>", Py_single_input);
                dum = PyEval_EvalCode(src, global_var, global_var);
                PyRun_SimpleString("sys.stdout.flush()");
                if (PyErr_Occurred()) {
                    PyErr_Print();
                    PyRun_SimpleString("sys.stderr.flush()");
                }
                func_status_list::load_input_func_body(code);

                // clean up
                Py_XDECREF(dum);
                Py_XDECREF(src);
            }

            // clean up and wait
            free(code);
            fflush(stdout);
            fflush(stderr);
            MPI_Barrier(MPI_COMM_WORLD);
        }

    }

    return YT_SUCCESS;
#endif // #ifndef INTERACTIVE_MODE
}
