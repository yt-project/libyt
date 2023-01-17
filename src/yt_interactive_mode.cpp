#ifdef INTERACTIVE_MODE

#include "yt_combo.h"
#include "define_command.h"
#include "libyt.h"
#include <ctype.h>
#include <readline/readline.h>
#include <readline/history.h>


//-------------------------------------------------------------------------------------------------------
// Function    :  yt_interactive_mode
// Description :  Enter libyt interactive mode.
//
// Note        :  1. Only enter this mode when inline functions have errors or flag_file_name is detacted.
//                2. Display inline script execute result finished/failed, and show errors if have.
//                3. Enter interactive mode, user will be operating in inline script's name space.
//                   (1) Python scripting
//                   (2) libyt command
//                   (3) Execute charactars should be less than INT_MAX.
//                4. Let user add and decide what inline function to run in the follow process.
//
// Parameter   :  char *flag_file_name : once this file is detacted, it enters yt_interactive_mode
//
// Return      :  YT_SUCCESS or YT_FAIL
//-------------------------------------------------------------------------------------------------------
int yt_interactive_mode(char* flag_file_name) {
    fflush(stdout);
    fflush(stderr);

    // run new added function and output func_status summary
    if (g_func_status_list.run_func() != YT_SUCCESS) YT_ABORT("Something went wrong when running new added functions\n");
    if (g_func_status_list.print_summary() != YT_SUCCESS) YT_ABORT("Something went wrong when summarizing inline function status\n");

    // check if we need to enter interactive prompt
    FILE *file;
    if ( file = fopen(flag_file_name, "r") ) {
        fclose(file);
    }
    else {
        int tot_status = 0;
        for (int i=0; i<g_func_status_list.size(); i++) {
            tot_status = tot_status + g_func_status_list[i].get_status();
        }
        if (tot_status == g_func_status_list.size()) {
            log_info("No failed inline function and no stop file %s detected ... leaving interactive mode\n", flag_file_name);
            return YT_SUCCESS;
        }
    }

    // create prompt interface
    char *ps1 = ">>> ";
    char *ps2 = "... ";
    char *prompt = ps1;
    bool done = false;
    int root = 0;

    // input line and error msg
    int input_len, code_len;
    char *err_msg, *input_line, *code = NULL;

    // get inline script's namespace, globals and locals are the same.
    PyObject *local_var, *global_var;
    global_var = PyDict_GetItemString(g_py_interactive_mode, "script_globals");
    local_var = global_var;

    // python object for interactive loop, parsing syntax error for code not yet done
    PyObject *src, *dum;
    PyObject *exc, *val, *traceback, *obj;

    // make sure every rank has reach here
    fflush(stdout);
    fflush(stderr);
    MPI_Barrier(MPI_COMM_WORLD);

    // enter interactive loop
    while (!done) {

        // root: prompt and input
        if (g_myrank == root) {
            input_line = readline(prompt);
            if (input_line == NULL) continue; // todo: this does not work, it prints lots of >>>

            if (prompt == ps1) {
                // check if it contains spaces only or null line if prompt >>>, otherwise python will counted as
                // not finished yet.
                long first_char = -1;
                for (long i=0; i<strlen(input_line); i++) {
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

                    // run code, and detect if there is callables
                    dum = PyEval_EvalCode(src, global_var, local_var);
                    if (PyErr_Occurred()) PyErr_Print();
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
            else if (PyErr_ExceptionMatches(PyExc_SyntaxError)) {
                // save current exception if there is any, and parse error msg
                PyErr_Fetch(&exc, &val, &traceback);
                PyArg_ParseTuple(val, "sO", &err_msg, &obj);

                // code not complete yet
                if (strcmp(err_msg, "unexpected EOF while parsing") == 0 ||
                    strcmp(err_msg, "EOF while scanning triple-quoted string literal") == 0) {
                    prompt = ps2;
                }
                // it's a real error
                else {
                    PyErr_Restore(exc, val, traceback);
                    PyErr_Print();

                    // clean up
                    free(code);
                    code = NULL;
                    prompt = ps1;
                }

                // clean up
                Py_XDECREF(exc);
                Py_XDECREF(val);
                Py_XDECREF(traceback);
                Py_XDECREF(obj);
            }
            // case 3: real errors in code
            else{
                PyErr_Print();

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
                dum = PyEval_EvalCode(src, global_var, local_var);
                if (PyErr_Occurred()) PyErr_Print();
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
}

#endif // #ifdef INTERACTIVE_MODE
