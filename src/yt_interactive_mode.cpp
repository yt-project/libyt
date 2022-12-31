#ifdef INTERACTIVE_MODE

#include "yt_combo.h"
#include "libyt.h"
#include <readline.h>
#include <history.h>

//-------------------------------------------------------------------------------------------------------
// Function    :  yt_interactive_mode
// Description :  Enter libyt interactive mode.
//
// Note        :  1. Only enter this mode when inline functions have errors or flag_file_name is detacted.
//                2. Display inline script execute result finished/failed, and show errors if have.
//                3. Enter interactive mode, user will be operating in inline script's name space.
//                   (1) Python scripting
//                   (2) TODO: libyt command
//                   (3) Execute charactars should be less than INT_MAX.
//                4. TODO: Let user add and decide what inline function to run in the follow process.
//
// Parameter   :  char *flag_file_name : once this file is detacted, it enters yt_interactive_mode
//
// Return      :  YT_SUCCESS or YT_FAIL
//-------------------------------------------------------------------------------------------------------
int yt_interactive_mode(char* flag_file_name) {
    fflush(stdout);
    fflush(stderr);

    // output func_status summary
    if (g_func_status_list.print_summary() != YT_SUCCESS) {
        YT_ABORT("Something went wrong when summarizing inline function status\n");
    }

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

            // parse input
            if (strcmp(input_line, "exit") == 0) { // todo libyt command
                done = true;

                // make sure code and input_line is freed
                if (code != NULL) free(code);
                if (input_line != NULL) free(input_line);

                // inform other ranks that we're done
                int temp = -1;
                MPI_Bcast(&temp, 1, MPI_INT, root, MPI_COMM_WORLD);
            }
            else {
                // get input line length and previous stored code length
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
                src = Py_CompileString(code, "<libyt-stdin>", Py_single_input); // todo: add rank num
                // case 1: it works fine, ready to run
                if (src != NULL) {
                    if (prompt == ps1 || code[code_len + input_len - 1] == '\n') {
                        // broadcast to other ranks, code character num no longer than INT_MAX
                        int temp = (int) strlen(code);
                        MPI_Bcast(&temp, 1, MPI_INT, root, MPI_COMM_WORLD);
                        MPI_Bcast(code, strlen(code), MPI_CHAR, root, MPI_COMM_WORLD);

                        // run code
                        dum = PyEval_EvalCode(src, global_var, local_var);
                        if (PyErr_Occurred()) PyErr_Print();

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
                    // TODO: more msg to handle code not complete, ex: triple-quote
                    if (strcmp(err_msg, "unexpected EOF while parsing") == 0) {
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
        }
        // other ranks: get and execute python line from root
        else {
            // get code length; if code_len < 0, break the loop
            MPI_Bcast(&code_len, 1, MPI_INT, root, MPI_COMM_WORLD);
            if (code_len < 0) {
                done = true;
                break;
            }

            // get code; additional 1 for '\0'
            code = (char*) malloc((code_len + 1) * sizeof(char));
            MPI_Bcast(code, code_len, MPI_CHAR, root, MPI_COMM_WORLD);
            code[code_len] = '\0';

            // compile and execute code
            src = Py_CompileString(code, "<libyt-stdin>", Py_single_input);  // todo: add rank num
            dum = PyEval_EvalCode(src, global_var, local_var);
            if (PyErr_Occurred()) PyErr_Print();

            // clean up
            free(code);
            Py_XDECREF(dum);
            Py_XDECREF(src);

            // wait
            fflush(stdout);
            fflush(stderr);
            MPI_Barrier(MPI_COMM_WORLD);
        }

    }

    return YT_SUCCESS;
}

#endif // #ifdef INTERACTIVE_MODE
