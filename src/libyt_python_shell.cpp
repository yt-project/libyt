#ifdef INTERACTIVE_MODE

#include "libyt_python_shell.h"

#include <iostream>
#include <string>

#include "yt_combo.h"

static bool check_colon_exist(const char* code);

std::array<std::string, LibytPythonShell::s_NotDone_Num> LibytPythonShell::s_NotDone_ErrMsg;
std::array<PyObject*, LibytPythonShell::s_NotDone_Num> LibytPythonShell::s_NotDone_PyErr;
PyObject* LibytPythonShell::s_PyGlobals;

//-------------------------------------------------------------------------------------------------------
// Class       :  LibytPythonShell
// Method      :  update_prompt_history / clear_prompt_history
//
// Notes       :  1. Only root rank will record all the successful inputs from interactive prompt.
//                2. Input that failed will not be recorded.
//                3. Inputs will not accumulate from iterations to iterations, which means input will be
//                   cleared after stepping out interactive mode (%libyt exit).
//
// Arguments   :  const std::string& cmd_prompt : input prompt from user.
//
// Return      :  YT_SUCCESS
//-------------------------------------------------------------------------------------------------------
int LibytPythonShell::update_prompt_history(const std::string& cmd_prompt) {
    SET_TIMER(__PRETTY_FUNCTION__);

    m_PromptHistory = m_PromptHistory + std::string("#In[") + std::to_string(m_PromptHistoryCount) + std::string("]\n");
    m_PromptHistory = m_PromptHistory + cmd_prompt + std::string("\n\n");
    m_PromptHistoryCount += 1;
    return YT_SUCCESS;
}

int LibytPythonShell::clear_prompt_history() {
    SET_TIMER(__PRETTY_FUNCTION__);

    m_PromptHistory = std::string("");
    m_PromptHistoryCount = 0;
    return YT_SUCCESS;
}

//-------------------------------------------------------------------------------------------------------
// Class         :  LibytPythonShell
// Static Method :  load_file_func_body
//
// Notes         :  1. This is a static method.
//                  2. It updates functions' body defined in filename and put it under
//                     libyt.interactive_mode["func_body"].
//                  3. Get only keyword def defined functions. If the functors are defined using __call__
//                     this method cannot grab the corresponding definition.
//
// Arguments     :  const char *filename: update function body for function defined inside filename
//
// Return        : YT_SUCCESS or YT_FAIL
//-------------------------------------------------------------------------------------------------------
int LibytPythonShell::load_file_func_body(const char* filename) {
    SET_TIMER(__PRETTY_FUNCTION__);

    int command_len = 500 + strlen(filename);
    char* command = (char*)malloc(command_len * sizeof(char));
    sprintf(command,
            "for key in libyt.interactive_mode[\"script_globals\"].keys():\n"
            "    if key.startswith(\"__\") and key.endswith(\"__\"):\n"
            "        continue\n"
            "    else:\n"
            "        var = libyt.interactive_mode[\"script_globals\"][key]\n"
            "        try:\n"
            "            if callable(var) and inspect.getsourcefile(var).split(\"/\")[-1] == \"%s\":\n"
            "                libyt.interactive_mode[\"func_body\"][key] = inspect.getsource(var)\n"
            "        except:\n"
            "            pass\n",
            filename);

    if (PyRun_SimpleString(command) == 0) {
        log_debug("Loading function body in script %s ... done\n", filename);
        free(command);
        return YT_SUCCESS;
    } else {
        log_debug("Loading function body in script %s ... failed\n", filename);
        free(command);
        return YT_FAIL;
    }
}

//-------------------------------------------------------------------------------------------------------
// Class         :  LibytPythonShell
// Static Method :  load_input_func_body
//
// Notes         :  1. This is a static method.
//                  2. Detect if there are functors defined in code object src_ptr, if yes, put it under
//                     libyt.interactive_mode["func_body"].
//                  3. It's not this method's responsibility to free code.
//                  4. To silent the printing when PyEval_EvalCode evaluates the code, that sys.stdout
//                     point to somewhere else when evaluating.
//                  5. It accepts indent size different from 4.
//                  6. TODO: It needs script's scope, otherwise some functors aren't detectable.
//                     (ex: b = np.random.rand)
//
// Arguments     :  char *code : code to detect.
//
// Return        : YT_SUCCESS or YT_FAIL
//-------------------------------------------------------------------------------------------------------
int LibytPythonShell::load_input_func_body(const char* code) {
    SET_TIMER(__PRETTY_FUNCTION__);

    // prepare subspace to silent printing from python
    PyObject* py_new_dict = PyDict_New();
    PyDict_SetItemString(py_new_dict, "__builtins__", PyEval_GetBuiltins());
    std::string command_str("import os, contextlib\n"
                            "with open(os.devnull, 'w') as f, contextlib.redirect_stdout(f):\n");
    std::string code_str(code);
    std::size_t start_pos = 0;
    while (true) {
        std::size_t found = code_str.find('\n', start_pos);
        if (found != std::string::npos) {
            command_str += "    ";
            for (std::size_t c = start_pos; c < found; c++) {
                command_str += code_str[c];
            }
            command_str += "\n";
        } else {
            break;
        }
        start_pos = found + 1;
    }

    // detecting callables
    // detecting callables: loop over keys in new dict, and check if it is callable
    PyObject* py_src = Py_CompileString(command_str.c_str(), "<libyt-stdin>", Py_file_input);
    if (py_src != NULL) {
        PyObject* py_dum_detect = PyEval_EvalCode(py_src, py_new_dict, py_new_dict);
        PyObject* py_new_dict_keys = PyDict_Keys(py_new_dict);
        Py_ssize_t py_size = PyList_GET_SIZE(py_new_dict_keys);
        for (Py_ssize_t i = 0; i < py_size; i++) {
            if (PyCallable_Check(PyDict_GetItem(py_new_dict, PyList_GET_ITEM(py_new_dict_keys, i)))) {
                // add new function to g_func_status_list and set to idle. if function exists already, get its index
                const char* func_name = PyUnicode_AsUTF8(PyList_GET_ITEM(py_new_dict_keys, i));
                g_func_status_list.add_new_func(func_name, 0);

                // update function body
                PyObject* py_func_body_dict = PyDict_GetItemString(g_py_interactive_mode, "func_body");
                PyObject* py_func_body = PyUnicode_FromString((const char*)code);
                PyDict_SetItemString(py_func_body_dict, func_name, py_func_body);
                Py_DECREF(py_func_body);
            }
        }

        // clean up
        Py_XDECREF(py_dum_detect);
        Py_DECREF(py_new_dict_keys);
    }

    // clean up, there might cause some error if it is not a functor, so clear err indicator
    Py_XDECREF(py_src);
    Py_DECREF(py_new_dict);
    PyErr_Clear();

    return YT_SUCCESS;
}

//-------------------------------------------------------------------------------------------------------
// Class         :  LibytPythonShell
// Static Method :  get_funcname_defined
//
// Notes         :  1. This is a static method.
//                  2. It grabs functions or any callable object's name defined in filename.
//
// Arguments     :  const char *filename: update function body for function defined inside filename
//
// Return        : std::vector<std::string> contains a list of function name defined in filename.
//-------------------------------------------------------------------------------------------------------
std::vector<std::string> LibytPythonShell::get_funcname_defined(const char* filename) {
    SET_TIMER(__PRETTY_FUNCTION__);

    int command_len = 400 + strlen(filename);
    char* command = (char*)malloc(command_len * sizeof(char));
    sprintf(command,
            "libyt.interactive_mode[\"temp\"] = []\n"
            "for key in libyt.interactive_mode[\"script_globals\"].keys():\n"
            "    if key.startswith(\"__\") and key.endswith(\"__\"):\n"
            "        continue\n"
            "    else:\n"
            "        var = libyt.interactive_mode[\"script_globals\"][key]\n"
            "        if callable(var) and inspect.getsourcefile(var).split(\"/\")[-1] == \"%s\":\n"
            "            libyt.interactive_mode[\"temp\"].append(key)\n",
            filename);
    if (PyRun_SimpleString(command) != 0) log_error("Unable to grab functions in python script %s.\n", filename);

    PyObject* py_func_list = PyDict_GetItemString(g_py_interactive_mode, "temp");
    Py_ssize_t py_list_len = PyList_Size(py_func_list);
    std::vector<std::string> func_list;
    func_list.reserve((long)py_list_len);
    for (Py_ssize_t i = 0; i < py_list_len; i++) {
        const char* func_name = PyUnicode_AsUTF8(PyList_GET_ITEM(py_func_list, i));
        func_list.emplace_back(std::string(func_name));
    }

    // clean up
    free(command);
    PyRun_SimpleString("del libyt.interactive_mode[\"temp\"]");

    return func_list;
}

//-------------------------------------------------------------------------------------------------------
// Class         :  LibytPythonShell
// Static Method :  set_exception_hook()
//
// Notes         :  1. This is a static method.
//                  2. Makes MPI not invoke MPI_Abort when getting errors in Python in interactive mode.
//                  3. Can only be called after libyt is initialized.
//
// Arguments     :  None
//
// Return        :  YT_SUCCESS or YT_FAIL
//-------------------------------------------------------------------------------------------------------
int LibytPythonShell::set_exception_hook() {
    SET_TIMER(__PRETTY_FUNCTION__);

    char command[600];
    sprintf(command,
            "import sys\n"
            "def mpi_libyt_interactive_mode_excepthook(exception_type, exception_value, tb):\n"
            "    traceback.print_tb(tb)\n"
            "    print(\"[YT_ERROR  ] {}: {}\".format(exception_type.__name__, exception_value), file=sys.stderr)\n"
            "    print(\"[YT_ERROR  ] Error occurred on rank {}.\".format(%d), file=sys.stderr)\n"
            "sys.excepthook = mpi_libyt_interactive_mode_excepthook\n",
            g_myrank);

    if (PyRun_SimpleString(command) == 0)
        return YT_SUCCESS;
    else
        return YT_FAIL;
}

//-------------------------------------------------------------------------------------------------------
// Class         :  LibytPythonShell
// Static Method :  init_not_done_err_msg
//
// Notes         :  1. This is a static method.
//                  2. Identify error messages that will show up when inputing statements like class, def
//                     if, and triple quotes etc.
//                  3. Error messages are version dependent.
//                  4. s_NotDone_ErrMsg's and s_NotDone_PyErr's elements are one-to-one relationship.
//
// Arguments     :  None
//
// Return        :  YT_SUCCESS or YT_FAIL
//-------------------------------------------------------------------------------------------------------
int LibytPythonShell::init_not_done_err_msg() {
    SET_TIMER(__PRETTY_FUNCTION__);

    // error msg from not done yet statement to grab
    std::array<std::string, s_NotDone_Num> not_done_statement = {
        std::string("if 1==1:\n"), std::string("tri = \"\"\"\n"), std::string("print(\n")};

    // get python error type and its statement.
    PyObject *py_src, *py_exc, *py_val, *py_traceback, *py_obj;
    const char* err_msg;
    for (int i = 0; i < s_NotDone_Num; i++) {
        py_src = Py_CompileString(not_done_statement[i].c_str(), "<get err msg>", Py_single_input);
        PyErr_Fetch(&py_exc, &py_val, &py_traceback);
        PyArg_ParseTuple(py_val, "sO", &err_msg, &py_obj);

        s_NotDone_ErrMsg[i] = std::string(err_msg);
        s_NotDone_PyErr[i] = py_exc;

        // dereference
        Py_XDECREF(py_src);
        Py_XDECREF(py_exc);
        Py_XDECREF(py_val);
        Py_XDECREF(py_traceback);
        Py_XDECREF(py_obj);
        PyErr_Clear();
    }

    return YT_SUCCESS;
}

//-------------------------------------------------------------------------------------------------------
// Class         :  LibytPythonShell
// Static Method :  init_script_namespace
//
// Notes         :  1. This is a static method.
//                  2. Initialize m_PyGlobals, which is a Python dict object that contains namespace of
//                     the script.
//                  3. It is a borrowed reference.
//
// Arguments     :  None
//
// Return        :  YT_SUCCESS or YT_FAIL
//-------------------------------------------------------------------------------------------------------
int LibytPythonShell::init_script_namespace() {
    SET_TIMER(__PRETTY_FUNCTION__);

    s_PyGlobals = PyDict_GetItemString(g_py_interactive_mode, "script_globals");

    return YT_SUCCESS;
}

//-------------------------------------------------------------------------------------------------------
// Class         :  LibytPythonShell
// Static Method :  is_not_done_err_msg
//
// Notes         :  1. This is a static method.
//                  2. Check current Python state to see if it is error msg that is caused by user input
//                     not done yet. Which means there is error buffer.
//                  3. If it is indeed caused by user not done its input, clear error buffer. If not,
//                     restore the buffer and let yt_run_InteractiveMode print the error msg.
//                  4. s_NotDone_ErrMsg's and s_NotDone_PyErr's elements are one-to-one relationship. Make
//                     sure to go through every element, since some of them might have error of same type
//                     but with different err msg.
//                  5. IndentationError (s_NotDon_PyErr[0]) can be caused by multi-line statement (with ':'),
//                     or simply an error.
//
// Arguments     :  None
//
// Return        :  true / false : true for user hasn't done inputting yet.
//-------------------------------------------------------------------------------------------------------
bool LibytPythonShell::is_not_done_err_msg(const char* code) {
    SET_TIMER(__PRETTY_FUNCTION__);

    bool user_not_done = false;

    for (int i = 0; i < s_NotDone_Num; i++) {
        // check error type
        if (PyErr_ExceptionMatches(s_NotDone_PyErr[i])) {
            // fetch err msg
            PyObject *py_exc, *py_val, *py_traceback, *py_obj;
            const char* err_msg = "";
            PyErr_Fetch(&py_exc, &py_val, &py_traceback);
            PyArg_ParseTuple(py_val, "sO", &err_msg, &py_obj);

            // check error msg
            if (s_NotDone_ErrMsg[i].compare(err_msg) == 0) {
                // if it is IndentationError (i == 0), then check if the last line contains ':' (a multi-line statement)
                if (i == 0) {
                    user_not_done = check_colon_exist(code);
                } else {
                    user_not_done = true;
                }
            }

            if (user_not_done) {
                // decrease reference error msg
                Py_XDECREF(py_exc);
                Py_XDECREF(py_val);
                Py_XDECREF(py_traceback);
                Py_XDECREF(py_obj);
                break;
            } else {
                // restore err msg, and I no longer own py_exc's, py_val's, and py_traceback's reference
                PyErr_Restore(py_exc, py_val, py_traceback);
            }
        }
    }

    return user_not_done;
}

//-------------------------------------------------------------------------------------------------------
// Class         :  LibytPythonShell
// Static Method :  execute_cell
// Description   :  Execute code get from cell in Jupyter Notebook
//
// Notes       :  1. This is a collective operation, requires every rank to call this function.
//                   Assuming every MPI process enter this function at the same state the same time.
//                2. Root rank will gather stdout and stderr from non-root rank, so the string returned
//                   contains each ranks dumped output in root, and non-root rank only returns output from
//                   itself.
//                3. This method is called by LibytWorker::start and LibytKernel::execute_request_impl,
//                   It is used by Jupyter Notebook access.
//                4. Root rank will pass in code, cell name; Non-root ranks only need to wait.
//
// Arguments   :  const std::array<std::string, 2>& code_split : code with upper and lower half
//                int                             cell_counter : cell counter on the left
//
// Return      :  std::array<std::string, 2> output[0] : stdout
//                                           output[1] : stderr
//-------------------------------------------------------------------------------------------------------
std::array<std::string, 2> LibytPythonShell::execute_cell(const std::array<std::string, 2>& code_split,
                                                          int cell_counter) {
    SET_TIMER(__PRETTY_FUNCTION__);

#ifndef SERIAL_MODE
    // Get code_split and cell name from root rank
    unsigned long code_len[2] = {code_split[0].length(), code_split[1].length()};
    MPI_Bcast(&code_len[0], 1, MPI_UNSIGNED_LONG, g_myroot, MPI_COMM_WORLD);
    MPI_Bcast(&code_len[1], 1, MPI_UNSIGNED_LONG, g_myroot, MPI_COMM_WORLD);

    char* code_get[2];
    if (g_myrank == g_myroot) {
        for (int i = 0; i < 2; i++) {
            MPI_Bcast((void*)code_split[i].c_str(), (int)code_len[i], MPI_CHAR, g_myroot, MPI_COMM_WORLD);
        }
    } else {
        for (int i = 0; i < 2; i++) {
            code_get[i] = new char[code_len[i] + 1];
            MPI_Bcast((void*)code_get[i], (int)code_len[i], MPI_CHAR, g_myroot, MPI_COMM_WORLD);
            code_get[i][code_len[i]] = '\0';
        }
    }

    // Get cell_counter
    MPI_Bcast(&cell_counter, 1, MPI_INT, g_myroot, MPI_COMM_WORLD);
#endif

    std::string cell_name = std::string("In [") + std::to_string(cell_counter) + std::string("]");

    // Clear the template buffer and redirect stdout, stderr
    PyRun_SimpleString("import sys, io\n");
    PyRun_SimpleString("sys.OUTPUT_STDOUT=''\nstdout_buf=io.StringIO()\nsys.stdout=stdout_buf\n");
    PyRun_SimpleString("sys.OUTPUT_STDERR=''\nstderr_buf=io.StringIO()\nsys.stderr=stderr_buf\n");

    // Execute upper half and lower half one after another, if error occurred, it will skip execute the lower half
    // Need to consider both parallel and serial. Root runs code_split, non-root runs code_get.
    PyObject *py_src, *py_dump;
    bool has_error = false;
    for (int i = 0; i < 2; i++) {
        if (g_myrank == g_myroot) {
            if (code_split[i].length() <= 0) continue;
            if (i == 0) {
                py_src = Py_CompileString(code_split[i].c_str(), cell_name.c_str(), Py_file_input);
            } else if (i == 1) {
                py_src = Py_CompileString(code_split[i].c_str(), cell_name.c_str(), Py_single_input);
            }
        }
#ifndef SERIAL_MODE
        else {
            if (code_len[i] <= 0) continue;
            if (i == 0) {
                py_src = Py_CompileString(code_get[i], cell_name.c_str(), Py_file_input);
            } else if (i == 1) {
                py_src = Py_CompileString(code_get[i], cell_name.c_str(), Py_single_input);
            }
        }
#endif

        // Every MPI process should have the same compile result
        // Evaluate code
        if (py_src != NULL) {
            py_dump = PyEval_EvalCode(py_src, LibytPythonShell::get_script_namespace(),
                                      LibytPythonShell::get_script_namespace());
            if (PyErr_Occurred()) {
                has_error = true;
                PyErr_Print();
                Py_DECREF(py_src);
                Py_XDECREF(py_dump);
            } else {
                Py_DECREF(py_src);
                Py_XDECREF(py_dump);
            }
        } else {
            has_error = true;
            PyErr_Print();
        }

#ifndef SERIAL_MODE
        int success = has_error ? 0 : 1;
        int all_success;
        MPI_Allreduce(&success, &all_success, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        if (all_success != g_mysize) {
            has_error = true;
        }
#endif
        if (has_error) {
            break;
        }
    }

#ifndef SERIAL_MODE
    if (g_myrank != g_myroot) {
        delete[] code_get[0];
        delete[] code_get[1];
    }
#endif

    // Collect stdout_buf/stderr_buf and store under sys.OUTPUT_STDOUT and sys.OUTPUT_STDERR
    PyRun_SimpleString("sys.stdout.flush()\n");
    PyRun_SimpleString("sys.stderr.flush()\n");
    PyRun_SimpleString("sys.OUTPUT_STDOUT=stdout_buf.getvalue()\nstdout_buf.close()\n");
    PyRun_SimpleString("sys.OUTPUT_STDERR=stderr_buf.getvalue()\nstderr_buf.close()\n");
    PyRun_SimpleString("sys.stdout=sys.__stdout__\n");
    PyRun_SimpleString("sys.stderr=sys.__stderr__\n");
    PyErr_Clear();

    // Parse the string
    PyObject* py_module_sys = PyImport_ImportModule("sys");
    PyObject* py_stdout_buf = PyObject_GetAttrString(py_module_sys, "OUTPUT_STDOUT");
    PyObject* py_stderr_buf = PyObject_GetAttrString(py_module_sys, "OUTPUT_STDERR");

    std::array<std::string, 2> output = {std::string(""), std::string("")};
    output[0] = std::string(PyUnicode_AsUTF8(py_stdout_buf));
    if (has_error) {
        output[1] = PyUnicode_AsUTF8(py_stderr_buf);
    }

#ifndef SERIAL_MODE
    // Collect output from each rank to root
    for (int i = 0; i < 2; i++) {
        if (g_myrank == g_myroot) {
            // Gather output length
            int output_len = (int)output[i].length();
            int* all_output_len = new int[g_mysize];
            MPI_Gather(&output_len, 1, MPI_INT, all_output_len, 1, MPI_INT, g_myroot, MPI_COMM_WORLD);

            // Gather output
            long sum_output_len = 0;
            int* displace = new int[g_mysize];
            for (int r = 0; r < g_mysize; r++) {
                displace[r] = 0;
                sum_output_len += all_output_len[r];
                for (int r1 = 0; r1 < r; r1++) {
                    displace[r] += all_output_len[r1];
                }
            }
            char* all_output = new char[sum_output_len + 1];
            MPI_Gatherv(output[i].c_str(), output_len, MPI_CHAR, all_output, all_output_len, displace, MPI_CHAR,
                        g_myroot, MPI_COMM_WORLD);
            all_output[sum_output_len] = '\0';
            output[i] = std::string(all_output);

            // Insert header to string, do it in reverse, so that the displacement won't lose count
            if (output[i].length() > 0) {
                for (int r = g_mysize - 1; r >= 0; r--) {
                    std::string head =
                        std::string("\033[1;34m[MPI Process ") + std::to_string(r) + std::string("]\n\033[0;30m");
                    if (all_output_len[r] == 0) {
                        head += std::string("(None)\n");
                    }
                    output[i].insert(displace[r], head);
                }
            }

            // Free
            delete[] all_output_len;
            delete[] all_output;
            delete[] displace;
        } else {
            int output_len = (int)output[i].length();
            MPI_Gather(&output_len, 1, MPI_INT, nullptr, 1, MPI_INT, g_myroot, MPI_COMM_WORLD);
            MPI_Gatherv(output[i].c_str(), output_len, MPI_CHAR, nullptr, nullptr, nullptr, MPI_CHAR, g_myroot,
                        MPI_COMM_WORLD);
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
#endif

    Py_DECREF(py_module_sys);
    Py_DECREF(py_stdout_buf);
    Py_DECREF(py_stderr_buf);

    return output;
}

//-------------------------------------------------------------------------------------------------------
// Function      :  check_colon_exist
//
// Notes         :  1. This function gets called when detects s_NotDone_PyErr[0] ("if 1==1:\n") error.
//                  2. If it is a true indentation error (which means a syntax err), then the code buffer
//                     will reset.
//                  3. An indentation error caused by user-not-done-yet will contain ':' in the last line.
//
// Arguments     :  const char *code : code
//
// Return        :  false : It is a true indentation error caused by user error.
//                  true  : Indentation error caused by user not done inputting yet.
//-------------------------------------------------------------------------------------------------------
static bool check_colon_exist(const char* code) {
    std::string code_str = std::string(code);
    std::size_t start_pos = 0, found;
    bool last_line_has_colon = false;

    while (code_str.length() > 0) {
        found = code_str.find('\n', start_pos);
        if (found != std::string::npos) {
            last_line_has_colon = false;

            // check if the line contains ':' only
            for (std::size_t i = start_pos; i < found; i++) {
                if (code_str.at(i) == ':') {
                    last_line_has_colon = true;
                    break;
                }
            }
        } else {
            break;
        }

        start_pos = found + 1;
    }

    return last_line_has_colon;
}

#endif  // #ifdef INTERACTIVE_MODE