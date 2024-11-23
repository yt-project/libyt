#if defined(INTERACTIVE_MODE) || defined(JUPYTER_KERNEL)

#include "libyt_python_shell.h"

#include <algorithm>
#include <cstring>
#include <iostream>
#include <string>

#include "LibytProcessControl.h"
#include "function_info.h"
#include "yt_combo.h"

static std::vector<std::string> generate_err_msg(const std::vector<std::string>& statements);
static bool last_line_has_backslash(const std::string& code);
static bool last_line_has_colon(const std::string& code);

std::vector<std::string> LibytPythonShell::s_Bracket_NotDoneErr;
std::vector<std::string> LibytPythonShell::s_CompoundKeyword_NotDoneErr;
PyObject* LibytPythonShell::s_PyGlobals;

int LibytPythonShell::mpi_size_;
int LibytPythonShell::mpi_root_;
int LibytPythonShell::mpi_rank_;

//-------------------------------------------------------------------------------------------------------
// Struct      :  AccumulatedOutputString
// Method      :  Constructor
//
// Notes       :  1. Used in execute_cell and execute_prompt, for passing concatenated string around in
//                   root rank.
//                2. Initialize string as "", and length as vector with size equal to mpi_size_ (number of
//                   MPI processes).
//                3. Elements in output_length represent string length produced in each MPI process.
//
// Arguments   :  (None)
//
// Return      :  (None)
//-------------------------------------------------------------------------------------------------------
AccumulatedOutputString::AccumulatedOutputString() {
    output_string = std::string("");
    output_length.reserve(LibytProcessControl::Get().mpi_size_);
    for (int i = 0; i < LibytProcessControl::Get().mpi_size_; i++) {
        output_length.emplace_back(0);
    }
}

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
//                  4. TODO: It needs script's scope, otherwise some functors aren't detectable.
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
//                  3. Every MPI process has a copy of function body.
//                  4. It's not this method's responsibility to free code.
//                  5. To silent the printing when PyEval_EvalCode evaluates the code, that sys.stdout
//                     point to somewhere else when evaluating.
//                  6. It accepts indent size different from 4.
//                  7. TODO: It needs script's scope, otherwise some functors aren't detectable.
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
                g_func_status_list.AddNewFunction(func_name, FunctionInfo::RunStatus::kWillIdle);

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
            mpi_rank_);

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
//                  2. The not-done-error message initialized are those that can span multi-line.
//                     s_Bracket_NotDoneErr:
//                       (1)  """ | '''
//                       (2)  [r|u|f|b|rf|rb](""" | ''')
//                       (3)  ( | [ | {
//                     s_CompoundKeyword_NotDoneErr:
//                       (1)  if / if-else / if-elif
//                       (2)  try / try-except / try-except-finally
//                       (3)  class
//                       (4)  for
//                       (5)  def
//                       (6)  while
//                       (7)  with
//                       (8)  match / match-case
//                     We will drop every string after first number occurs in the error message,
//                     since some of them will print the lineno and it is Python version dependent.
//                     TODO: Be careful that the lineno (if it has) is at the end of the string,
//                           we should put this inside the test.
//                  3. Error messages are version dependent.
//
// Arguments     :  None
//
// Return        :  YT_SUCCESS or YT_FAIL
//-------------------------------------------------------------------------------------------------------
int LibytPythonShell::init_not_done_err_msg() {
    SET_TIMER(__PRETTY_FUNCTION__);

    // statement that can have newline in it
    std::vector<std::string> bracket_statement = {
        std::string("\"\"\""),  std::string("'''"),     std::string("r\"\"\""),  std::string("u\"\"\""),
        std::string("f\"\"\""), std::string("b\"\"\""), std::string("rf\"\"\""), std::string("rb\"\"\""),
        std::string("r'''"),    std::string("u'''"),    std::string("f'''"),     std::string("b'''"),
        std::string("rf'''"),   std::string("rb'''"),   std::string("("),        std::string("["),
        std::string("{")};

    std::vector<std::string> compound_keyword = {std::string("if 1==1:"),
                                                 std::string("if 1==1:\n  pass\nelse:"),
                                                 std::string("if 1==1:\n  pass\nelif 2==2:"),
                                                 std::string("try:"),
                                                 std::string("try:\n  pass\nexcept:"),
                                                 std::string("try:\n  pass\nfinally:"),
                                                 std::string("class A:"),
                                                 std::string("for _ in range(1):"),
                                                 std::string("def func():"),
                                                 std::string("while(False):"),
                                                 std::string("with open('') as f:")};
#if PY_MAJOR_VERSION >= 3 && PY_MINOR_VERSION >= 10
    compound_keyword.emplace_back("match (100):");
    compound_keyword.emplace_back("match (100):\n  case 100:");
#endif

    // get python error type and its statement.
    s_Bracket_NotDoneErr = std::move(generate_err_msg(bracket_statement));
    s_CompoundKeyword_NotDoneErr = std::move(generate_err_msg(compound_keyword));

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
//                  2. Check if the error buffer is error msg that is caused by user input
//                     not done yet. The error msg is initialized in init_not_done_err_msg.
//                  3. If it is indeed caused by user not done its input, clear error buffer. If not,
//                     restore the buffer and let others deal with the error msg.
//                  4. The rule to check if it is a real error or not-yet-done is in this order:
//                     (1) If last line of the code is ended by '\', it is not-yet-done.
//                     (2) If match error msg of s_CompoundKeyword_NotDoneErr:
//                         (2)-1 and if colon exist at last line and err lineno at the last line,
//                               then it's a user not-yet-done.
//                     (3) If the error is caused by bracket not closing s_Bracket_NotDoneErr,
//                         then it is not-yet-done.
//
// Arguments     :  const std::string& code : code to check
//
// Return        :  true / false : true for not-yet-done, false for a real error.
//-------------------------------------------------------------------------------------------------------
bool LibytPythonShell::is_not_done_err_msg(const std::string& code) {
    SET_TIMER(__PRETTY_FUNCTION__);

    bool user_not_done = false;

    // (1) check if last line has '\'
    if (last_line_has_backslash(code)) {
        user_not_done = true;
        return user_not_done;
    }

    // parse error msg and lineno
    std::string err_msg_str;
    long err_lineno;

#if PY_MAJOR_VERSION >= 3 && PY_MINOR_VERSION >= 12
    PyObject *py_exc, *py_msg, *py_lineno;
    py_exc = PyErr_GetRaisedException();
    py_msg = PyObject_GetAttrString(py_exc, "msg");
    py_lineno = PyObject_GetAttrString(py_exc, "lineno");
    err_msg_str = std::string(PyUnicode_AsUTF8(py_msg));
    err_lineno = PyLong_AsLong(py_lineno);
    Py_DECREF(py_msg);
    Py_DECREF(py_lineno);
#else
    // parse error
    PyObject *py_exc, *py_val, *py_traceback, *py_obj;
    const char* err_msg = "";
    PyErr_Fetch(&py_exc, &py_val, &py_traceback);
    PyArg_ParseTuple(py_val, "sO", &err_msg, &py_obj);

    // get error msg and lineno
    err_msg_str = std::string(err_msg);
    err_lineno = PyLong_AsLong(PyTuple_GetItem(py_obj, 1));
#endif

    // (2) error msg matches && if ':' exist && lineno is at last line
    std::size_t line_count = std::count(code.begin(), code.end(), '\n');
    bool match_compoundkeyword_errmsg = false;
    for (int i = 0; i < s_CompoundKeyword_NotDoneErr.size(); i++) {
        if (err_msg_str.find(s_CompoundKeyword_NotDoneErr[i]) == 0) {
            match_compoundkeyword_errmsg = true;
            break;
        }
    }

    if (match_compoundkeyword_errmsg && last_line_has_colon(code) && err_lineno == line_count) {
        user_not_done = true;
    }

    // (3) check if it is caused by brackets
    if (!user_not_done) {
        for (int i = 0; i < s_Bracket_NotDoneErr.size(); i++) {
            if (err_msg_str.find(s_Bracket_NotDoneErr[i]) == 0) {
                user_not_done = true;
                break;
            }
        }
    }

    // deal with reference or restore error buffer
    if (user_not_done) {
#if PY_MAJOR_VERSION >= 3 && PY_MINOR_VERSION >= 12
        Py_XDECREF(py_exc);
#else
        Py_XDECREF(py_exc);
        Py_XDECREF(py_val);
        Py_XDECREF(py_traceback);
        Py_XDECREF(py_obj);
#endif
    } else {
#if PY_MAJOR_VERSION >= 3 && PY_MINOR_VERSION >= 12
        PyErr_SetRaisedException(py_exc);
#else
        PyErr_Restore(py_exc, py_val, py_traceback);
#endif
    }

    return user_not_done;
}

//-------------------------------------------------------------------------------------------------------
// Class         :  LibytPythonShell
// Static Method :  check_code_validity
//
// Notes       :  1. Test if it can compile based on Py_single_input (if in prompt env), otherwise compile
//                   base on Py_file_input.
//
// Arguments   :  const std::string&  code : code to check
//                bool          prompt_env : if it is in prompt environment
//                const char    *cell_name : cell name
//
// Return      :  CodeValidity.is_valid : "complete", "incomplete", "invalid", "unknown"
//                             error_msg: error message from Python if it failed.
//-------------------------------------------------------------------------------------------------------
CodeValidity LibytPythonShell::check_code_validity(const std::string& code, bool prompt_env, const char* cell_name) {
    SET_TIMER(__PRETTY_FUNCTION__);

    CodeValidity code_validity;

    // clear error buffer before redirecting stderr
    PyErr_Clear();
    PyRun_SimpleString("import sys, io\n");
    PyRun_SimpleString("sys.OUTPUT_STDERR=''\nstderr_buf=io.StringIO()\nsys.stderr=stderr_buf\n");

    PyObject* py_test_compile;
    if (prompt_env) {
        py_test_compile = Py_CompileString(code.c_str(), cell_name, Py_single_input);
    } else {
        py_test_compile = Py_CompileString(code.c_str(), cell_name, Py_file_input);
    }

    if (py_test_compile != NULL) {
        code_validity.is_valid = "complete";
    } else if (prompt_env && is_not_done_err_msg(code)) {
        code_validity.is_valid = "incomplete";
    } else {
        code_validity.is_valid = "invalid";

        PyErr_Print();
        PyRun_SimpleString("sys.stderr.flush()\n");
        PyRun_SimpleString("sys.OUTPUT_STDERR=stderr_buf.getvalue()\n");
        PyObject* py_module_sys = PyImport_ImportModule("sys");
        PyObject* py_stderr_buf = PyObject_GetAttrString(py_module_sys, "OUTPUT_STDERR");
        code_validity.error_msg = std::string(PyUnicode_AsUTF8(py_stderr_buf));

        Py_DECREF(py_module_sys);
        Py_DECREF(py_stderr_buf);
    }

    // Clear buffer and dereference
    PyRun_SimpleString("stderr_buf.close()\nsys.stderr=sys.__stderr__\n");
    Py_XDECREF(py_test_compile);

    return code_validity;
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
//                3. This method is the based method for python code execution. It updates function def.
//                4. Root rank will pass in code, cell name; Non-root ranks only need to wait.
//
// Arguments   :  const std::array<std::string, 2>& code_split : code with upper and lower half
//                const std::string&                cell_name  : cell name (default = "")
//
// Return      :  std::array<AccumulatedOutputString, 2> output[0] : stdout
//                                                       output[1] : stderr
//-------------------------------------------------------------------------------------------------------
std::array<AccumulatedOutputString, 2> LibytPythonShell::execute_cell(const std::array<std::string, 2>& code_split,
                                                                      const std::string& cell_name) {
    SET_TIMER(__PRETTY_FUNCTION__);

#ifndef SERIAL_MODE
    // Get code_split from root rank
    unsigned long code_len[2] = {code_split[0].length(), code_split[1].length()};
    MPI_Bcast(&code_len[0], 1, MPI_UNSIGNED_LONG, mpi_root_, MPI_COMM_WORLD);
    MPI_Bcast(&code_len[1], 1, MPI_UNSIGNED_LONG, mpi_root_, MPI_COMM_WORLD);

    char* code_get[2];
    if (mpi_rank_ == mpi_root_) {
        for (int i = 0; i < 2; i++) {
            MPI_Bcast((void*)code_split[i].c_str(), (int)code_len[i], MPI_CHAR, mpi_root_, MPI_COMM_WORLD);
        }
    } else {
        for (int i = 0; i < 2; i++) {
            code_get[i] = new char[code_len[i] + 1];
            MPI_Bcast((void*)code_get[i], (int)code_len[i], MPI_CHAR, mpi_root_, MPI_COMM_WORLD);
            code_get[i][code_len[i]] = '\0';
        }
    }

    // Get cell_name from root rank
    unsigned long cell_name_len = cell_name.length();
    MPI_Bcast(&cell_name_len, 1, MPI_UNSIGNED_LONG, mpi_root_, MPI_COMM_WORLD);

    char* cell_name_get;
    if (mpi_rank_ == mpi_root_) {
        MPI_Bcast((void*)cell_name.c_str(), (int)cell_name_len, MPI_CHAR, mpi_root_, MPI_COMM_WORLD);
    } else {
        cell_name_get = new char[cell_name_len + 1];
        MPI_Bcast((void*)cell_name_get, (int)cell_name_len, MPI_CHAR, mpi_root_, MPI_COMM_WORLD);
        cell_name_get[cell_name_len] = '\0';
    }
#endif

    // Clear the template buffer and redirect stdout, stderr
    PyErr_Clear();
    PyRun_SimpleString("import sys, io\n");
    PyRun_SimpleString("sys.OUTPUT_STDOUT=''\nstdout_buf=io.StringIO()\nsys.stdout=stdout_buf\n");
    PyRun_SimpleString("sys.OUTPUT_STDERR=''\nstderr_buf=io.StringIO()\nsys.stderr=stderr_buf\n");

    // Execute upper half and lower half one after another, if error occurred, it will skip execute the lower half
    PyObject *py_src, *py_dump;
    bool has_error = false;
    for (int i = 0; i < 2; i++) {
        const char* code_sync;
        const char* cell_name_sync;
        if (mpi_rank_ == mpi_root_) {
            code_sync = code_split[i].c_str();
            cell_name_sync = cell_name.c_str();
        }
#ifndef SERIAL_MODE
        else {
            code_sync = code_get[i];
            cell_name_sync = cell_name_get;
        }
#endif
        if (strlen(code_sync) <= 0) continue;
        if (i == 0) {
            py_src = Py_CompileString(code_sync, cell_name_sync, Py_file_input);
        } else {
            py_src = Py_CompileString(code_sync, cell_name_sync, Py_single_input);
        }

        // Every MPI process should have the same compile result
        // Evaluate code
        if (py_src != NULL) {
            py_dump = PyEval_EvalCode(py_src, get_script_namespace(), get_script_namespace());
            if (PyErr_Occurred()) {
                has_error = true;
                PyErr_Print();
                load_input_func_body(code_sync);

                Py_DECREF(py_src);
                Py_XDECREF(py_dump);
            } else {
                load_input_func_body(code_sync);
                update_prompt_history(std::string(code_sync));

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
        if (all_success != mpi_size_) {
            has_error = true;
        }
#endif
        if (has_error) {
            break;
        }
    }

#ifndef SERIAL_MODE
    if (mpi_rank_ != mpi_root_) {
        delete[] code_get[0];
        delete[] code_get[1];
        delete[] cell_name_get;
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

    std::array<AccumulatedOutputString, 2> output;
    output[0].output_string = std::string(PyUnicode_AsUTF8(py_stdout_buf));
    output[0].output_length[mpi_rank_] = output[0].output_string.length();
    if (has_error) {
        output[1].output_string = std::string(PyUnicode_AsUTF8(py_stderr_buf));
        output[1].output_length[mpi_rank_] = output[1].output_string.length();
    }

#ifndef SERIAL_MODE
    // Collect output from each rank to root
    for (int i = 0; i < 2; i++) {
        if (mpi_rank_ == mpi_root_) {
            // Gather output length
            int output_len = (int)output[i].output_length[mpi_rank_];
            MPI_Gather(&output_len, 1, MPI_INT, output[i].output_length.data(), 1, MPI_INT, mpi_root_, MPI_COMM_WORLD);

            // Gather output
            long sum_output_len = 0;
            int* displace = new int[mpi_size_];
            for (int r = 0; r < mpi_size_; r++) {
                displace[r] = 0;
                sum_output_len += output[i].output_length[r];
                for (int r1 = 0; r1 < r; r1++) {
                    displace[r] += output[i].output_length[r1];
                }
            }
            char* all_output = new char[sum_output_len + 1];
            MPI_Gatherv(output[i].output_string.c_str(), output_len, MPI_CHAR, all_output,
                        output[i].output_length.data(), displace, MPI_CHAR, mpi_root_, MPI_COMM_WORLD);
            all_output[sum_output_len] = '\0';
            output[i].output_string = std::string(all_output);

            // Free
            delete[] all_output;
            delete[] displace;
        } else {
            int output_len = (int)output[i].output_length[mpi_rank_];
            MPI_Gather(&output_len, 1, MPI_INT, nullptr, 1, MPI_INT, mpi_root_, MPI_COMM_WORLD);
            MPI_Gatherv(output[i].output_string.c_str(), output_len, MPI_CHAR, nullptr, nullptr, nullptr, MPI_CHAR,
                        mpi_root_, MPI_COMM_WORLD);
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
// Class         :  LibytPythonShell
// Static Method :  execute_prompt
// Description   :  Execute single statement code get from prompt.
//
// Notes       :  1. This is a collective operation, requires every rank to call this function.
//                   Assuming every MPI process enter this function at the same state the same time.
//                2. Root rank will gather stdout and stderr from non-root rank, so the string returned
//                   contains each ranks dumped output in root, and non-root rank only returns output from
//                   itself.
//                3.
//
// Arguments   :  const std::string&         code : single statement code (default = "")
//                const std::string&    cell_name : cell name             (default = "<libyt-stdin>")
//
// Return      :  std::array<AccumulatedOutputString, 2> output[0] : stdout
//                                                       output[1] : stderr
//-------------------------------------------------------------------------------------------------------
std::array<AccumulatedOutputString, 2> LibytPythonShell::execute_prompt(const std::string& code,
                                                                        const std::string& cell_name) {
    SET_TIMER(__PRETTY_FUNCTION__);

    std::array<AccumulatedOutputString, 2> output = execute_cell({"", code}, cell_name);

    return output;
}

//-------------------------------------------------------------------------------------------------------
// Class         :  LibytPythonShell
// Static Method :  execute_file
// Description   :  Execute a file
//
// Notes       :  1. This is a collective operation, requires every rank to call this function.
//                   Assuming every MPI process enter this function at the same state the same time.
//                2. Root rank will gather stdout and stderr from non-root rank, so the string returned
//                   contains each ranks dumped output in root, and non-root rank only returns output from
//                   itself.
//                3. Root rank will broadcast codes and related info for non-root rank.
//
// Arguments   :  const std::string&         code : full code in a file   (default = "")
//                const std::string&    file_name : file name             (default = "")
//
// Return      :  std::array<AccumulatedOutputString, 2> output[0] : stdout
//                                                       output[1] : stderr
//-------------------------------------------------------------------------------------------------------
std::array<AccumulatedOutputString, 2> LibytPythonShell::execute_file(const std::string& code,
                                                                      const std::string& file_name) {
    SET_TIMER(__PRETTY_FUNCTION__);

    std::array<AccumulatedOutputString, 2> output = execute_cell({code, ""}, file_name);

    return output;
}

//-------------------------------------------------------------------------------------------------------
// Function      :  generate_err_msg
//
// Notes         :  1. Generate error msg that are caused by user not-done-yet.
//                  2. The error msg drops everything after the numbers.
//
// Arguments     :  const std::vector<std::string>& statements : a list of fail statements
//
// Return        :  std::vector<std::string> : a list of error msg, neglecting everything after numbers
//                                             if there is.
//-------------------------------------------------------------------------------------------------------
static std::vector<std::string> generate_err_msg(const std::vector<std::string>& statements) {
    SET_TIMER(__PRETTY_FUNCTION__);

    std::vector<std::string> generated_error_msg;

    for (int i = 0; i < statements.size(); i++) {
        PyObject *py_src, *py_exc, *py_val;
        std::string err_msg_str;

        py_src = Py_CompileString(statements[i].c_str(), "<get err msg>", Py_single_input);

#if PY_MAJOR_VERSION >= 3 && PY_MINOR_VERSION >= 12
        py_exc = PyErr_GetRaisedException();
        py_val = PyObject_GetAttrString(py_exc, "msg");
        err_msg_str = std::string(PyUnicode_AsUTF8(py_val));
#else
        PyObject *py_traceback, *py_obj;
        const char* err_msg;
        PyErr_Fetch(&py_exc, &py_val, &py_traceback);
        PyArg_ParseTuple(py_val, "sO", &err_msg, &py_obj);
        err_msg_str = std::string(err_msg);
#endif

        std::size_t found = err_msg_str.find_first_of("1234567890");
        if (found != std::string::npos) {
            generated_error_msg.emplace_back(err_msg_str.substr(0, found));
        } else {
            generated_error_msg.emplace_back(err_msg_str);
        }
        generated_error_msg[i].shrink_to_fit();

        // dereference
        Py_XDECREF(py_src);
        Py_XDECREF(py_val);
        Py_XDECREF(py_exc);
#if PY_MAJOR_VERSION >= 3 && PY_MINOR_VERSION >= 12
#else
        Py_XDECREF(py_traceback);
        Py_XDECREF(py_obj);
#endif

        PyErr_Clear();
    }

    generated_error_msg.shrink_to_fit();

    return generated_error_msg;
}

//-------------------------------------------------------------------------------------------------------
// Function      :  last_line_has_backslash
//
// Notes         :  1. Find if last line ends with '\' and does not have comments '#'
//                  2. Code passed in always ends with '\n'.
//
// Arguments     :  const char *code : code
//
// Return        :  false : last line has no '\' at the very end, neglecting the comments
//                  true  : last line has '\' at the very end, neglecting the comments
//-------------------------------------------------------------------------------------------------------
static bool last_line_has_backslash(const std::string& code) {
    SET_TIMER(__PRETTY_FUNCTION__);

    std::size_t code_len = code.length();

    // find last line, and since code always ends with '\n', ignore it
    std::size_t last_line_start_no = code.rfind('\n', code_len - 2);

    // check if last line have '#'
    std::size_t found_pound;
    if (last_line_start_no != std::string::npos) {
        found_pound = code.find('#', last_line_start_no);
    } else {
        found_pound = code.find('#');
    }

    return code.at(code_len - 2) == '\\' && found_pound == std::string::npos;
}

//-------------------------------------------------------------------------------------------------------
// Function      :  last_line_has_colon
//
// Notes         :  1. This function is used for distinguishing keywords for multi-line and the true error.
//                  2. An indentation error caused by user-not-done-yet will end with ':' in the last line.
//
// Arguments     :  const std::string& code : full code to check
//
// Return        :  true  : contains valid ':' in the last line at the end
//                  false : doesn't contain ':' in the last line at the end
//-------------------------------------------------------------------------------------------------------
static bool last_line_has_colon(const std::string& code) {
    SET_TIMER(__PRETTY_FUNCTION__);

    std::size_t code_len = code.length();

    // find last line, and since code always ends with '\n', ignore it
    std::size_t last_line_start_no = code.rfind('\n', code_len - 2);
    if (last_line_start_no == std::string::npos) {
        last_line_start_no = 0;
    }

    // find first '#' in last line
    std::size_t last_line_found_pound = code.find('#', last_line_start_no);

    // find last character that is not spaces or tabs
    std::size_t found_char = code.find_last_not_of("\t\n\v\f\r ", last_line_found_pound);

    // make sure the char is at the last line, then compare and make sure it is ':'
    if (found_char == std::string::npos) {
        return false;
    } else {
        if (last_line_start_no < found_char) {
            return code.at(found_char) == ':';
        } else {
            return false;
        }
    }
}

#endif  // #if defined( INTERACTIVE_MODE) || defined(JUPYTER_KERNEL)
