#if defined(INTERACTIVE_MODE) || defined(JUPYTER_KERNEL)

#include "libyt_python_shell.h"

#include <algorithm>
#include <cstring>
#include <string>

#include "function_info.h"
#include "libyt_process_control.h"
#include "yt_combo.h"

static std::vector<std::string> GenerateErrMsg(const std::vector<std::string>& statements);
static bool LastLineHasBackslash(const std::string& code);
static bool LastLineHasColon(const std::string& code);
static void SplitOnLine(const std::string& code, unsigned int lineno, std::array<std::string, 2>& code_split);

std::vector<std::string> LibytPythonShell::bracket_not_done_err_;
std::vector<std::string> LibytPythonShell::compound_keyword_not_done_err_;
PyObject* LibytPythonShell::execution_namespace_;
PyObject* LibytPythonShell::function_body_dict_;

int LibytPythonShell::mpi_size_;
int LibytPythonShell::mpi_root_;
int LibytPythonShell::mpi_rank_;

//-------------------------------------------------------------------------------------------------------
// Class       :  LibytPythonShell
// Method      :  UpdateHistory / ClearHistory
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
int LibytPythonShell::UpdateHistory(const std::string& cmd_prompt) {
    SET_TIMER(__PRETTY_FUNCTION__);

    history_ = history_ + std::string("#In[") + std::to_string(history_count_) + std::string("]\n");
    history_ = history_ + cmd_prompt + std::string("\n\n");
    history_count_ += 1;
    return YT_SUCCESS;
}

int LibytPythonShell::ClearHistory() {
    SET_TIMER(__PRETTY_FUNCTION__);

    history_ = std::string("");
    history_count_ = 0;
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
//                  5. TODO: Re-write this using C API instead of PyRun_SimpleString, and make it store
//                           the function body under some PyObject* dictionary in the class data member.
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
//                  7. TODO: (Unit Test) It probably needs a mock for LibytProcessControl::Get().function_info_list_
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
                // add new function to LibytProcessControl::Get().function_info_list_ and set to idle. if function
                // exists already, get its index
                const char* func_name = PyUnicode_AsUTF8(PyList_GET_ITEM(py_new_dict_keys, i));
                LibytProcessControl::Get().function_info_list_.AddNewFunction(func_name,
                                                                              FunctionInfo::RunStatus::kWillIdle);

                // update function body
                PyObject* py_func_body_dict = GetFunctionBodyDict();
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
//                  3. TODO: Re-write this using C API instead of PyRun_SimpleString, and make it store
//                           the function body under some PyObject* dictionary in the class data member.
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

    PyObject* py_func_list = PyDict_GetItemString(LibytProcessControl::Get().py_interactive_mode_, "temp");
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
// Static Method :  SetExceptionHook()
//
// Notes         :  1. This is a static method.
//                  2. Makes MPI not invoke MPI_Abort when getting errors in Python in interactive mode.
//                  3. Can only be called after libyt is initialized.
//
// Arguments     :  None
//
// Return        :  YT_SUCCESS or YT_FAIL
//-------------------------------------------------------------------------------------------------------
int LibytPythonShell::SetExceptionHook() {
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
// Static Method :  InitializeNotDoneErrMsg
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
//                     since some of them will print the lineno, and it is Python version dependent.
//                     Ex: "SyntaxError: unterminated triple-quoted string literal (detected at line 1)"
//                     This is put in unit test to make sure that the number appears within the last 3 chars
//                     of the error msg.
//                  3. Error messages are version dependent.
//
// Arguments     :  None
//
// Return        :  YT_SUCCESS or YT_FAIL
//-------------------------------------------------------------------------------------------------------
int LibytPythonShell::InitializeNotDoneErrMsg() {
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
    bracket_not_done_err_ = std::move(GenerateErrMsg(bracket_statement));
    compound_keyword_not_done_err_ = std::move(GenerateErrMsg(compound_keyword));

    return YT_SUCCESS;
}

//-------------------------------------------------------------------------------------------------------
// Class         :  LibytPythonShell
// Static Method :  SetExecutionNamespace
//
// Notes         :  1. This is a static method.
//                  2. Set the execution namespace for Python interpreter. The namespace is used to execute
//                     Python code.
//                  3. Only sets the pointer, and doesn't alter the reference count.
//
// Arguments     :  None
//
// Return        :  YT_SUCCESS or YT_FAIL
//-------------------------------------------------------------------------------------------------------
int LibytPythonShell::SetExecutionNamespace(PyObject* execution_namespace) {
    SET_TIMER(__PRETTY_FUNCTION__);

    execution_namespace_ = execution_namespace;

    return YT_SUCCESS;
}

//-------------------------------------------------------------------------------------------------------
// Class         :  LibytPythonShell
// Static Method :  SetExecutionNamespace
//
// Notes         :  1. This is a static method.
//                  2. Set the dictionary storage pointer where function body is stored.
//                  3. Only sets the pointer, and doesn't alter the reference count.
//
// Arguments     :  None
//
// Return        :  YT_SUCCESS or YT_FAIL
//-------------------------------------------------------------------------------------------------------
int LibytPythonShell::SetFunctionBodyDict(PyObject* function_body_dict) {
    SET_TIMER(__PRETTY_FUNCTION__);

    function_body_dict_ = function_body_dict;

    return YT_SUCCESS;
}

//-------------------------------------------------------------------------------------------------------
// Class         :  LibytPythonShell
// Static Method :  IsNotDoneErrMsg
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
bool LibytPythonShell::IsNotDoneErrMsg(const std::string& code) {
    SET_TIMER(__PRETTY_FUNCTION__);

    bool user_not_done = false;

    // (1) check if last line has '\'
    if (LastLineHasBackslash(code)) {
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
    for (const std::string& not_done_err : compound_keyword_not_done_err_) {
        if (err_msg_str.find(not_done_err) == 0) {
            match_compoundkeyword_errmsg = true;
            break;
        }
    }

    if (match_compoundkeyword_errmsg && LastLineHasColon(code) && err_lineno == line_count) {
        user_not_done = true;
    }

    // (3) check if it is caused by brackets
    if (!user_not_done) {
        for (const std::string& not_done_err : bracket_not_done_err_) {
            if (err_msg_str.find(not_done_err) == 0) {
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
// Static Method :  CheckCodeValidity
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
CodeValidity LibytPythonShell::CheckCodeValidity(const std::string& code, bool prompt_env, const char* cell_name) {
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
    } else if (prompt_env && IsNotDoneErrMsg(code)) {
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
// Public Method :  AllExecute
// Description   :  Execute code on every MPI process based on Python input type. (collective operation)
//
// Notes       :  1. This is a collective operation, requires every rank to call this function.
//                2. To avoid copying data string back and forth, client will need to pass in the designated
//                   output storage.
//                3. The local output are stored in the mpi rank order, which is output[mpi_rank_], and the
//                   returned status represents all ranks successfully done or failed the job.
//                   Only the output_mpi_rank has full knowledge (status, output, error) of each rank,
//                   the others only knows itself and the overall status.
//                   output_mpi_rank should be the same in every rank.
//                4. To avoid unnecessary copying of code and cell name, they are stored in different variables,
//                   ending with _sync, even though this will make the code less readable.
//                5. Python input type should be: Py_single_input (256), Py_file_input (257), Py_eval_input (258)
//                   define in Python header. (TODO: should check in unit test)
//                6. This function doesn't check code validity. It's better to call CheckCodeValidity before calling.
//                7. TODO: probably need to find another to redirect and capture the stdout/stderr, or
//                         create a new class. also, the current method is probably not thread-safe.
//-------------------------------------------------------------------------------------------------------
PythonStatus LibytPythonShell::AllExecute(int python_input_type, const std::string& code,
                                          const std::string& cell_base_name, int src_rank,
                                          std::vector<PythonOutput>& output, int output_mpi_rank) {
    SET_TIMER(__PRETTY_FUNCTION__);

#ifndef SERIAL_MODE
    // Sync the code and cell name
    std::string code_sync, cell_base_name_sync;
    if (mpi_rank_ == src_rank) {
        CommMpi::SetStringUsingValueOnRank(const_cast<std::string&>(code), src_rank);
        CommMpi::SetStringUsingValueOnRank(const_cast<std::string&>(cell_base_name), src_rank);
    } else {
        CommMpi::SetStringUsingValueOnRank(code_sync, src_rank);
        CommMpi::SetStringUsingValueOnRank(cell_base_name_sync, src_rank);
    }
#endif

    // Refer to code and cell_name using pointers
    const char* code_ptr = nullptr;
    const char* cell_base_name_ptr = nullptr;
    if (mpi_rank_ == src_rank) {
        code_ptr = code.c_str();
        cell_base_name_ptr = cell_base_name.c_str();
    }
#ifndef SERIAL_MODE
    else {
        code_ptr = code_sync.c_str();
        cell_base_name_ptr = cell_base_name_sync.c_str();
    }
#endif

    if (strlen(code_ptr) <= 0) {
        output.clear();
        if (mpi_rank_ == output_mpi_rank) {
            output.assign(mpi_size_, PythonOutput{.status = PythonStatus::kPythonSuccess,
                                                  .output = std::string(""),
                                                  .error = std::string("")});
        } else {
            output.assign(mpi_size_, PythonOutput{.status = PythonStatus::kPythonUnknown,
                                                  .output = std::string(""),
                                                  .error = std::string("")});
            output[mpi_rank_].status = PythonStatus::kPythonSuccess;
        }
        return PythonStatus::kPythonSuccess;
    }

    // Clear the template buffer and redirect stdout, stderr
    // TODO: store it in other variable for the stdout/stderr buffer (create a new class??)
    PyErr_Clear();
    PyRun_SimpleString("import sys, io\n");
    PyRun_SimpleString("sys.OUTPUT_STDOUT=''\nstdout_buf=io.StringIO()\nsys.stdout=stdout_buf\n");
    PyRun_SimpleString("sys.OUTPUT_STDERR=''\nstderr_buf=io.StringIO()\nsys.stderr=stderr_buf\n");

    PyObject* py_src = Py_CompileString(code_ptr, cell_base_name_ptr, python_input_type);
    bool has_error = false;
    if (py_src != NULL) {
        PyObject* py_dump = PyEval_EvalCode(py_src, GetExecutionNamespace(), GetExecutionNamespace());
        if (PyErr_Occurred()) {
            has_error = true;
            PyErr_Print();
            load_input_func_body(code_ptr);
        } else {
            load_input_func_body(code_ptr);
            UpdateHistory(std::string(code_ptr));
        }
        Py_DECREF(py_src);
        Py_XDECREF(py_dump);
    } else {
        has_error = true;
        PyErr_Print();
    }

    // Close the redirect buffer and set it back to standard
    PyRun_SimpleString("sys.stdout.flush()\n");
    PyRun_SimpleString("sys.stderr.flush()\n");
    PyRun_SimpleString("sys.OUTPUT_STDOUT=stdout_buf.getvalue()\nstdout_buf.close()\n");
    PyRun_SimpleString("sys.OUTPUT_STDERR=stderr_buf.getvalue()\nstderr_buf.close()\n");
    PyRun_SimpleString("sys.stdout=sys.__stdout__\n");
    PyRun_SimpleString("sys.stderr=sys.__stderr__\n");
    PyErr_Clear();

    // Sync the results
#ifndef SERIAL_MODE
    bool all_has_error = CommMpi::CheckAllStates(has_error, false, false, true);
#else
    bool all_has_error = has_error;
#endif

    // Collect the local stdout/stderr and store in output vector, the elements are stored in the same order as rank
    output.clear();
    output.assign(
        mpi_size_,
        PythonOutput{.status = PythonStatus::kPythonUnknown, .output = std::string(""), .error = std::string("")});

    PyObject* py_module_sys = PyImport_ImportModule("sys");
    PyObject* py_stdout_buf = PyObject_GetAttrString(py_module_sys, "OUTPUT_STDOUT");
    PyObject* py_stderr_buf = PyObject_GetAttrString(py_module_sys, "OUTPUT_STDERR");

    output[mpi_rank_].output = std::string(PyUnicode_AsUTF8(py_stdout_buf));
    output[mpi_rank_].error = std::string(PyUnicode_AsUTF8(py_stderr_buf));

    if (has_error) {
        output[mpi_rank_].status = PythonStatus::kPythonFailed;
    } else {
        output[mpi_rank_].status = PythonStatus::kPythonSuccess;
    }

    Py_DECREF(py_module_sys);
    Py_DECREF(py_stdout_buf);
    Py_DECREF(py_stderr_buf);

#ifndef SERIAL_MODE
    // Gather all results to root rank
    std::vector<int> all_status;
    CommMpi::GatherAllIntsToRank(all_status, static_cast<int>(output[mpi_rank_].status), output_mpi_rank);
    if (mpi_rank_ == output_mpi_rank) {
        for (int i = 0; i < mpi_size_; i++) {
            output[i].status = static_cast<PythonStatus>(all_status[i]);
        }
    }

    // Gather the output from all ranks to root, move the gathered string to output
    std::vector<std::string> all_gather_string;
    CommMpi::GatherAllStringsToRank(all_gather_string, output[mpi_rank_].output, output_mpi_rank);
    if (mpi_rank_ == output_mpi_rank) {
        for (int i = 0; i < mpi_size_; i++) {
            output[i].output = std::move(all_gather_string[i]);
        }
    }
    CommMpi::GatherAllStringsToRank(all_gather_string, output[mpi_rank_].error, output_mpi_rank);
    if (mpi_rank_ == output_mpi_rank) {
        for (int i = 0; i < mpi_size_; i++) {
            output[i].error = std::move(all_gather_string[i]);
        }
    }
#endif

    return all_has_error ? PythonStatus::kPythonFailed : PythonStatus::kPythonSuccess;
}

//-------------------------------------------------------------------------------------------------------
// Class         :  LibytPythonShell
// Public Method :  AllExecutePrompt
// Description   :  Execute a single statement code on every MPI process. (collective operation)
//
// Notes       :  1. Call AllExecute with python input type as Py_single_input (int).
//-------------------------------------------------------------------------------------------------------
PythonStatus LibytPythonShell::AllExecutePrompt(const std::string& code, const std::string& cell_base_name,
                                                int src_rank, std::vector<PythonOutput>& output, int output_mpi_rank) {
    SET_TIMER(__PRETTY_FUNCTION__);

    PythonStatus status = AllExecute(Py_single_input, code, cell_base_name, src_rank, output, output_mpi_rank);
    return status;
}

//-------------------------------------------------------------------------------------------------------
// Class         :  LibytPythonShell
// Public Method :  AllExecuteFile
// Description   :  Execute an arbitrary length code on every MPI process. (collective operation)
//
// Notes       :  1. Call AllExecute with python input type as Py_file_input (int).
//-------------------------------------------------------------------------------------------------------
PythonStatus LibytPythonShell::AllExecuteFile(const std::string& code, const std::string& cell_base_name, int src_rank,
                                              std::vector<PythonOutput>& output, int output_mpi_rank) {
    SET_TIMER(__PRETTY_FUNCTION__);

    PythonStatus status = AllExecute(Py_file_input, code, cell_base_name, src_rank, output, output_mpi_rank);
    return status;
}

//-------------------------------------------------------------------------------------------------------
// Class         :  LibytPythonShell
// Public Method :  AllExecuteCell
// Description   :  Execute an arbitrary length code and display the output like a JupyterLab cell
//                  (collective operation)
//
// Notes       :  1. For the code execution to be like a JupyterLab cell, the last line of code should
//                   behave like AllExecutePrompt and the rest of the code should behave like AllExecuteFile.
//                2. Since only the src_rank will input the code, we do the parsing on the src_rank only.
//                3  If error occurred in AllExecuteFile, it will skip AllExecutePrompt and return the error.
//                   Otherwise, it will combine the output and error from AllExecuteFile and AllExecutePrompt.
//                4. AllExecute doesn't check validity, it just executes it. But this method should be able to
//                   resolve an invalid code.
//-------------------------------------------------------------------------------------------------------
PythonStatus LibytPythonShell::AllExecuteCell(const std::string& code, const std::string& cell_base_name, int src_rank,
                                              std::vector<PythonOutput>& output, int output_mpi_rank) {
    SET_TIMER(__PRETTY_FUNCTION__);

    // Parse the code using ast and separate the last statement on src_rank only
    std::array<std::string, 2> code_split = {std::string(""), std::string("")};
    long last_statement_lineno = -1;
    if (mpi_rank_ == src_rank) {
        last_statement_lineno = GetLastStatementLineno(code);

        // Early return if the code is invalid or its empty
#ifndef SERIAL_MODE
        MPI_Bcast(&last_statement_lineno, 1, MPI_LONG, src_rank, MPI_COMM_WORLD);
#endif
        if (last_statement_lineno > 0) {
            SplitOnLine(code, last_statement_lineno - 1, code_split);

            // Add newline at the end of the last statement, so that Python won't produce EOF error
            // Add newline at the front of the last statement, so that Python error buffer can catch the correct lineno.
            code_split[1].insert(0, std::string(last_statement_lineno - 1, '\n'));
            if (!code_split[1].empty()) {
                code_split[1].append("\n");
            }
        }
    }
#ifndef SERIAL_MODE
    else {
        MPI_Bcast(&last_statement_lineno, 1, MPI_LONG, src_rank, MPI_COMM_WORLD);
    }
#endif

    // Early return if the code is invalid or its empty
    output.clear();
    if (last_statement_lineno == 0) {
        if (mpi_rank_ == output_mpi_rank) {
            output.assign(mpi_size_, PythonOutput{.status = PythonStatus::kPythonSuccess,
                                                  .output = std::string(""),
                                                  .error = std::string("")});
        } else {
            output.assign(mpi_size_, PythonOutput{.status = PythonStatus::kPythonUnknown,
                                                  .output = std::string(""),
                                                  .error = std::string("")});
            output[mpi_rank_].status = PythonStatus::kPythonSuccess;
        }
        return PythonStatus::kPythonSuccess;
    } else if (last_statement_lineno < 0) {
        if (mpi_rank_ == output_mpi_rank) {
            output.assign(mpi_size_, PythonOutput{.status = PythonStatus::kPythonFailed,
                                                  .output = std::string(""),
                                                  .error = std::string("")});
        } else {
            output.assign(mpi_size_, PythonOutput{.status = PythonStatus::kPythonUnknown,
                                                  .output = std::string(""),
                                                  .error = std::string("")});
            output[mpi_rank_].status = PythonStatus::kPythonFailed;
        }
        return PythonStatus::kPythonFailed;
    }

    // Call AllExecuteFile and AllExecutePrompt and combine the output
    PythonStatus status = AllExecute(Py_file_input, code_split[0], cell_base_name, src_rank, output, output_mpi_rank);
    if (status == PythonStatus::kPythonFailed) {
        return status;
    }

    std::vector<PythonOutput> output_prompt;
    status = AllExecute(Py_single_input, code_split[1], cell_base_name, src_rank, output_prompt, output_mpi_rank);
    for (int r = 0; r < mpi_size_; r++) {
        output[r].status = output_prompt[r].status;
        output[r].output += output_prompt[r].output;
        output[r].error += output_prompt[r].error;
    }
    return status;
}

//-------------------------------------------------------------------------------------------------------
// Class         :  LibytPythonShell
// Private Method:  GetLastStatementLineno
// Description   :  Get the last statement lineno in the code
//
// Notes       :  1. Use Python ast to split the last statement and the rest of the code.
//                2. If the code is valid, it will return lineno > 0;
//                   If the code is invalid, it will return -1.
//                3. Python lineno count starts at 1. So If lineno is 0, it means the code is blank.
//                   And if lineno is -1, it means the code is invalid.
//-------------------------------------------------------------------------------------------------------
long LibytPythonShell::GetLastStatementLineno(const std::string& code) {
    SET_TIMER(__PRETTY_FUNCTION__);

    PyObject* py_module_ast = PyImport_ImportModule("ast");
    PyObject* py_ast_parse = PyObject_GetAttrString(py_module_ast, "parse");
    PyObject* py_result = PyObject_CallFunction(py_ast_parse, "s", code.c_str());

    Py_DECREF(py_module_ast);
    Py_DECREF(py_ast_parse);

    long last_statement_lineno;
    if (py_result == NULL) {
        last_statement_lineno = -1;  // code invalid
    } else {
        PyObject* py_result_body = PyObject_GetAttrString(py_result, "body");
        Py_ssize_t num_statements = PyList_Size(py_result_body);
        if (num_statements <= 0) {
            last_statement_lineno = 0;  // blank code
        } else {
            PyObject* py_lineno = PyObject_GetAttrString(PyList_GET_ITEM(py_result_body, num_statements - 1), "lineno");
            last_statement_lineno = PyLong_AsLong(py_lineno);
            Py_DECREF(py_lineno);
        }
        Py_DECREF(py_result_body);
        Py_DECREF(py_result);
    }

    return last_statement_lineno;
}

//-------------------------------------------------------------------------------------------------------
// Function      :  GenerateErrMsg
//
// Notes         :  1. Generate error msg that are caused by user not-done-yet.
//                  2. The error msg drops everything after the numbers.
//
// Arguments     :  const std::vector<std::string>& statements : a list of fail statements
//
// Return        :  std::vector<std::string> : a list of error msg, neglecting everything after numbers
//                                             if there is.
//-------------------------------------------------------------------------------------------------------
static std::vector<std::string> GenerateErrMsg(const std::vector<std::string>& statements) {
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
// Function      :  LastLineHasBackSlash
//
// Notes         :  1. Find if last line ends with '\' and does not have comments '#'
//                  2. Code passed in always ends with '\n'.
//
// Arguments     :  const char *code : code
//
// Return        :  false : last line has no '\' at the very end, neglecting the comments
//                  true  : last line has '\' at the very end, neglecting the comments
//-------------------------------------------------------------------------------------------------------
static bool LastLineHasBackslash(const std::string& code) {
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
// Function      :  LastLineHasColon
//
// Notes         :  1. This function is used for distinguishing keywords for multi-line and the true error.
//                  2. An indentation error caused by user-not-done-yet will end with ':' in the last line.
//
// Arguments     :  const std::string& code : full code to check
//
// Return        :  true  : contains valid ':' in the last line at the end
//                  false : doesn't contain ':' in the last line at the end
//-------------------------------------------------------------------------------------------------------
static bool LastLineHasColon(const std::string& code) {
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

//-------------------------------------------------------------------------------------------------------
// Method      :  SplitOnLine
// Description :  Split the string to two parts on lineno.
//
// Notes       :  1. It's a local method.
//                2. Line count starts at 1.
//                3. code_split[0] contains line 1 ~ lineno, code_split[1] contains the rest.
//-------------------------------------------------------------------------------------------------------
static void SplitOnLine(const std::string& code, unsigned int lineno, std::array<std::string, 2>& code_split) {
    SET_TIMER(__PRETTY_FUNCTION__);

    std::size_t start_pos = 0, found;
    unsigned int line = 1;
    while (!code.empty()) {
        found = code.find('\n', start_pos);
        if (found != std::string::npos) {
            if (line == lineno) {
                code_split[0] = std::move(code.substr(0, found));
                code_split[1] = std::move(code.substr(found + 1, code.length() - found));
                break;
            }
        } else {
            code_split[1] = std::move(code.substr(0, code.length()));
            break;
        }
        start_pos = found + 1;
        line += 1;
    }
}

#endif  // #if defined( INTERACTIVE_MODE) || defined(JUPYTER_KERNEL)
