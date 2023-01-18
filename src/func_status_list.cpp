#ifdef INTERACTIVE_MODE

#include "yt_combo.h"
#include <string.h>
#include "func_status_list.h"


std::array<std::string, 2> func_status_list::s_NotDone_ErrMsg = {"", ""};
std::array<PyObject*, 2>   func_status_list::s_NotDone_PyErr = {NULL, NULL};

//-------------------------------------------------------------------------------------------------------
// Class       :  func_status_list
// Method      :  reset
//
// Notes       :  1. Reset every func_status data member m_Status in list to -1 (not run yet).
//
// Arguments   :  None
//
// Return      : YT_SUCCESS
//-------------------------------------------------------------------------------------------------------
int func_status_list::reset() {
    for (int i=0; i<size(); i++) {
        m_FuncStatusList[i].set_status(-1);
    }
    return YT_SUCCESS;
}


//-------------------------------------------------------------------------------------------------------
// Class       :  func_status_list
// Method      :  print_summary
//
// Notes       :  1. Print function status and error msg in func_status_list.
//                2. normal      -> bold white
//                   idle        -> bold blue
//                   not run yet -> bold yellow
//                   success     -> bold green
//                   failed      -> bold red
//                   MPI process -> bold cyan
//
// Arguments   :  None
//
// Return      : YT_SUCCESS or YT_FAIL
//-------------------------------------------------------------------------------------------------------
int func_status_list::print_summary() {
    // make sure every rank has reach here, so that printing in other ranks are done
    fflush(stdout);
    fflush(stderr);

    if (g_myrank == 0) {
        printf("\033[1;37m");
        printf("=====================================================================\n");
        printf("  %-40s     %-12s   %s\n", "Inline Function", "Status", "Run");
        printf("---------------------------------------------------------------------\n");
        for (int i=0; i<size(); i++) {
            printf("\033[1;37m"); // change to bold white
            printf("  * %-43s", m_FuncStatusList[i].get_func_name());
            int run = m_FuncStatusList[i].get_run();
            int status = m_FuncStatusList[i].get_status();

            if (status == 0) {
                printf("\033[1;31m"); // bold red: failed
                printf("%-12s", "failed");
            }
            else if (status == 1) {
                printf("\033[1;32m"); // bold green: success
                printf("%-12s", "success");
            }
            else if (status == -1 ) {
                printf("\033[1;34m"); // bold blue: idle
                printf("%-12s", "idle");
            }
            else {
                printf("\033[0;37m"); // change to white
                printf("%-12s (%d)", "unkown status", status);
            }

            printf("\033[1;33m"); // bold yellow
            if (run == 1) printf("%5s\n", "V");
            else          printf("%5s\n", "X");

            fflush(stdout);
        }
        printf("\033[1;37m");
        printf("=====================================================================\n");
        printf("\033[0;37m"); // change to white
    }

    return YT_SUCCESS;
}


//-------------------------------------------------------------------------------------------------------
// Class       :  func_status_list
// Method      :  get_func_index
//
// Notes       :  1. Look up index of func_name in m_FuncStatusList.
//
// Arguments   :  char   *func_name: inline function name
//
// Return      :  index : index of func_name in list, return -1 if doesn't exist.
//-------------------------------------------------------------------------------------------------------
int func_status_list::get_func_index(const char *func_name) {
    int index = -1;
    for (int i=0; i<size(); i++) {
        if (strcmp(m_FuncStatusList[i].get_func_name(), func_name) == 0) {
            index = i;
            break;
        }
    }
    return index;
}


//-------------------------------------------------------------------------------------------------------
// Class       :  func_status_list
// Method      :  add_new_func
//
// Notes       :  1. Check if func_name is defined inside the vector, if not create one.
//                2. Return function index.
//
// Arguments   :  char   *func_name: inline function name
//                int     run      : run in next inline analysis or not.
//
// Return      : Function index in list.
//-------------------------------------------------------------------------------------------------------
int func_status_list::add_new_func(const char *func_name, int run) {
    // Check if func_name exist, return YT_SUCCESS if exist
    int index = get_func_index(func_name);
    if (index >= 0) return index;

    // add func_name, since it adds to the end, its index is equal to original size
    index = size();
    func_status new_func(func_name, run);
    m_FuncStatusList.push_back(new_func);

    return index;
}


//-------------------------------------------------------------------------------------------------------
// Class       :  func_status_list
// Method      :  run_func
//
// Notes       :  1. This is a collective call. It executes new added functions that haven't run by
//                   yt_inline/yt_inline_argument yet.
//                2. Only supports function that don't have input arguments. It runs function func() in
//                   Python.
//
// Arguments   :  None
//
// Return      :  YT_SUCCESS
//-------------------------------------------------------------------------------------------------------
int func_status_list::run_func() {
    for (int i=0; i<size(); i++) {
        int run = m_FuncStatusList[i].get_run();
        int status = m_FuncStatusList[i].get_status();
        if (run == 1 && status == -1) {
            // command
            char *funcname = m_FuncStatusList[i].get_func_name();
            int command_width = 150 + strlen(g_param_libyt.script) + strlen(funcname) * 2;
            char *command = (char*) malloc(command_width * sizeof(char));
            sprintf(command, "try:\n"
                             "    %s.%s(%s)\n"
                             "except Exception as e:\n"
                             "    libyt.interactive_mode[\"func_err_msg\"][\"%s\"] = traceback.format_exc()\n",
                             g_param_libyt.script, funcname, m_FuncStatusList[i].get_args().c_str(), funcname);

            // run and update status
            m_FuncStatusList[i].set_status(-2);
            PyRun_SimpleString(command);
            m_FuncStatusList[i].get_status();

            // clean up
            free(command);
        }
    }
    return YT_SUCCESS;
}


//-------------------------------------------------------------------------------------------------------
// Class         :  func_status_list
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
int func_status_list::load_file_func_body(const char *filename) {
    int command_len = 500 + strlen(filename);
    char *command = (char*) malloc(command_len * sizeof(char));
    sprintf(command, "for key in libyt.interactive_mode[\"script_globals\"].keys():\n"
                     "    if key.startswith(\"__\") and key.endswith(\"__\"):\n"
                     "        continue\n"
                     "    else:\n"
                     "        var = libyt.interactive_mode[\"script_globals\"][key]\n"
                     "        try:\n"
                     "            if callable(var) and inspect.getsourcefile(var).split(\"/\")[-1] == \"%s\":\n"
                     "                libyt.interactive_mode[\"func_body\"][key] = inspect.getsource(var)\n"
                     "        except:\n"
                     "            pass\n", filename);

    if (PyRun_SimpleString(command) == 0) {
        log_debug("Loading function body in script %s ... done\n", filename);
        free(command);
        return YT_SUCCESS;
    }
    else {
        log_debug("Loading function body in script %s ... failed\n", filename);
        free(command);
        return YT_FAIL;
    }
}


//-------------------------------------------------------------------------------------------------------
// Class         :  func_status_list
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
int func_status_list::load_input_func_body(char *code) {
    // prepare subspace to silent printing from python
    PyObject *py_new_dict = PyDict_New();
    PyDict_SetItemString(py_new_dict, "__builtins__", PyEval_GetBuiltins());
    std::string command_str("import os, contextlib\n"
                            "with open(os.devnull, 'w') as f, contextlib.redirect_stdout(f):\n");
    std::string code_str(code);
    std::size_t start_pos = 0;
    while (true) {
        std::size_t found = code_str.find('\n', start_pos);
        if (found != std::string::npos) {
            command_str = command_str + "    ";
            for (std::size_t c=start_pos; c<found; c++) {
                command_str = command_str + code_str[c];
            }
            command_str = command_str + "\n";
        }
        else {
            break;
        }
        start_pos = found + 1;
    }

    // detecting callables
    // detecting callables: loop over keys in new dict, and check if it is callable
    PyObject *py_src = Py_CompileString(command_str.c_str(), "<libyt-stdin>", Py_file_input);
    if (py_src != NULL) {
        PyObject *py_dum_detect = PyEval_EvalCode(py_src, py_new_dict, py_new_dict);
        PyObject *py_new_dict_keys = PyDict_Keys(py_new_dict);
        Py_ssize_t py_size = PyList_GET_SIZE(py_new_dict_keys);
        for (Py_ssize_t i=0; i<py_size; i++) {
            if (PyCallable_Check(PyDict_GetItem(py_new_dict, PyList_GET_ITEM(py_new_dict_keys, i)))) {
                // add new function to g_func_status_list and set to idle
                // if function exists already, get its index
                const char *func_name = PyUnicode_AsUTF8(PyList_GET_ITEM(py_new_dict_keys, i));
                int func_index = g_func_status_list.add_new_func(func_name, 0);
                printf("get functor: %s, %d\n", func_name, func_index);
                // update function body. todo
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
// Class         :  func_status_list
// Static Method :  get_funcname_defined
//
// Notes         :  1. This is a static method.
//                  2. It grabs functions or any callable object's name defined in filename.
//
// Arguments     :  const char *filename: update function body for function defined inside filename
//
// Return        : std::vector<std::string> contains a list of function name defined in filename.
//-------------------------------------------------------------------------------------------------------
std::vector<std::string> func_status_list::get_funcname_defined(const char *filename) {
    int command_len = 400 + strlen(filename);
    char *command = (char*) malloc(command_len * sizeof(char));
    sprintf(command, "libyt.interactive_mode[\"temp\"] = []\n"
                     "for key in libyt.interactive_mode[\"script_globals\"].keys():\n"
                     "    if key.startswith(\"__\") and key.endswith(\"__\"):\n"
                     "        continue\n"
                     "    else:\n"
                     "        var = libyt.interactive_mode[\"script_globals\"][key]\n"
                     "        if callable(var) and inspect.getsourcefile(var).split(\"/\")[-1] == \"%s\":\n"
                     "            libyt.interactive_mode[\"temp\"].append(key)\n", filename);
    if (PyRun_SimpleString(command) != 0) log_error("Unable to grab functions in python script %s.\n", filename);

    PyObject *py_func_list = PyDict_GetItemString(g_py_interactive_mode, "temp");
    Py_ssize_t py_list_len = PyList_Size(py_func_list);
    std::vector<std::string> func_list;
    func_list.reserve((long)py_list_len);
    for (Py_ssize_t i=0; i<py_list_len; i++) {
        const char *func_name = PyUnicode_AsUTF8(PyList_GET_ITEM(py_func_list, i));
        func_list.emplace_back(std::string(func_name));
    }

    // clean up
    free(command);
    PyRun_SimpleString("del libyt.interactive_mode[\"temp\"]");

    return func_list;
}


//-------------------------------------------------------------------------------------------------------
// Class         :  func_status_list
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
int func_status_list::set_exception_hook() {
    const char *command = "import sys\n"
                          "def mpi_libyt_interactive_mode_excepthook(exception_type, exception_value, tb):\n"
                          "    from mpi4py import MPI\n"
                          "    myrank = MPI.COMM_WORLD.Get_rank()\n"
                          "    traceback.print_tb(tb)\n"
                          "    print(\"[YT_ERROR  ] {}: {}\".format(exception_type.__name__, exception_value))\n"
                          "    print(\"[YT_ERROR  ] Error occurred on rank {}.\".format(myrank))\n"
                          "sys.excepthook = mpi_libyt_interactive_mode_excepthook\n";
    if (PyRun_SimpleString(command) == 0) return YT_SUCCESS;
    else return YT_FAIL;
}


//-------------------------------------------------------------------------------------------------------
// Class         :  func_status_list
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
int func_status_list::init_not_done_err_msg() {
    // error msg from not done yet statement to grab
    std::array<std::string, s_NotDone_Num> not_done_statement = { std::string("if 1==1:\n"),
                                                                  std::string("tri = \"\"\"\n") };

    // get python error type and its statement.
    PyObject *py_src, *py_exc, *py_val, *py_traceback, *py_obj;
    const char *err_msg;
    for (int i=0; i<s_NotDone_Num; i++) {
        py_src = Py_CompileString(not_done_statement[i].c_str(), "<get err msg>", Py_single_input);
        PyErr_Fetch(&py_exc, &py_val, &py_traceback);
        PyArg_ParseTuple(py_val, "sO", &err_msg, &py_obj);

        s_NotDone_ErrMsg[i] = std::string(err_msg);
        s_NotDone_PyErr[i]  = py_exc;

        PyErr_Clear();
    }

    // clean up
    PyErr_Clear();
    Py_XDECREF(py_src);
    Py_XDECREF(py_exc);
    Py_XDECREF(py_val);
    Py_XDECREF(py_traceback);
    Py_XDECREF(py_obj);

    return YT_SUCCESS;
}


//-------------------------------------------------------------------------------------------------------
// Class         :  func_status_list
// Static Method :  is_not_done_err_msg
//
// Notes         :  1. This is a static method.
//                  2. Check current Python state to see if it is error msg that is caused by user input
//                     not done yet.
//                  3. If it is indeed caused by user not done its input, clear error buffer. If not,
//                     restore the buffer and that yt_interactive_mode print the error msg.
//                  4. s_NotDone_ErrMsg's and s_NotDone_PyErr's elements are one-to-one relationship.
//
// Arguments     :  None
//
// Return        :  true / false : true for user hasn't done inputting yet.
//-------------------------------------------------------------------------------------------------------
bool func_status_list::is_not_done_err_msg() {

    bool user_not_done = false;

    for (int i=0; i<s_NotDone_Num; i++) {
        // check error type
        if (PyErr_ExceptionMatches(s_NotDone_PyErr[i])) {
            // fetch err msg
            PyObject *py_exc, *py_val, *py_traceback, *py_obj;
            const char *err_msg = "";
            PyErr_Fetch(&py_exc, &py_val, &py_traceback);
            PyArg_ParseTuple(py_val, "sO", &err_msg, &py_obj);

            // check err msg:
            //   if it is not done yet, no need to restore the error msg buffer in python
            if (s_NotDone_ErrMsg[i].compare(err_msg) == 0) {
                user_not_done = true;
                Py_XDECREF(py_exc);
                Py_XDECREF(py_val);
                Py_XDECREF(py_traceback);
                Py_XDECREF(py_obj);
            }
            else {
                // restore err msg, and I no longer own py_exc's, py_val's, and py_traceback's reference
                PyErr_Restore(py_exc, py_val, py_traceback);
                Py_XDECREF(py_obj);
            }

            break;
        }
    }

    return user_not_done;
}

#endif // #ifdef INTERACTIVE_MODE
