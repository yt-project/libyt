#ifdef INTERACTIVE_MODE

#include "define_command.h"
#include "func_status_list.h"
#include "yt_combo.h"
#include <iostream>
#include <sstream>
#include <fstream>
#include <vector>

int define_command::s_Root = 0;

//-------------------------------------------------------------------------------------------------------
// Class       :  define_command
// Method      :  run
//
// Notes       :  1. Parst m_Command, and call according method.
//                2. stringstream is slow and string copying is slow, but ..., too lazy to do that.
//
// Arguments   :  None
//
// Return      : true / false   : whether or not to exit interactive loop.
//-------------------------------------------------------------------------------------------------------
bool define_command::run() {

    std::stringstream ss(m_Command);
    std::string arg;
    std::vector<std::string> arg_list;

    // get rid of %libyt, and start parsing from the second word.
    ss >> arg;
    while (ss >> arg) {
        arg_list.emplace_back(arg);
    }

    // call corresponding static method
    if (arg_list.size() == 1) {
        if      (arg_list[0].compare("exit") == 0)      return true;
        else if (arg_list[0].compare("status") == 0)    print_status();
        else if (arg_list[0].compare("help") == 0)      print_help_msg();
    }
    else if (arg_list.size() == 2) {
        if      (arg_list[0].compare("load") == 0)      load_script(arg_list[1].c_str());
        else if (arg_list[0].compare("export") == 0)    export_script(arg_list[1].c_str());
        else if (arg_list[1].compare("run") == 0)       set_func_run(arg_list[0].c_str(), true);
        else if (arg_list[1].compare("idle") == 0)      set_func_run(arg_list[0].c_str(), false);
        else if (arg_list[1].compare("status") == 0)    get_func_status(arg_list[0].c_str());
    }

    if (m_Undefine && g_myrank == s_Root) log_error("Unkown libyt command : %s\n"
                                                    "(Type %%libyt help for help ...)\n", m_Command.c_str());

    fflush(stdout);
    fflush(stderr);

    return false;
}


//-------------------------------------------------------------------------------------------------------
// Class      :  define_command
// Method     :  print_status
//
// Notes      :  1. Print all the function status, without error msg.
//
//
// Arguments  :  None
//
// Return     :  YT_SUCCESS or YT_FAIL
//-------------------------------------------------------------------------------------------------------
int define_command::print_status() {
    m_Undefine = false;
    return YT_SUCCESS;
}


//-------------------------------------------------------------------------------------------------------
// Class      :  define_command
// Method     :  print_help_msg
//
// Notes      :  1.
//
// Arguments  :  None
//
// Return     :  YT_SUCCESS or YT_FAIL
//-------------------------------------------------------------------------------------------------------
int define_command::print_help_msg() {
    m_Undefine = false;
    if (g_myrank == s_Root) {
        printf("Usage: %libyt [options]\n");
    }
    return YT_SUCCESS;
}


//-------------------------------------------------------------------------------------------------------
// Class      :  define_command
// Method     :  load_script
//
// Notes      :  1. This is a collective call.
//               2. Reload all the variables and functions defined inside the script. It will erase
//                  the previous Python workspace originally defined in the ongoing inline analysis.
//               3. Parse functions in script and add to g_func_status_list. If function name already
//                  exists in the list, the source code in libyt.interactive_mode["func_body"] will
//                  be rewritten.
//               4. Charactar in the file loaded cannot exceed INT_MAX.
//
// Arguments  :  const char *filename : file name to reload
//
// Return     : YT_SUCCESS or YT_FAIL
//-------------------------------------------------------------------------------------------------------
int define_command::load_script(const char *filename) {
    m_Undefine = false;

    // root rank reads script and broadcast to other ranks if compile successfully
    PyObject *src;
    if (g_myrank == s_Root) {
        // read file
        std::ifstream stream;
        stream.open(filename);
        if (!stream) {
            int temp = -1;
            MPI_Bcast(&temp, 1, MPI_INT, s_Root, MPI_COMM_WORLD);
            printf("File %s doesn't exist.\n", filename);
            printf("Loading script %s ... failed\n", filename);
            return YT_FAIL;
        }
        std::string line;
        std::stringstream ss;
        while (getline(stream, line)) { ss << line << "\n"; }
        stream.close();

        // check compilation, if failed return directly, so no need to allocate script.
        src = Py_CompileString(ss.str().c_str(), filename, Py_file_input);
        if (src == NULL) {
            PyErr_Print();
            int temp = -1;
            MPI_Bcast(&temp, 1, MPI_INT, s_Root, MPI_COMM_WORLD);
            printf("Loading script %s ... failed\n", filename);
            return YT_FAIL;
        }

        // broadcast when compile successfully
        int script_len = (int) ss.str().length();
        MPI_Bcast(&script_len, 1, MPI_INT, s_Root, MPI_COMM_WORLD);
        MPI_Bcast(const_cast<char*>(ss.str().c_str()), script_len, MPI_CHAR, s_Root, MPI_COMM_WORLD);
    }
    else {
        // get script from file read by root rank, return YT_FAIL if script_len < 0
        int script_len;
        MPI_Bcast(&script_len, 1, MPI_INT, s_Root, MPI_COMM_WORLD);
        if (script_len < 0) return YT_FAIL;

        char *script;
        script = (char*) malloc((script_len + 1) * sizeof(char));
        MPI_Bcast(script, script_len, MPI_CHAR, s_Root, MPI_COMM_WORLD);
        script[script_len] = '\0';

        // compile code
        src = Py_CompileString(script, filename, Py_file_input);

        free(script);
    }

    // execute src in script's namespace
    PyObject *global_var = PyDict_GetItemString(g_py_interactive_mode, "script_globals");
    PyObject *dum = PyEval_EvalCode(src, global_var, global_var);
    if (PyErr_Occurred()) PyErr_Print();

    // update libyt.interactive_mode["func_body"]
    func_status_list::load_func_body(filename);

    // get function list defined inside the script
    std::vector<std::string> func_list = func_status_list::get_funcname_defined(filename);
    for (int i=0; i<func_list.size(); i++) {
        g_func_status_list.add_new_func(const_cast<char*>(func_list[i].c_str()), false);
    }

    // clean up
    Py_XDECREF(src);
    Py_XDECREF(dum);

    if (g_myrank == s_Root) printf("Loading script %s ... done\n", filename);

    return YT_SUCCESS;
}


//-------------------------------------------------------------------------------------------------------
// Class      :  define_command
// Method     :  export_script
//
// Notes      :  1.
//
// Arguments  :  const char *filename : output file name
//
// Return     :  YT_SUCCESS or YT_FAIL
//-------------------------------------------------------------------------------------------------------
int define_command::export_script(const char *filename) {
    m_Undefine = false;
    printf("Exporting script %s ...\n", filename);
    return YT_SUCCESS;
}


//-------------------------------------------------------------------------------------------------------
// Class      :  define_command
// Method     :  set_func_run
//
// Notes      :  1.
//
// Arguments  :  const char *funcname : function name
//               bool        run      : run in next inline process or not
//
// Return     :  YT_SUCCESS or YT_FAIL
//-------------------------------------------------------------------------------------------------------
int define_command::set_func_run(const char *funcname, bool run) {
    m_Undefine = false;

    int index = g_func_status_list.get_func_index(funcname);
    if (index == -1) {
        if (g_myrank == s_Root) printf("Function %s not found\n", funcname);
        return YT_FAIL;
    }
    else {
        g_func_status_list[index].set_run(run);
        if (g_myrank == s_Root) printf("Function %s set to %s ... done\n", funcname, run ? "run" : "idle");
        return YT_SUCCESS;
    }
}


int define_command::get_func_status(const char *funcname) {
    return YT_SUCCESS;
}

#endif // #ifdef INTERACTIVE_MODE
