#ifdef INTERACTIVE_MODE

#include "define_command.h"
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
// Return      : YT_SUCCESS or YT_FAIL
//-------------------------------------------------------------------------------------------------------
int define_command::run() {

    std::stringstream ss(m_Command);
    std::string arg;
    std::vector<std::string> arg_list;

    // get rid of %libyt, and start parsing from the second word.
    ss >> arg;
    while (ss >> arg) {
        arg_list.push_back(arg);
    }

    // call corresponding static method
    if (arg_list.size() == 2) {
        if      (arg_list[0].compare("load") == 0)    load_script(arg_list[1].c_str());
        else if (arg_list[0].compare("export") == 0)  export_script(arg_list[1].c_str());
    }
    else {
        if (g_myrank == s_Root) log_error("Unkown libyt command : %s\n", m_Command.c_str());
    }

    fflush(stdout);
    fflush(stderr);

    return YT_SUCCESS;
}


//-------------------------------------------------------------------------------------------------------
// Class       :  define_command
// Method      :  is_exit
//
// Notes       :  1. Parse m_Command to see if it is exit.
//                2. Since we need to set variable in interactive mode while loop, we single this method
//                   out of run method.
//
// Arguments   :  None
//
// Return      : true or false
//-------------------------------------------------------------------------------------------------------
bool define_command::is_exit() {
    std::size_t start_pos = 0;
    std::size_t found = m_Command.find("exit", start_pos);
    if (found != std::string::npos) return true;
    else return false;
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
// Arguments  :  char *filename : file name to reload
//
// Return     : YT_SUCCESS or YT_FAIL
//-------------------------------------------------------------------------------------------------------
int define_command::load_script(const char *filename) {

    // root rank reads script and broadcast to other ranks if compile successfully
    char *script = NULL;
    PyObject *src;
    if (g_myrank == s_Root) {
        // read file
        std::ifstream stream(filename);
        std::string line;
        std::stringstream ss;
        while (getline(stream, line)) { ss << line << "\n"; }
        stream.close();

        // check compilation, if failed return directly, so no need to allocate script.
        src = Py_CompileString(ss.str().c_str(), filename, Py_file_input);
        printf("[MPI %d] compile code\n", g_myrank);
        if (src == NULL) {
            PyErr_Print();
            int temp = -1;
            MPI_Bcast(&temp, 1, MPI_INT, s_Root, MPI_COMM_WORLD);
            return YT_FAIL;
        }

        // broadcast when compile successfully
        int script_len = (int) ss.str().length();
        script = (char*) malloc((script_len + 1) * sizeof(char));
        strncpy(script, ss.str().c_str(), script_len);
        script[script_len] = '\0';

        MPI_Bcast(&script_len, 1, MPI_INT, s_Root, MPI_COMM_WORLD);
        MPI_Bcast(script, script_len, MPI_CHAR, s_Root, MPI_COMM_WORLD);
    }
    else {
        // get script from file read by root rank, return YT_FAIL if script_len < 0
        int script_len;
        MPI_Bcast(&script_len, 1, MPI_INT, s_Root, MPI_COMM_WORLD);
        if (script_len < 0) return YT_FAIL;

        script = (char*) malloc((script_len + 1) * sizeof(char));
        MPI_Bcast(script, script_len, MPI_CHAR, s_Root, MPI_COMM_WORLD);
        script[script_len] = '\0';

        // compile code
        src = Py_CompileString(script, filename, Py_file_input);
    }

    // execute src in script's namespace
    PyObject *global_var = PyDict_GetItemString(g_py_interactive_mode, "script_globals");
    PyObject *local_var = global_var;
    PyObject *dum = PyEval_EvalCode(src, global_var, local_var);
    if (PyErr_Occurred()) PyErr_Print();

    // clean up
    free(script);
    Py_XDECREF(src);
    Py_XDECREF(dum);

    return YT_SUCCESS;
}


//-------------------------------------------------------------------------------------------------------
// Class         :  define_command
// Static Method :  export_script
//
// Notes         :  1.
//
// Arguments     :  char *filename : output file name
//
// Return        :  YT_SUCCESS or YT_FAIL
//-------------------------------------------------------------------------------------------------------
int define_command::export_script(const char *filename) {
    printf("Exporting script %s ...\n", filename);
    return YT_SUCCESS;
}
#endif // #ifdef INTERACTIVE_MODE
