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

    // Mapping %libyt defined commands to methods
    ss >> arg;
    if (arg.compare("%libyt") == 0) {
        // parsing
        while (ss >> arg) {
            arg_list.emplace_back(arg);
        }

        // call corresponding method
        if (arg_list.size() == 1) {
            if      (arg_list[0].compare("exit") == 0)      return true;
            else if (arg_list[0].compare("status") == 0)    print_status();
            else if (arg_list[0].compare("help") == 0)      print_help_msg();
        }
        else if (arg_list.size() == 2) {
            if      (arg_list[0].compare("load") == 0)      load_script(arg_list[1].c_str());
            else if (arg_list[0].compare("export") == 0)    export_script(arg_list[1].c_str());
            else if (arg_list[0].compare("run") == 0)       set_func_run(arg_list[1].c_str(), true);
            else if (arg_list[0].compare("idle") == 0)      set_func_run(arg_list[1].c_str(), false);
            else if (arg_list[0].compare("status") == 0)    get_func_status(arg_list[1].c_str());
        }
        else if (arg_list.size() > 2) {
            if      (arg_list[0].compare("run") == 0)       set_func_run(arg_list[1].c_str(), true, arg_list);
        }
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
//               2. Call g_func_status_list.print_summary().
//
// Arguments  :  None
//
// Return     :  YT_SUCCESS or YT_FAIL
//-------------------------------------------------------------------------------------------------------
int define_command::print_status() {
    m_Undefine = false;
    g_func_status_list.print_summary();
    return YT_SUCCESS;
}


//-------------------------------------------------------------------------------------------------------
// Class      :  define_command
// Method     :  print_help_msg
//
// Notes      :  1. Print help message.
//
// Arguments  :  None
//
// Return     :  YT_SUCCESS or YT_FAIL
//-------------------------------------------------------------------------------------------------------
int define_command::print_help_msg() {
    m_Undefine = false;
    if (g_myrank == s_Root) {
        printf("Usage: %%libyt [options]\n");
        printf("  %-6s  %-11s  %-8s  %s\n", "help",   "",            "",         "print help message");
        printf("  %-6s  %-11s  %-8s  %s\n", "exit",   "",            "",         "exit and continue simulation");
        printf("  %-6s  %-11s  %-8s  %s\n", "status", "",            "",         "get overall function status");
        printf("  %-6s  %-11s  %-8s  %s\n", "status", "<func_name>", "",         "get function status");
        printf("  %-6s  %-11s  %-8s  %s\n", "run",    "<func_name>", "[arg1 ]", "function will run in next iteration");
        printf("  %-6s  %-11s  %-8s  %s\n", "",       "",            "",         "using args, ex: func(arg1, arg2)");
        printf("  %-6s  %-11s  %-8s  %s\n", "idle",   "<func_name>", "",         "function will idle in next iteration");
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
    func_status_list::load_file_func_body(filename);

    // get function list defined inside the script, add the function name to list if it doesn't exist
    // and set to idle
    std::vector<std::string> func_list = func_status_list::get_funcname_defined(filename);
    for (int i=0; i<func_list.size(); i++) {
        g_func_status_list.add_new_func(func_list[i].c_str(), 0);
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
// Notes      :  1. Export input during this step's interactive loop.
//               2. Let user maintain their script imported.
//               3. todo
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
// Notes      :  1. Determine which function will run or idle in next step.
//               2. arg_list is optional.
//
// Arguments  :  const char               *funcname : function name
//               bool                      run      : run in next inline process or not
//               std::vector<std::string>  arg_list : contains input args starting from index arg_list[2]
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

        // update input args to empty string
        std::string args("");
        g_func_status_list[index].set_args(args);

        return YT_SUCCESS;
    }
}

int define_command::set_func_run(const char *funcname, bool run, std::vector<std::string>& arg_list) {
    m_Undefine = false;

    int index = g_func_status_list.get_func_index(funcname);
    if (index == -1) {
        if (g_myrank == s_Root) printf("Function %s not found\n", funcname);
        return YT_FAIL;
    }
    else {
        g_func_status_list[index].set_run(run);
        if (g_myrank == s_Root) printf("Function %s set to %s ... done\n", funcname, run ? "run" : "idle");

        // update input args in func_status
        std::string args("");
        for (int i=2; i<arg_list.size(); i++) {
            args = args + arg_list[i] + " ,";
        }
        args.pop_back();
        g_func_status_list[index].set_args(args);

        return YT_SUCCESS;
    }
}


//-------------------------------------------------------------------------------------------------------
// Class      :  define_command
// Method     :  get_func_status
//
// Notes      :  1. Get function status and print error msg if has.
//               2. libyt.interactive_mode["func_err_msg"] only stores function's error msg when using
//                  yt_inline/yt_inline_argument.
//               3. A collective call, since it uses func_status::serial_print_error
//
// Arguments  :  const char *funcname : function name
//
// Return     :  YT_SUCCESS or YT_FAIL
//-------------------------------------------------------------------------------------------------------
int define_command::get_func_status(const char *funcname) {
    m_Undefine = false;

    // check if function exist
    int index = g_func_status_list.get_func_index(funcname);
    if (index == -1) {
        if (g_myrank == s_Root) printf("Function %s not found\n", funcname);
        return YT_FAIL;
    }

    // print function status and function body
    int status = g_func_status_list[index].get_status();
    if (g_myrank == s_Root) {
        printf("%s ... ", g_func_status_list[index].get_func_name());
        if      (status == 1)  printf("success\n");
        else if (status == 0)  printf("failed\n");
        else if (status == -1) printf("idle\n");

        printf("\033[1;35m"); // bold purple
        printf("[Function Def]\n");
        g_func_status_list[index].print_func_body(2, 0);
    }

    // print error msg if it failed when running in yt_inline/yt_inline_argument. (collective call)
    if (status == 0) {
        if (g_myrank == s_Root) {
            printf("\033[1;35m"); // bold purple
            printf("[Error Msg]\n");
        }
        g_func_status_list[index].serial_print_error(2, 1);
        printf("\033[0;37m");
    }

    return YT_SUCCESS;
}

#endif // #ifdef INTERACTIVE_MODE
