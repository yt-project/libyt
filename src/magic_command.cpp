#if defined(INTERACTIVE_MODE) && defined(JUPYTER_KERNEL)
#include "magic_command.h"

#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

#include "func_status_list.h"
#include "libyt_python_shell.h"
#include "yt_combo.h"

int MagicCommand::s_Root = g_myroot;

//-------------------------------------------------------------------------------------------------------
// Class       :  MagicCommand
// Method      :  run
//
// Notes       :  1. This is a collective operation and an entry point for yt_run_JupyterKernel to call
//                   libyt defined command.
//                2. stringstream is slow and string copying is slow, but ..., too lazy to do that.
//                3. Returned from magic commands indicates if this operation is success or not.
//                   m_Undefine indicates if there is a corresponding method.
//
// Arguments   :  const std::string& command : command to run (default is "", which will get command from root).
//
// Return      : OutputData& m_OutputData : output data are generated and stored while calling methods.
//-------------------------------------------------------------------------------------------------------
OutputData& MagicCommand::run(const std::string& command) {
    SET_TIMER(__PRETTY_FUNCTION__);

    // get m_Command from s_Root
#ifndef SERIAL_MODE
    if (g_myrank == s_Root) {
        int code_len = (int)command.length();
        MPI_Bcast(&code_len, 1, MPI_INT, s_Root, MPI_COMM_WORLD);
        MPI_Bcast((void*)command.c_str(), code_len, MPI_CHAR, s_Root, MPI_COMM_WORLD);

        m_Command = command;
    } else {
        int code_len;
        MPI_Bcast(&code_len, 1, MPI_INT, s_Root, MPI_COMM_WORLD);

        char* code;
        code = (char*)malloc((code_len + 1) * sizeof(char));
        MPI_Bcast(code, code_len, MPI_CHAR, s_Root, MPI_COMM_WORLD);
        code[code_len] = '\0';

        m_Command = std::string(code);
        free(code);
    }
#else
    m_Command = command;
#endif

    std::stringstream ss(m_Command);
    std::string arg;
    std::vector<std::string> arg_list;

    bool run_success = false;

    // Mapping %libyt defined commands to methods
    ss >> arg;
    if (arg.compare("%libyt") == 0) {
        // parsing
        while (ss >> arg) {
            arg_list.emplace_back(arg);
        }

        // dispatch to method
        switch (arg_list.size()) {
            case 1: {
                // exit, status, help
                if (arg_list[0].compare("exit") == 0) {
                    run_success = exit();
                } else if (arg_list[0].compare("status") == 0) {
                    run_success = get_status();
                } else if (arg_list[0].compare("help") == 0) {
                    run_success = get_help_msg();
                }
                break;
            }
            case 2: {
                // load, export, run, idle, status
                if (arg_list[0].compare("load") == 0) {
                    run_success = load_script(arg_list[1]);
                } else if (arg_list[0].compare("export") == 0) {
                    run_success = export_script(arg_list[1]);
                } else if (arg_list[0].compare("run") == 0) {
                    run_success = set_func_run(arg_list[1], true);
                } else if (arg_list[0].compare("idle") == 0) {
                    run_success = set_func_run(arg_list[1], false);
                } else if (arg_list[0].compare("status") == 0) {
                    run_success = get_func_status(arg_list[1]);
                }
                break;
            }
            default: {
                // > 2, run with args
                if (arg_list[0].compare("run") == 0) {
                    run_success = set_func_run(arg_list[1], true, arg_list);
                }
                break;
            }
        }
    }

    if (g_myrank == s_Root) {
        if (m_Undefine) {
            m_OutputData.error = std::string("Unknown libyt command : ") + m_Command + std::string("\n") +
                                 std::string("(Type %libyt help for help ...)");
        }
        if (run_success) {
            g_libyt_python_shell.update_prompt_history(std::string("# ") + m_Command + std::string("\n"));
        }
    }

    return m_OutputData;
}

//-------------------------------------------------------------------------------------------------------
// Class      :  MagicCommand
// Method     :  exit
//
// Notes      :  1. Generate helper msg for JupyterKernel if this method is called, since this is not
//                  supported in JupyterKernel.
//
// Arguments  :  None
//
// Return     :  YT_SUCCESS or YT_FAIL
//-------------------------------------------------------------------------------------------------------
int MagicCommand::exit() {
    SET_TIMER(__PRETTY_FUNCTION__);

    m_Undefine = false;

    m_OutputData.error =
        std::string("%libyt exit is not supported in Jupyter, please use frontend UI 'shutdown' to exit libyt kernel.");

    return YT_FAIL;
}

//-------------------------------------------------------------------------------------------------------
// Class      :  MagicCommand
// Method     :  print_status
//
// Notes      :  1. Get all the function status in html format, without error msg.
//
// Arguments  :  None
//
// Return     :  YT_SUCCESS or YT_FAIL
//-------------------------------------------------------------------------------------------------------
int MagicCommand::get_status() {
    SET_TIMER(__PRETTY_FUNCTION__);

    m_Undefine = false;

    m_OutputData.mimetype = std::string("text/html");
    m_OutputData.output = g_func_status_list.get_summary_html();

    return YT_SUCCESS;
}

//-------------------------------------------------------------------------------------------------------
// Class      :  MagicCommand
// Method     :  print_help_msg
//
// Notes      :  1. Get help message in readme format.
//
// Arguments  :  None
//
// Return     :  YT_SUCCESS or YT_FAIL
//-------------------------------------------------------------------------------------------------------
int MagicCommand::get_help_msg() {
    SET_TIMER(__PRETTY_FUNCTION__);

    m_Undefine = false;

    m_OutputData.mimetype = std::string("text/markdown");
    m_OutputData.output =
        std::string("**Usage:** %libyt COMMAND\n"
                    "| COMMAND | Arguments | Usage |\n"
                    "|---|---|---|\n"
                    "| help |  | Print help messages. |\n"
                    "| load | filename | Load and run Python file `filename` in imported script's namespace. |\n"
                    "| export | filename | Export history to `filename`. |\n"
                    "| status |  | Get overall function status. <br> - \"Status\" indicates the execute status call by "
                    "libyt API. <br> - \"Run\" indicates the the function will run automatically. |\n"
                    "| status | function_name | Get `function_name` status. |\n"
                    "| run | function_name [arg1, arg2, ...] | Run `function_name` automatically by calling "
                    "`function_name(arg1, arg2, ...)`. Arguments are optional. |\n"
                    "| idle | function_name | Make `function_name` idle. |");

    return YT_SUCCESS;
}

//-------------------------------------------------------------------------------------------------------
// Class      :  MagicCommand
// Method     :  load_script
//
// Notes      :  1. This is a collective call.
//               2. Reload all the variables and functions defined inside the script. It will erase
//                  the previous Python workspace originally defined in the ongoing inline analysis.
//               3. Parse functions in script and add to g_func_status_list. If function name already
//                  exists in the list, the source code in libyt.interactive_mode["func_body"] will
//                  be rewritten.
//               4. Character in the file loaded cannot exceed INT_MAX.
//
// Arguments  :  const std::string& filename : file name to reload
//
// Return     : YT_SUCCESS or YT_FAIL
//-------------------------------------------------------------------------------------------------------
int MagicCommand::load_script(const std::string& filename) {
    SET_TIMER(__PRETTY_FUNCTION__);

    m_Undefine = false;

    // root rank checks the script, if worked, call execute_file
    if (g_myrank == s_Root) {
        // make sure file exist and read the file
        std::ifstream stream;
        stream.open(filename);
        if (!stream) {
#ifndef SERIAL_MODE
            int indicator = -1;
            MPI_Bcast(&indicator, 1, MPI_INT, s_Root, MPI_COMM_WORLD);
#endif
            m_OutputData.error = std::string("File ") + filename + std::string("doesn't exist.\n");
            m_OutputData.error += std::string("Loading script '") + filename + std::string("' ... failed.\n");
            return YT_FAIL;
        }
        std::string line;
        std::stringstream ss;
        while (getline(stream, line)) {
            ss << line << "\n";
        }
        stream.close();

        // check code validity
        CodeValidity code_validity = LibytPythonShell::check_code_validity(ss.str(), false, filename.c_str());
        if (code_validity.is_valid.compare("complete") == 0) {
#ifndef SERIAL_MODE
            int indicator = 1;
            MPI_Bcast(&indicator, 1, MPI_INT, s_Root, MPI_COMM_WORLD);
#endif
            // run file and parse output
            std::array<AccumulatedOutputString, 2> output = LibytPythonShell::execute_file(ss.str(), filename);
            for (int i = 0; i < 2; i++) {
                if (output[i].output_string.length() > 0) {
                    int offset = 0;
                    for (int r = 0; r < g_mysize; r++) {
                        std::string head =
                            std::string("\033[1;34m[MPI Process ") + std::to_string(r) + std::string("]\n\033[0;30m");
                        if (output[i].output_length[r] == 0) {
                            head += std::string("(None)\n");
                        }
                        output[i].output_string.insert(offset, head);
                        offset = offset + head.length() + output[i].output_length[r];
                    }
                }
            }
            m_OutputData.output = output[0].output_string;
            m_OutputData.error = output[1].output_string;
        } else {
#ifndef SERIAL_MODE
            int indicator = -1;
            MPI_Bcast(&indicator, 1, MPI_INT, s_Root, MPI_COMM_WORLD);
#endif
            m_OutputData.error = code_validity.error_msg + std::string("\n");
            m_OutputData.error += std::string("Loading script '") + filename + std::string("' ... failed\n");
            return YT_FAIL;
        }
    }
#ifndef SERIAL_MODE
    else {
        // return YT_FAIL if no file found or file cannot compile
        int indicator;
        MPI_Bcast(&indicator, 1, MPI_INT, s_Root, MPI_COMM_WORLD);

        if (indicator < 0) {
            return YT_FAIL;
        } else {
            std::array<AccumulatedOutputString, 2> output = LibytPythonShell::execute_file();
        }
    }
#endif

    // update libyt.interactive_mode["func_body"]
    LibytPythonShell::load_file_func_body(filename.c_str());

    // get function list defined inside the script, add the function name to list if it doesn't exist
    // and set to idle
    std::vector<std::string> func_list = LibytPythonShell::get_funcname_defined(filename.c_str());
    for (int i = 0; i < (int)func_list.size(); i++) {
        g_func_status_list.add_new_func(func_list[i].c_str(), 0);
    }

    // returned type from file must be text/plain, like other python results
    m_OutputData.mimetype = std::string("text/plain");
    m_OutputData.output += std::string("Loading script '") + filename + std::string("' ... done\n");

    return YT_SUCCESS;
}

//-------------------------------------------------------------------------------------------------------
// Class      :  MagicCommand
// Method     :  export_script
//
// Notes      :  1. Export input during this step's interactive loop.
//               2. Overwriting existing file.
//               3. Only process s_Root will write to file.
//
// Arguments  :  const char *filename : output file name
//
// Return     :  YT_SUCCESS or YT_FAIL
//-------------------------------------------------------------------------------------------------------
int MagicCommand::export_script(const std::string& filename) {
    SET_TIMER(__PRETTY_FUNCTION__);

    m_Undefine = false;

    if (g_myrank == s_Root) {
        std::ofstream dump_file;
        dump_file.open(filename, std::ofstream::trunc);
        dump_file << g_libyt_python_shell.get_prompt_history();
        dump_file.close();
        m_OutputData.mimetype = std::string("text/plain");
        m_OutputData.output = std::string("Exporting script to '") + filename + std::string("' ... done\n");
    }

    return YT_SUCCESS;
}

//-------------------------------------------------------------------------------------------------------
// Class      :  MagicCommand
// Method     :  set_func_run
//
// Notes      :  1. Determine which function will run or idle in next step.
//               2. arg_list is optional.
//
// Arguments  :  const std::string&        funcname : function name
//               bool                      run      : run in next inline process or not
//               std::vector<std::string>  arg_list : contains input args starting from index arg_list[2]
//
// Return     :  YT_SUCCESS or YT_FAIL
//-------------------------------------------------------------------------------------------------------
int MagicCommand::set_func_run(const std::string& funcname, bool run) {
    SET_TIMER(__PRETTY_FUNCTION__);

    m_Undefine = false;

    int index = g_func_status_list.get_func_index(funcname.c_str());
    if (index == -1) {
        if (g_myrank == s_Root) {
            m_OutputData.error = std::string("Function '") + funcname + std::string("' not found\n");
        }
        return YT_FAIL;
    } else {
        g_func_status_list[index].set_run(run);

        // update input args to empty string
        std::string args("");
        g_func_status_list[index].set_args(args);

        // print args if function is set to run
        if (g_myrank == s_Root) {
            m_OutputData.mimetype = std::string("text/plain");
            m_OutputData.output = std::string("Function '") + funcname + std::string("' set to ");
            m_OutputData.output += run ? std::string("run") : std::string("idle");
            m_OutputData.output += std::string(" ... done\n");
            if (run) {
                m_OutputData.output += std::string("Run ") + g_func_status_list[index].get_full_func_name() +
                                       std::string(" in next iteration\n");
            }
        }

        return YT_SUCCESS;
    }
}

int MagicCommand::set_func_run(const std::string& funcname, bool run, std::vector<std::string>& arg_list) {
    SET_TIMER(__PRETTY_FUNCTION__);

    m_Undefine = false;

    // get function index
    int index = g_func_status_list.get_func_index(funcname.c_str());
    if (index == -1) {
        if (g_myrank == s_Root) {
            m_OutputData.error = std::string("Function '") + funcname + std::string("' not found\n");
        }
        return YT_FAIL;
    }

    // update input args in func_status, and determine whether the wrapper is """ or '''
    bool wrapper_detected = false, unable_to_wrapped = false;
    std::string args("");

    // input parameters starts at index 2
    for (int i = 2; i < (int)arg_list.size(); i++) {
        // determining wrapper
        if (!wrapper_detected) {
            if (arg_list[i].find("\"\"\"") != std::string::npos) {
                wrapper_detected = true;
                g_func_status_list[index].set_wrapper(false);
            } else if (arg_list[i].find("'''") != std::string::npos) {
                wrapper_detected = true;
                g_func_status_list[index].set_wrapper(true);
            }
        } else {
            const char* wrapper = g_func_status_list[index].get_wrapper() ? "\"\"\"" : "'''";
            if (arg_list[i].find(wrapper) != std::string::npos) {
                unable_to_wrapped = true;
            }
        }

        // joining args
        args += arg_list[i];
        args += ",";
    }
    args.pop_back();

    if (unable_to_wrapped) {
        if (g_myrank == s_Root) {
            m_OutputData.error = std::string("Please avoid using both \"\"\" and ''' for triple quotes\n");
        }
        return YT_FAIL;
    } else {
        g_func_status_list[index].set_args(args);
        g_func_status_list[index].set_run(run);
        if (g_myrank == s_Root) {
            m_OutputData.mimetype = std::string("text/plain");
            m_OutputData.output = std::string("Function '") + funcname + std::string("' set to run ... done\n");
            m_OutputData.output += std::string("Run ") + g_func_status_list[index].get_full_func_name() +
                                   std::string(" in next iteration\n");
        }
        return YT_SUCCESS;
    }
}

//-------------------------------------------------------------------------------------------------------
// Class      :  MagicCommand
// Method     :  get_func_status
//
// Notes      :  1. Get function status and print error msg if has.
//               2. libyt.interactive_mode["func_err_msg"] only stores function's error msg when using
//                  yt_run_Function/yt_run_FunctionArguments.
//               3. A collective call, since it uses func_status::serial_print_error
//
// Arguments  :  const std::string& funcname : function name
//
// Return     :  YT_SUCCESS or YT_FAIL
//-------------------------------------------------------------------------------------------------------
int MagicCommand::get_func_status(const std::string& funcname) {
    SET_TIMER(__PRETTY_FUNCTION__);

    m_Undefine = false;

    // check if function exist
    int index = g_func_status_list.get_func_index(funcname.c_str());
    if (index == -1) {
        if (g_myrank == s_Root) {
            m_OutputData.error = std::string("Function '") + funcname + std::string("' not found\n");
        }
        return YT_FAIL;
    }

    int status = g_func_status_list[index].get_status();
    if (g_myrank == s_Root) {
        m_OutputData.mimetype = std::string("text/markdown");
        m_OutputData.output = std::string("#### `") + funcname + std::string("`\n");

        // Execute status
        m_OutputData.output += std::string("- **Execute status in current call:** ");
        if (status == 1) {
            m_OutputData.output += std::string("_Success_\n");
        } else if (status == 0) {
            m_OutputData.output += std::string("_Failed_\n");
        } else if (status == -1) {
            m_OutputData.output += std::string("_Idle_\n");
        }

        // Function call in next iteration
        m_OutputData.output += std::string("- **Function call in next iteration:** ");
        if (g_func_status_list[index].get_run() == 1) {
            m_OutputData.output +=
                std::string("`") + g_func_status_list[index].get_full_func_name() + std::string("`\n");
        } else {
            m_OutputData.output += std::string("(None)\n");
        }

        // Function definition
        m_OutputData.output += std::string("- **Function definition:**\n");
        m_OutputData.output += std::string("  ```python\n");
        // TODO: get function definition
        m_OutputData.output += std::string("  def func():\n"
                                           "      print('PLACE HOLDER')\n");
        m_OutputData.output += std::string("  ```\n");

        // Error message if it has (status == 0)
        if (status == 0) {
            m_OutputData.output += std::string("- **Error message from current call:**\n");
        }
    }

    // Call getting error message, this is a collective call
    if (status == 0) {
        // TODO: get error message and parse it
        m_OutputData.output += std::string("<details>\n"
                                           "  <summary>MPI Process 1</summary>\n"
                                           "  <p>Content 1 Content 1 Content 1 Content 1 Content 1 <br>\n"
                                           "      Content 1 Content 1 Content 1 Content 1 Content 1</p>\n"
                                           "</details>\n");
    }

    return YT_SUCCESS;
}

#endif