#if defined(INTERACTIVE_MODE) || defined(JUPYTER_KERNEL)

#include "new_magic_command.h"

#include <fstream>

#include "yt_combo.h"
#include "yt_global.h"
#include "yt_macro.h"

int NewMagicCommand::root_ = g_myroot;

//-------------------------------------------------------------------------------------------------------
// Method      :  SplitBySpace
// Description :  Split the code based on spaces
//
// Notes       :  1. Using std::stringstream to split the code.
//
// Arguments   :  const std::string& code  : raw code
//
// Return      :  std::vector<std::string>
//-------------------------------------------------------------------------------------------------------
static std::vector<std::string> SplitBySpace(const std::string& code) {
    SET_TIMER(__PRETTY_FUNCTION__);

    std::vector<std::string> output;
    std::stringstream ss(code);
    std::string elem;
    while (ss >> elem) {
        output.emplace_back(elem);
    }

    return output;
}

//-------------------------------------------------------------------------------------------------------
// Class       :  MagicCommand
// Method      :  Run
//
// Notes       :  1. This is a collective operation and an entry point to call libyt defined magic command (%libyt).
//                2. stringstream is slow and string copying is slow, but ..., too lazy to do that.
//                3. Output data and error message are cached in output_.
//                4. Only root rank has full history.
//                5. Make it able to call it again and again.
//
// Arguments   :  const std::string& command : command to run (default is "", which will get command from root).
//
// Return      : MagicCommandOutput& : output data are generated and cached while calling methods.
//-------------------------------------------------------------------------------------------------------
MagicCommandOutput& NewMagicCommand::Run(const std::string& command) {
    SET_TIMER(__PRETTY_FUNCTION__);

    // Get command from root_ if command is empty
#ifndef SERIAL_MODE
    if (g_myrank == root_) {
        int code_len = (int)command.length();
        MPI_Bcast(&code_len, 1, MPI_INT, root_, MPI_COMM_WORLD);
        MPI_Bcast((void*)command.c_str(), code_len, MPI_CHAR, root_, MPI_COMM_WORLD);

        command_ = command;
    } else {
        int code_len;
        MPI_Bcast(&code_len, 1, MPI_INT, root_, MPI_COMM_WORLD);

        char* code;
        code = (char*)malloc((code_len + 1) * sizeof(char));
        MPI_Bcast(code, code_len, MPI_CHAR, root_, MPI_COMM_WORLD);
        code[code_len] = '\0';

        command_ = std::string(code);
        free(code);
    }
#else
    command_ = command;
#endif

    std::vector<std::string> code_list = SplitBySpace(command_);
    size_t code_list_size = code_list.size();

    // Reset the status
    command_undefined_ = true;
    output_ = MagicCommandOutput();
    bool write_to_history = false;

    // Dispatch commands to methods: works fine, but I think this is ugly
    // probably will replace it with abstract factory pattern.
    if (code_list_size > 1 && code_list[0] == "%libyt") {
        if (code_list[1] == "exit") {
            write_to_history = Exit();
        } else if (code_list[1] == "help") {
            if (entry_point_ == kLibytInteractiveMode || entry_point_ == kLibytReloadScript) {
                write_to_history = GetHelpMsgText();
            } else if (entry_point_ == kLibytJupyterKernel) {
                write_to_history = GetHelpMsgMarkdown();
            }
        } else if (code_list[1] == "load") {
            write_to_history = LoadScript(code_list);
        } else if (code_list[1] == "export") {
            write_to_history = ExportScript(code_list);
        } else if (code_list[1] == "status") {
            if (code_list_size > 2) {
                if (entry_point_ == kLibytInteractiveMode || entry_point_ == kLibytReloadScript) {
                    write_to_history = GetFunctionStatusText(code_list);
                } else if (entry_point_ == kLibytJupyterKernel) {
                    write_to_history = GetFunctionStatusMarkdown(code_list);
                }
            } else {
                if (entry_point_ == kLibytInteractiveMode || entry_point_ == kLibytReloadScript) {
                    write_to_history = GetStatusText();
                } else if (entry_point_ == kLibytJupyterKernel) {
                    write_to_history = GetStatusHtml();
                }
            }
        } else if (code_list[1] == "run") {
            write_to_history = SetFunctionRun(code_list);
        } else if (code_list[1] == "idle") {
            write_to_history = SetFunctionIdle(code_list);
        }
    }

    if (command_undefined_) {
        output_.exit_entry_point = false;
        output_.status = "Error";
        output_.error = std::string("Unknown libyt command : ") + command_ + std::string("\n") +
                        std::string("(Type %libyt help for help ...)");
    }
    if (write_to_history && g_myrank == root_) {
        g_libyt_python_shell.update_prompt_history(std::string("# ") + command_ + std::string("\n"));
    }

    return output_;
}

//-------------------------------------------------------------------------------------------------------
// Class      :  MagicCommand
// Method     :  Exit
//
// Notes      :  1. Generate helper msg for JupyterKernel if this method is called, since this is not
//                  supported in JupyterKernel.
//               2. Exit libyt prompt or reloading script mode in interactive mode and reloading script.
//                  and then clear prompt history.
//
// Arguments  :  None
//
// Return     :  YT_SUCCESS or YT_FAIL
//-------------------------------------------------------------------------------------------------------
int NewMagicCommand::Exit() {
    SET_TIMER(__PRETTY_FUNCTION__);

    command_undefined_ = false;

    if (entry_point_ == kLibytJupyterKernel) {
        output_.exit_entry_point = false;
        output_.status = "Error";
        output_.error = std::string(
            "\"%libyt exit\" is not supported in Jupyter, please use frontend UI 'shutdown' to exit libyt kernel.");

        return YT_FAIL;
    } else if (entry_point_ == kLibytReloadScript) {
        output_.exit_entry_point = false;
        output_.status = "Error";
        output_.error =
            std::string("\"%libyt exit\" is not supported in reloading script, please create flagged file to exit.");

        return YT_FAIL;
    } else {
        g_libyt_python_shell.clear_prompt_history();

        output_.exit_entry_point = true;
        output_.status = "Success";

        return YT_SUCCESS;
    }
}

//-------------------------------------------------------------------------------------------------------
// Class      :  MagicCommand
// Method     :  GetStatusHtml
//
// Notes      :  1. Get all the function status in html format, without error msg.
//               2. Getting function information (status) is a collective call .
//
// Arguments  :  None
//
// Return     :  YT_SUCCESS or YT_FAIL
//-------------------------------------------------------------------------------------------------------
int NewMagicCommand::GetStatusHtml() {
    SET_TIMER(__PRETTY_FUNCTION__);

    command_undefined_ = false;

    output_.status = "Success";
    output_.mimetype = "text/html";

    const char* kSuccessCell =
        "<td><span style=\"color:#28B463;font-weight:bold;font-family:'arial'\">Success</span></td>";
    const char* kFailedCell =
        "<td><span style=\"color:#E74C3C;font-weight:bold;font-family:'arial'\">Failed</span></td>";
    const char* kIdleCell = "<td><span style=\"color:#2874A6;font-weight:bold;font-family:'arial'\">Idle</span></td>";
    const char* kUnknownCell =
        "<td><span style=\"color:#A569BD;font-weight:bold;font-family:'arial'\">Unknown</span></td>";
    const char* kWillRunCell = "<td><span style=\"color:#F1C40F;font-weight:bold;font-family:'arial'\">V</span></td>";
    const char* kWillIdleCell = "<td><span style=\"color:#F1C40F;font-weight:bold;font-family:'arial'\">X</span></td>";

    output_.output += "<table style=\"width: 100%\"><tr><th>Inline Function</th><th>Status</th><th>Run</th></tr>";

    for (int i = 0; i < g_func_status_list.size(); i++) {
        // Get function name
        output_.output += "<tr><td style=\"text-alight: left;\"><span style=\"font-family:'Courier New'\">";
        output_.output += g_func_status_list[i].get_func_name();
        output_.output += "</span></td>";

        // Get function status
        int status = g_func_status_list[i].get_status();
        if (status == 0) {
            output_.output += kFailedCell;
        } else if (status == 1) {
            output_.output += kSuccessCell;
        } else if (status == -1) {
            output_.output += kIdleCell;
        } else {
            output_.output += kUnknownCell;
        }

        // Get function run status
        int run = g_func_status_list[i].get_run();
        if (run == 1) {
            output_.output += kWillRunCell;
        } else {
            output_.output += kWillIdleCell;
        }

        output_.output += "</tr>";
    }

    output_.output += "</table>";

    return YT_SUCCESS;
}

//-------------------------------------------------------------------------------------------------------
// Class      :  MagicCommand
// Method     :  GetStatusText
//
// Notes      :  1. Get all the function status, without error msg.
//               2. Getting function information (status) is a collective call.
//
// Arguments  :  None
//
// Return     :  YT_SUCCESS or YT_FAIL
//-------------------------------------------------------------------------------------------------------
int NewMagicCommand::GetStatusText() {
    SET_TIMER(__PRETTY_FUNCTION__);

    command_undefined_ = false;

    output_.status = "Success";

    const char* kSuccess = "success     ";
    const char* kFailed = "failed      ";
    const char* kIdle = "idle        ";
    const char* kUnknown = "unknown     ";
    const char* kV = "    V";
    const char* kX = "    X";

    if (entry_point_ == kLibytInteractiveMode) {
        kSuccess = "\033[1;32msuccess     ";
        kFailed = "\033[1;31mfailed      ";
        kIdle = "\033[1;34midle        ";
        kUnknown = "\033[0;37munknown     ";
        kV = "\033[1;33m    V";
        kX = "\033[1;33m    X";
    }

    const int kStringMaxSize = 1024;
    char dest[kStringMaxSize];
    int snprintf_return = -1;

    if (entry_point_ == kLibytInteractiveMode) {
        output_.output += "\033[1;37m";
    }

    snprintf(dest, kStringMaxSize,
             "==========================================================================\n"
             "  %-40s     %-12s   %s\n"
             "--------------------------------------------------------------------------\n",
             "Inline Function", "Status", "Run/Idle");
    output_.output += dest;

    for (int i = 0; i < g_func_status_list.size(); i++) {
        // Get function name
        snprintf_return = snprintf(dest, kStringMaxSize, "  * %-43s", g_func_status_list[i].get_func_name());
        if (entry_point_ == kLibytInteractiveMode) {
            output_.output += "\033[1;37m";
        }
        if (snprintf_return >= 0 && snprintf_return <= kStringMaxSize) {
            output_.output += dest;
        } else {
            output_.status = "Error";
            output_.error = "Function name too long.\n";
            return YT_FAIL;
        }

        // Get function status and run
        int run = g_func_status_list[i].get_run();
        int status = g_func_status_list[i].get_status();
        if (status == 0) {
            output_.output += kFailed;
        } else if (status == 1) {
            output_.output += kSuccess;
        } else if (status == -1) {
            output_.output += kIdle;
        } else {
            output_.output += kUnknown;
            output_.error += std::string("Unknown status code ") + std::to_string(status) + std::string("\n");
        }

        if (run == 1) {
            output_.output += kV;
        } else {
            output_.output += kX;
        }

        output_.output += "\n";
    }

    if (entry_point_ == kLibytInteractiveMode) {
        output_.output += "\033[1;37m"
                          "==========================================================================\n"
                          "\033[0;37m";
    } else {
        output_.output += "==========================================================================\n";
    }

    return YT_SUCCESS;
}

//-------------------------------------------------------------------------------------------------------
// Class      :  MagicCommand
// Method     :  GetHelpMsgMarkdown
//
// Notes      :  1. Get help message in readme format.
//
// Arguments  :  None
//
// Return     :  YT_SUCCESS or YT_FAIL
//-------------------------------------------------------------------------------------------------------
int NewMagicCommand::GetHelpMsgMarkdown() {
    SET_TIMER(__PRETTY_FUNCTION__);

    command_undefined_ = false;

    output_.status = "Success";
    output_.mimetype = "text/markdown";
    output_.output =
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
// Method     :  GetHelpMsgText
//
// Notes      :  1. Get help message in plain text format.
//
// Arguments  :  None
//
// Return     :  YT_SUCCESS or YT_FAIL
//-------------------------------------------------------------------------------------------------------
int NewMagicCommand::GetHelpMsgText() {
    SET_TIMER(__PRETTY_FUNCTION__);

    command_undefined_ = false;

    // TODO: make it colored???
    output_.status = "Success";
    if (entry_point_ == EntryPoint::kLibytInteractiveMode) {
        output_.output =
            "Usage: \033[33;1;4m%libyt COMMAND\033[0;37m\n"
            "Commands:\n"
            "  \033[32;1mhelp  \033[1;31m              \033[0;37m              Print help messages.\n"
            "  \033[32;1mexit  \033[1;31m              \033[0;37m              Exit entry point.\n"
            "  \033[32;1mload  \033[1;31m  <file name> \033[0;37m              Load and run Python file <file name>.\n"
            "  \033[32;1mexport\033[1;31m  <file name> \033[0;37m              Export history to <file name>.\n"
            "  \033[32;1mstatus\033[1;31m              \033[0;37m              Get all function status.\n"
            "  \033[32;1mstatus\033[1;31m  <function>  \033[0;37m              Get <function> status.\n"
            "  \033[32;1mrun   \033[1;31m  <function>  \033[0;37m [\033[1;31marg1\033[0;37m, ...]  Run "
            "<function>(args) automatically in next iteration.\n"
            "  \033[32;1midle  \033[1;31m  <function>  \033[0;37m              Make <function> idle in next "
            "iteration.\n";
    } else {
        output_.output = "Usage:  %libyt COMMAND\n"
                         "Commands:\n"
                         "  help                             Print help messages.\n"
                         "  exit                             Exit entry point.\n"
                         "  load    <file name>              Load and run Python file <file name>.\n"
                         "  export  <file name>              Export history to <file name>.\n"
                         "  status                           Get all function status.\n"
                         "  status  <function>               Get <function> status.\n"
                         "  run     <function>  [arg1, ...]  Run <function>(args) automatically in next iteration.\n"
                         "  idle    <function>               Make <function> idle in next iteration.\n";
    }

    return YT_SUCCESS;
}

//-------------------------------------------------------------------------------------------------------
// Class      :  MagicCommand
// Method     :  LoadScript
//
// Notes      :  1. This is a collective call.
//               2. Run the Python file line-by-line, if variable names already exist in inline script's
//                  namespace, it will be overwritten by this file.
//               3. Parse functions in script and add to g_func_status_list. If function name already
//                  exists in the list, the source code in libyt.interactive_mode["func_body"] will
//                  be updated and overwritten.
//               4. Character in the file loaded cannot exceed INT_MAX.
//               5. CAUTION: Currently, only root rank gets the full error msg and output, so broadcast
//                  the results. (TODO: probably should update the data structure in execute_file.)
//
// Arguments  :  const std::vector<std::string>& args : Full magic commands. (ex: %libyt load filename.py)
//
// Return     : YT_SUCCESS or YT_FAIL
//-------------------------------------------------------------------------------------------------------
int NewMagicCommand::LoadScript(const std::vector<std::string>& args) {
    SET_TIMER(__PRETTY_FUNCTION__);

    command_undefined_ = false;

    if (args.size() != 3) {
        output_.status = "Error";
        output_.error = std::string("Usage: %libyt load filename.py\n"
                                    "Description: Load and run Python file filename.py\n");
        return YT_FAIL;
    }

    // Root rank checks and broadcast the script, if worked, everyone calls execute_file
    bool python_run_successfully = false;
    if (g_myrank == root_) {
        // make sure file exist and read the file
        std::ifstream stream;
        stream.open(args[2]);
        if (!stream) {
#ifndef SERIAL_MODE
            int indicator = -1;
            MPI_Bcast(&indicator, 1, MPI_INT, root_, MPI_COMM_WORLD);
#endif
            output_.status = "Error";
            output_.error = std::string("File ") + args[2] + std::string("doesn't exist.\n");
            output_.error += std::string("Loading script '") + args[2] + std::string("' ... failed.\n");
            return YT_FAIL;
        }
        std::string line;
        std::stringstream ss;
        while (getline(stream, line)) {
            ss << line << "\n";
        }
        stream.close();

        // Check code validity
        CodeValidity code_validity = LibytPythonShell::check_code_validity(ss.str(), false, args[2].c_str());
        if (code_validity.is_valid == "complete") {
            // Run file and format output from the results, and check if Python run successfully
#ifndef SERIAL_MODE
            int indicator = 1;
            MPI_Bcast(&indicator, 1, MPI_INT, root_, MPI_COMM_WORLD);
#endif
            std::array<AccumulatedOutputString, 2> output = LibytPythonShell::execute_file(ss.str(), args[2]);
            if (output[1].output_string.empty()) {
                python_run_successfully = true;
            } else {
                python_run_successfully = false;
            }
#ifndef SERIAL_MODE
            MPI_Bcast(&python_run_successfully, 1, MPI_C_BOOL, root_, MPI_COMM_WORLD);
#endif

            for (int i = 0; i < 2; i++) {
                if (!output[i].output_string.empty()) {
                    size_t offset = 0;
                    for (int r = 0; r < g_mysize; r++) {
                        std::string head;
                        if (entry_point_ == kLibytInteractiveMode || entry_point_ == kLibytJupyterKernel) {
                            head += std::string("\033[1;34m[MPI Process ") + std::to_string(r) +
                                    std::string("]\n\033[0;30m");
                        } else {
                            head += std::string("[MPI Process ") + std::to_string(r) + std::string("]\n");
                        }

                        if (output[i].output_length[r] == 0) {
                            head += std::string("(None)\n");
                        }
                        output[i].output_string.insert(offset, head);
                        offset = offset + head.length() + output[i].output_length[r];
                    }
                }
            }
            output_.output = std::move(output[0].output_string);
            output_.error = std::move(output[1].output_string);
        } else {
#ifndef SERIAL_MODE
            int indicator = -1;
            MPI_Bcast(&indicator, 1, MPI_INT, root_, MPI_COMM_WORLD);
#endif
            output_.status = "Error";
            output_.error = code_validity.error_msg + std::string("\n");
            output_.error += std::string("Loading script '") + args[2] + std::string("' ... failed\n");
            return YT_FAIL;
        }
    }
#ifndef SERIAL_MODE
    else {
        // return YT_FAIL if no file found or file cannot compile
        int indicator;
        MPI_Bcast(&indicator, 1, MPI_INT, root_, MPI_COMM_WORLD);

        if (indicator < 0) {
            output_.status = "Error";
            return YT_FAIL;
        } else {
            std::array<AccumulatedOutputString, 2> output = LibytPythonShell::execute_file();
            MPI_Bcast(&python_run_successfully, 1, MPI_C_BOOL, root_, MPI_COMM_WORLD);
        }
    }
#endif

    // Update function body even if Python run failed.
    LibytPythonShell::load_file_func_body(args[2].c_str());

    // Get function list defined inside the script, add the function name to list if it doesn't exist
    // and set to idle
    std::vector<std::string> func_list = LibytPythonShell::get_funcname_defined(args[2].c_str());
    for (int i = 0; i < (int)func_list.size(); i++) {
        g_func_status_list.add_new_func(func_list[i].c_str(), 0);
    }

    // Returned type from file must be text/plain, like other python results
    if (python_run_successfully) {
        output_.status = "Success";
        output_.output += std::string("Loading script '") + args[2] + std::string("' ... done\n");
    } else {
        output_.status = "Error";
        output_.output += std::string("Loading script '") + args[2] + std::string("' ... failed\n");
    }

    return YT_SUCCESS;
}

//-------------------------------------------------------------------------------------------------------
// Class      :  MagicCommand
// Method     :  ExportScript
//
// Notes      :  1. Export input during this step's interactive loop.
//               2. Overwriting existing file.
//               3. Only process root will write to file.
//
// Arguments  :  const std::vector<std::string>& args : Full magic commands. (ex: %libyt export history.txt)
//
// Return     :  YT_SUCCESS or YT_FAIL
//-------------------------------------------------------------------------------------------------------
int NewMagicCommand::ExportScript(const std::vector<std::string>& args) {
    SET_TIMER(__PRETTY_FUNCTION__);

    command_undefined_ = false;

    if (args.size() != 3) {
        output_.status = "Error";
        output_.error = std::string("Usage: %libyt export history.py\n"
                                    "Description: Export history to history.py\n");
        return YT_FAIL;
    }

    if (g_myrank == root_) {
        std::ofstream dump_file;
        dump_file.open(args[2], std::ofstream::trunc);
        dump_file << g_libyt_python_shell.get_prompt_history();
        dump_file.close();
    }

    output_.status = "Success";
    output_.output = std::string("Exporting history to '") + args[2] + std::string("' ... done\n");

    return YT_SUCCESS;
}

//-------------------------------------------------------------------------------------------------------
// Class      :  MagicCommand
// Method     :  SetFunctionRun
//
// Notes      :  1. Make function run in next step iteration.
//
// Arguments  :  const std::vector<std::string>& args : Full magic commands. (ex: %libyt run func a b c)
//
// Return     :  YT_SUCCESS or YT_FAIL
//-------------------------------------------------------------------------------------------------------
int NewMagicCommand::SetFunctionRun(const std::vector<std::string>& args) {
    SET_TIMER(__PRETTY_FUNCTION__);

    command_undefined_ = false;

    if (args.size() < 3) {
        output_.status = "Error";
        output_.error = std::string("Usage: %libyt run function_name [arg1, arg2, ...]\n"
                                    "Description: Run function_name(args) automatically in next iteration.\n");
        return YT_FAIL;
    }

    // Get function index
    int index = g_func_status_list.get_func_index(args[2].c_str());
    if (index == -1) {
        output_.status = "Error";
        output_.error = std::string("Function '") + args[2] + std::string("' not found\n");
        return YT_FAIL;
    }

    // Set function run based on if it has input arguments
    std::string input_args;
    if (args.size() != 3) {
        // Get wrapper """ or ''' and join input args
        bool wrapper_detected = false, unable_to_wrapped = false;
        for (size_t i = 3; i < args.size(); i++) {
            if (!wrapper_detected) {
                if (args[i].find("\"\"\"") != std::string::npos) {
                    wrapper_detected = true;
                    g_func_status_list[index].set_wrapper(false);
                } else if (args[i].find("'''") != std::string::npos) {
                    wrapper_detected = true;
                    g_func_status_list[index].set_wrapper(true);
                }
            } else {
                const char* wrapper = g_func_status_list[index].get_wrapper() ? "\"\"\"" : "'''";
                if (args[i].find(wrapper) != std::string::npos) {
                    unable_to_wrapped = true;
                }
            }

            input_args += args[i] + ",";
        }
        input_args.pop_back();

        if (unable_to_wrapped) {
            output_.status = "Error";
            output_.error += std::string("Unable to wrap input arguments\n");
            output_.error += std::string("Please avoid using both \"\"\" and ''' for triple quotes\n");
            return YT_FAIL;
        }
    }
    g_func_status_list[index].set_run(true);
    g_func_status_list[index].set_args(input_args);

    output_.status = "Success";
    output_.output += std::string("Function '") + args[2] + std::string("' set to run ... done\n");
    output_.output +=
        std::string("Run ") + g_func_status_list[index].get_full_func_name() + std::string(" in next iteration\n");

    return YT_SUCCESS;
}

//-------------------------------------------------------------------------------------------------------
// Class      :  MagicCommand
// Method     :  SetFunctionIdle
//
// Notes      :  1. Make function idle in next step iteration.
//
// Arguments  :  const std::vector<std::string>& args : Full magic commands. (ex: %libyt idle func)
//
// Return     :  YT_SUCCESS or YT_FAIL
//-------------------------------------------------------------------------------------------------------
int NewMagicCommand::SetFunctionIdle(const std::vector<std::string>& args) {
    SET_TIMER(__PRETTY_FUNCTION__);

    command_undefined_ = false;

    if (args.size() != 3) {
        output_.status = "Error";
        output_.error = std::string("Usage: %libyt idle function_name\n"
                                    "Description: Make function_name idle in next iteration.\n");
        return YT_FAIL;
    }

    // Get function index
    int index = g_func_status_list.get_func_index(args[2].c_str());
    if (index == -1) {
        output_.status = "Error";
        output_.error = std::string("Function '") + args[2] + std::string("' not found\n");
        return YT_FAIL;
    }

    std::string input_args;
    g_func_status_list[index].set_run(false);
    g_func_status_list[index].set_args(input_args);

    output_.status = "Success";
    output_.output += std::string("Function '") + args[2] + std::string("' set to idle ... done\n");

    return YT_SUCCESS;
}

//-------------------------------------------------------------------------------------------------------
// Class      :  MagicCommand
// Method     :  GetFunctionStatusMarkdown
//
// Notes      :  1. Get function status and get error msg if it has.
//               2. Fetch libyt.interactive_mode["func_err_msg"], it is a collective call.
//               3. Only store output on root.
//
// Arguments  :  const std::vector<std::string>& args : Full magic commands. (ex: %libyt status func)
//
// Return     :  YT_SUCCESS or YT_FAIL
//-------------------------------------------------------------------------------------------------------
int NewMagicCommand::GetFunctionStatusMarkdown(const std::vector<std::string>& args) {
    SET_TIMER(__PRETTY_FUNCTION__);

    command_undefined_ = false;

    if (args.size() != 3) {
        output_.status = "Error";
        output_.error = std::string("Usage: %libyt status function_name\n"
                                    "Description: Get function_name status and information.\n");
        return YT_FAIL;
    }

    // Get function index
    int index = g_func_status_list.get_func_index(args[2].c_str());
    if (index == -1) {
        output_.status = "Error";
        output_.error = std::string("Function '") + args[2] + std::string("' not found\n");
        return YT_FAIL;
    }

    // Get function status and error msg and format it in markdown
    output_.mimetype = "text/markdown";
    output_.status = "Success";

    int status = g_func_status_list[index].get_status();
    if (g_myrank == root_) {
        output_.output += std::string("#### `") + args[2] + std::string("`\n");

        // Execute status
        output_.output += std::string("- **Execute status in previous call:** ");
        if (status == 1) {
            output_.output += std::string("_Success_\n");
        } else if (status == 0) {
            output_.output += std::string("_Failed_\n");
        } else if (status == -1) {
            output_.output += std::string("_Idle_\n");
        }

        // Function call in next iteration
        output_.output += std::string("- **Function call in next iteration:** ");
        if (g_func_status_list[index].get_run() == 1) {
            output_.output += std::string("`") + g_func_status_list[index].get_full_func_name() + std::string("`\n");
        } else {
            output_.output += std::string("(None)\n");
        }

        // Function definition
        output_.output += std::string("- **Current function definition:**\n");
        output_.output += std::string("  ```python\n");

        std::string func_body = g_func_status_list[index].get_func_body();
        std::size_t start_pos = 0, found;
        while (true) {
            found = func_body.find('\n', start_pos);
            if (found != std::string::npos) {
                output_.output +=
                    std::string("  ") + func_body.substr(start_pos, found - start_pos) + std::string("\n");
            } else {
                output_.output += std::string("  ") + func_body.substr(start_pos) + std::string("\n");
                break;
            }
            start_pos = found + 1;
        }
        output_.output += std::string("  ```\n");
    }

    // Call getting error message if it has (status == 0), this is a collective call
    if (status == 0) {
        std::vector<std::string> output_error = g_func_status_list[index].get_error_msg();
        if (g_myrank == root_) {
            output_.output += std::string("- **Error message from previous call:**\n");
            for (size_t r = 0; r < output_error.size(); r++) {
                if (output_error[r].empty()) continue;

                output_.output +=
                    std::string("<details><summary>MPI Process ") + std::to_string(r) + (" </summary><p>");

                std::size_t start_pos = 0, found;
                while (true) {
                    found = output_error[r].find('\n', start_pos);
                    if (found != std::string::npos) {
                        output_.output += output_error[r].substr(start_pos, found - start_pos) + std::string("<br>");
                    } else {
                        output_.output += output_error[r].substr(start_pos);
                        break;
                    }
                    start_pos = found + 1;
                }

                output_.output += std::string("</p></details>");
            }
        }
    }

    return YT_SUCCESS;
}

//-------------------------------------------------------------------------------------------------------
// Class      :  MagicCommand
// Method     :  GetFunctionStatusText
//
// Notes      :  1. Get function status and print error msg if it has.
//               2. libyt.interactive_mode["func_err_msg"] only stores function's error msg when using
//                  yt_run_Function/yt_run_FunctionArguments.
//               3. Fetch libyt.interactive_mode["func_err_msg"], it is a collective call.
//               4. Only store output on root.
//
// Arguments  :  const std::vector<std::string>& args : Full magic commands. (ex: %libyt status func)
//
// Return     :  YT_SUCCESS or YT_FAIL
//-------------------------------------------------------------------------------------------------------
int NewMagicCommand::GetFunctionStatusText(const std::vector<std::string>& args) {
    SET_TIMER(__PRETTY_FUNCTION__);

    command_undefined_ = false;

    if (args.size() != 3) {
        output_.status = "Error";
        output_.error = std::string("Usage: %libyt status function_name\n"
                                    "Description: Get function_name status and information.\n");
        return YT_FAIL;
    }

    // Get function index
    int index = g_func_status_list.get_func_index(args[2].c_str());
    if (index == -1) {
        output_.status = "Error";
        output_.error = std::string("Function '") + args[2] + std::string("' not found\n");
        return YT_FAIL;
    }

    // Get function status and error msg and format it in plain text
    output_.status = "Success";

    int status = g_func_status_list[index].get_status();
    if (g_myrank == root_) {
        // Get status
        output_.output += g_func_status_list[index].get_func_name() + std::string(" ... ");
        if (status == 1) {
            output_.output += std::string("success\n");
        } else if (status == 0) {
            output_.output += std::string("failed\n");
        } else if (status == -1) {
            output_.output += std::string("idle\n");
        }

        // Get function definition
        if (entry_point_ == kLibytInteractiveMode) {
            output_.output += std::string("\033[1;35m[Function Def]\033[0;37m\n");
        } else {
            output_.output += std::string("[Function Def]\n");
        }
        std::string func_body = g_func_status_list[index].get_func_body();
        std::size_t start_pos = 0, found;
        while (true) {
            found = func_body.find('\n', start_pos);
            if (found != std::string::npos) {
                output_.output +=
                    std::string("  ") + func_body.substr(start_pos, found - start_pos) + std::string("\n");
            } else {
                output_.output += std::string("  ") + func_body.substr(start_pos) + std::string("\n");
                break;
            }
            start_pos = found + 1;
        }
    }

    // Get error msg if it failed when running in yt_run_Function/yt_run_FunctionArguments. (collective call)
    if (status == 0) {
        std::vector<std::string> output_error = g_func_status_list[index].get_error_msg();
        if (g_myrank == root_) {
            if (entry_point_ == kLibytInteractiveMode) {
                output_.output += std::string("\033[1;35m[Error Msg]\033[0;37m\n");
            } else {
                output_.output += std::string("[Error Msg]\n");
            }
            for (size_t r = 0; r < output_error.size(); r++) {
#ifndef SERIAL_MODE
                if (entry_point_ == kLibytInteractiveMode) {
                    output_.output +=
                        std::string("\033[1;36m  [ MPI process ") + std::to_string(r) + (" ]\033[0;37m\n");
                } else {
                    output_.output += std::string("  [ MPI process ") + std::to_string(r) + (" ]\n");
                }
#endif
                if (output_error[r].empty()) {
                    output_.output += std::string("    (none)\n");
                    continue;
                }

                std::size_t start_pos = 0, found;
                while (true) {
                    found = output_error[r].find('\n', start_pos);
                    if (found != std::string::npos) {
                        output_.output += std::string("    ");
                        output_.output += output_error[r].substr(start_pos, found - start_pos) + std::string("\n");
                    } else {
                        output_.output += output_error[r].substr(start_pos);
                        break;
                    }
                    start_pos = found + 1;
                }
            }
        }
    }

    return YT_SUCCESS;
}

#endif  // #if defined(INTERACTIVE_MODE) || defined(JUPYTER_KERNEL)
