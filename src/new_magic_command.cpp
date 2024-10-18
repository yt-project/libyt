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
//
// Arguments  :  None
//
// Return     :  YT_SUCCESS or YT_FAIL
//-------------------------------------------------------------------------------------------------------
int NewMagicCommand::GetStatusHtml() {
    SET_TIMER(__PRETTY_FUNCTION__);

    command_undefined_ = false;

    // TODO: move html part to this method and use GetSummaryMethod.
    output_.status = "Success";
    output_.mimetype = "text/html";
    output_.output = std::move(g_func_status_list.get_summary_html());

    return YT_SUCCESS;
}

//-------------------------------------------------------------------------------------------------------
// Class      :  MagicCommand
// Method     :  GetStatusText
//
// Notes      :  1. Get all the function status, without error msg.
//
// Arguments  :  None
//
// Return     :  YT_SUCCESS or YT_FAIL
//-------------------------------------------------------------------------------------------------------
int NewMagicCommand::GetStatusText() {
    SET_TIMER(__PRETTY_FUNCTION__);

    command_undefined_ = false;

    output_.status = "Success";

    // TODO: (EXTRA CARE)
    // TODO: single out to GetSummary in function_status_list
    // TODO: single out a colored text static function
    bool status = false;
    if (entry_point_ == kLibytInteractiveMode) {
        status = g_func_status_list.print_summary();
    } else if (entry_point_ == kLibytReloadScript) {
        // TODO: this is for temporary use, will print the summary in yt_run_ReloadScript
        status = g_func_status_list.print_summary_to_file("DEBUG-libyt_summary.txt");
    } else {
        return YT_FAIL;
    }

    return status;
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
    output_.output =
        std::string("Usage:  %%libyt COMMAND\n"
                    "Commands:\n"
                    "  help                             Print help messages.\n"
                    "  exit                             Exit entry point.\n"
                    "  load    <file name>              Load and run Python file <file name>.\n"
                    "  export  <file name>              Export history to <file name>.\n"
                    "  status                           Get all function status.\n"
                    "  status  <function>               Get <function> status.\n"
                    "  run     <function>  [arg1, ...]  Run <function>(args) automatically in next iteration.\n"
                    "  idle    <function>               Make <function> idle in next iteration.\n");

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

    // TODO: START HERE

    return 0;
}
int NewMagicCommand::GetFunctionStatusText(const std::vector<std::string>& args) { return 0; }

// TODO: also implement coloring text functionality

#endif  // #if defined(INTERACTIVE_MODE) || defined(JUPYTER_KERNEL)
