#if defined(INTERACTIVE_MODE) || defined(JUPYTER_KERNEL)

#include "func_status_list.h"

#include <iostream>

#include "yt_combo.h"

//-------------------------------------------------------------------------------------------------------
// Class       :  func_status_list
// Method      :  reset
//
// Notes       :  1. Reset every func_status data member m_Status in list to -1 (not run yet).
//                2. Clear the error buffer.
//
// Arguments   :  None
//
// Return      : YT_SUCCESS
//-------------------------------------------------------------------------------------------------------
int func_status_list::reset() {
    SET_TIMER(__PRETTY_FUNCTION__);

    for (int i = 0; i < size(); i++) {
        m_FuncStatusList[i].set_status(-1);
        m_FuncStatusList[i].clear_error_msg();
    }
    return YT_SUCCESS;
}

//-------------------------------------------------------------------------------------------------------
// Class       :  func_status_list
// Method      :  print_summary
//
// Notes       :  1. Print function status and run status in func_status_list.
//                2. Only root rank prints.
//                3. normal      -> bold white
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
    SET_TIMER(__PRETTY_FUNCTION__);

    // make sure every rank has reach here, so that printing in other ranks are done
    fflush(stdout);
    fflush(stderr);

    if (g_myrank == g_myroot) {
        printf("\033[1;37m");
        printf("=====================================================================\n");
        printf("  %-40s     %-12s   %s\n", "Inline Function", "Status", "Run");
        printf("---------------------------------------------------------------------\n");
        for (int i = 0; i < size(); i++) {
            printf("\033[1;37m");  // change to bold white
            printf("  * %-43s", m_FuncStatusList[i].get_func_name());
            int run = m_FuncStatusList[i].get_run();
            int status = m_FuncStatusList[i].get_status();

            if (status == 0) {
                printf("\033[1;31m");  // bold red: failed
                printf("%-12s", "failed");
            } else if (status == 1) {
                printf("\033[1;32m");  // bold green: success
                printf("%-12s", "success");
            } else if (status == -1) {
                printf("\033[1;34m");  // bold blue: idle
                printf("%-12s", "idle");
            } else {
                printf("\033[0;37m");  // change to white
                printf("%-12s (%d)", "unknown status", status);
            }

            printf("\033[1;33m");  // bold yellow
            if (run == 1)
                printf("%5s\n", "V");
            else
                printf("%5s\n", "X");

            fflush(stdout);
        }
        printf("\033[1;37m");
        printf("=====================================================================\n");
        printf("\033[0;37m");  // change to white
    }

    return YT_SUCCESS;
}

//-------------------------------------------------------------------------------------------------------
// Class       :  func_status_list
// Method      :  print_summary_to_file
//
// Notes       :  1. Instead of printing the summary string like print_summary, it prints it to file.
//
// Arguments   :  const std::string& file_name : write summary to file
//
// Return      :  YT_SUCCESS / YT_FAIL
//-------------------------------------------------------------------------------------------------------
bool func_status_list::print_summary_to_file(const std::string& file_name) {
    SET_TIMER(__PRETTY_FUNCTION__);

    if (g_myrank == g_myroot) {
        // open file to write
        FILE* output_file;
        output_file = fopen(file_name.c_str(), "a");
        if (output_file == nullptr) {
            log_error("Unable to write summary to file '%s'.\n", file_name.c_str());
            return YT_FAIL;
        }

        fprintf(output_file, "=====================================================================\n");
        fprintf(output_file, "  %-40s     %-12s   %s\n", "Inline Function", "Status", "Run");
        fprintf(output_file, "---------------------------------------------------------------------\n");
        for (int i = 0; i < size(); i++) {
            fprintf(output_file, "  * %-43s", m_FuncStatusList[i].get_func_name());
            int run = m_FuncStatusList[i].get_run();
            int status = m_FuncStatusList[i].get_status();

            if (status == 0) {
                fprintf(output_file, "%-12s", "failed");
            } else if (status == 1) {
                fprintf(output_file, "%-12s", "success");
            } else if (status == -1) {
                fprintf(output_file, "%-12s", "idle");
            } else {
                fprintf(output_file, "%-12s (%d)", "unknown status", status);
            }

            if (run == 1)
                fprintf(output_file, "%5s\n", "V");
            else
                fprintf(output_file, "%5s\n", "X");
        }
        fprintf(output_file, "=====================================================================\n");

        // close file
        fclose(output_file);
    }

    return YT_SUCCESS;
}

//-------------------------------------------------------------------------------------------------------
// Class       :  func_status_list
// Method      :  get_summary_html
//
// Notes       :  1. Get function status and run status in func_status_list, returned as html format.
//                2. Return in html format.
//                3. normal      -> bold white
//                   idle        -> bold blue
//                   not run yet -> bold yellow
//                   success     -> bold green
//                   failed      -> bold red
//                   MPI process -> bold cyan
//
// Arguments   :  None
//
// Return      :  std::string output_html : summary in html format
//-------------------------------------------------------------------------------------------------------
std::string func_status_list::get_summary_html() {
    SET_TIMER(__PRETTY_FUNCTION__);

    // Font style
    std::string base_style = std::string("font-weight:bold;font-family:'arial'");
    std::string success_cell =
        std::string("<td><span style=\"color: #28B463;") + base_style + std::string("\">success</span></td>");
    std::string failed_cell =
        std::string("<td><span style=\"color: #E74C3C;") + base_style + std::string("\">failed</span></td>");
    std::string idle_cell =
        std::string("<td><span style=\"color: #2874A6;") + base_style + std::string("\">idle</span></td>");
    std::string unknown_cell =
        std::string("<td><span style=\"color: #A569BD;") + base_style + std::string("\">unknown</span></td>");
    std::string will_run_cell =
        std::string("<td><span style=\"color: #F1C40F;") + base_style + std::string("\">V</span></td>");
    std::string will_idle_cell =
        std::string("<td><span style=\"color: #F1C40F;") + base_style + std::string("\">X</span></td>");

    // Create table header, Inline Function, Status, Run
    std::string output_html =
        std::string("<table style=\"width: 100%\"><tr><th>Inline Function</th><th>Status</th><th>Run</th></tr>");

    // Loop through each function
    for (int i = 0; i < size(); i++) {
        // Initialize row
        output_html.append("<tr>");

        // Get function name
        output_html.append("<td style=\"text-alight: left;\">");
        output_html.append("<span style=\"font-family:'Courier New'\">");
        output_html.append(m_FuncStatusList[i].get_func_name());
        output_html.append("</span>");
        output_html.append("</td>");

        // Get status
        int status = m_FuncStatusList[i].get_status();
        switch (status) {
            case 0: {
                output_html.append(failed_cell);
                break;
            }
            case 1: {
                output_html.append(success_cell);
                break;
            }
            case -1: {
                output_html.append(idle_cell);
                break;
            }
            default: {
                output_html.append(unknown_cell);
                break;
            }
        }

        int run = m_FuncStatusList[i].get_run();
        if (run == 1) {
            output_html.append(will_run_cell);
        } else {
            output_html.append(will_idle_cell);
        }

        // Close row
        output_html.append("</tr>");
    }

    // Close the table
    output_html.append("</table>");

    return output_html;
}

//-------------------------------------------------------------------------------------------------------
// Class       :  func_status_list
// Method      :  get_func_index
//
// Notes       :  1. Look up index of func_name in m_FuncStatusList.
//
// Arguments   :  char   *func_name: inline function name
//
// Return      :  index : index of func_name in list, return -1 if it doesn't exist.
//-------------------------------------------------------------------------------------------------------
int func_status_list::get_func_index(const char* func_name) {
    SET_TIMER(__PRETTY_FUNCTION__);

    int index = -1;
    for (int i = 0; i < size(); i++) {
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
int func_status_list::add_new_func(const char* func_name, int run) {
    SET_TIMER(__PRETTY_FUNCTION__);

    // Check if func_name exist, return YT_SUCCESS if exist
    int index = get_func_index(func_name);
    if (index >= 0) return index;

    // add func_name, since it adds to the end, its index is equal to original size
    index = size();
    m_FuncStatusList.emplace_back(func_name, run);

    return index;
}

//-------------------------------------------------------------------------------------------------------
// Class       :  func_status_list
// Method      :  run_func
//
// Notes       :  1. This is a collective call. It executes new added functions that haven't run by
//                   yt_run_Function/yt_run_FunctionArguments yet.
//                2. How this method runs python function is identical to yt_run_Function*. It use
//                   PyRun_SimpleString.
//                3. libyt uses either """ or ''' to wrap the code to execute in exec(). It finds if the
//                   arguments are using triple quotes, if yes, it chooses using """ or '''.
//                4. When setting arguments in %libyt run func args, libyt will make sure user are only
//                   using either """ or ''', if they are using triple quotes.
//                5. Get input arguments from func_status.m_Args if it has.
//
// Arguments   :  None
//
// Return      :  YT_SUCCESS
//-------------------------------------------------------------------------------------------------------
int func_status_list::run_func() {
    SET_TIMER(__PRETTY_FUNCTION__);

    for (int i = 0; i < size(); i++) {
        int run = m_FuncStatusList[i].get_run();
        int status = m_FuncStatusList[i].get_status();
        if (run == 1 && status == -1) {
            // command
            const char* funcname = m_FuncStatusList[i].get_func_name();
            int command_width = 350 + strlen(g_param_libyt.script) + strlen(funcname) * 2;
            char* command = (char*)malloc(command_width * sizeof(char));
            const char* wrapped = m_FuncStatusList[i].get_wrapper() ? "\"\"\"" : "'''";
            sprintf(command,
                    "try:\n"
                    "    exec(%s%s%s, sys.modules[\"%s\"].__dict__)\n"
                    "except Exception as e:\n"
                    "    libyt.interactive_mode[\"func_err_msg\"][\"%s\"] = traceback.format_exc()\n",
                    wrapped, m_FuncStatusList[i].get_full_func_name().c_str(), wrapped, g_param_libyt.script, funcname);

            // run and update status
            log_info("Performing YT inline analysis %s ...\n", m_FuncStatusList[i].get_full_func_name().c_str());
            m_FuncStatusList[i].set_status(-2);
            if (PyRun_SimpleString(command) != 0) {
                m_FuncStatusList[i].set_status(0);
                free(command);
                YT_ABORT("Unexpected error occurred while executing %s in script's namespace.\n",
                         m_FuncStatusList[i].get_full_func_name().c_str());
            }
            m_FuncStatusList[i].get_status();
            log_info("Performing YT inline analysis %s ... done\n", funcname,
                     m_FuncStatusList[i].get_full_func_name().c_str());

            // clean up
            free(command);
        }
    }
    return YT_SUCCESS;
}

#endif  // #if defined(INTERACTIVE_MODE) || defined(JUPYTER_KERNEL)
