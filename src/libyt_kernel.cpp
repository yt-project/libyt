#ifdef JUPYTER_KERNEL
#include "libyt_kernel.h"

#include <string>
#include <xeus/xhelper.hpp>

#include "libyt.h"
#include "libyt_python_shell.h"
#include "magic_command.h"
#include "yt_combo.h"

static std::vector<std::string> split(const std::string& code, const char* c);
static std::array<std::string, 2> split_on_line(const std::string& code, unsigned int lineno);
static std::array<int, 2> find_lineno_columno(const std::string& code, int pos);

//-------------------------------------------------------------------------------------------------------
// Class       :  LibytKernel
// Method      :  configure_impl
// Description :  Configure kernel before it runs anything
//
// Notes       :  1. Import io, sys. We need io.StringIO to get the output and error.
//                2. Initialize jedi auto-completion.
//
// Arguments   :  (None)
//
// Return      :  (None)
//-------------------------------------------------------------------------------------------------------
void LibytKernel::configure_impl() {
    SET_TIMER(__PRETTY_FUNCTION__);

    PyRun_SimpleString("import io, sys\n");

    PyObject* py_module_jedi = PyImport_ImportModule("jedi");
    if (py_module_jedi == NULL) {
        log_info(
            "Unable to import jedi, jedi auto-completion library is disabled (See https://jedi.readthedocs.io/)\n");
        m_py_jedi_interpreter = NULL;
    } else {
        m_py_jedi_interpreter = PyObject_GetAttrString(py_module_jedi, "Interpreter");
        Py_DECREF(py_module_jedi);
    }

    log_info("libyt kernel: configure the kernel ... done\n");
}

//-------------------------------------------------------------------------------------------------------
// Class       :  LibytKernel
// Method      :  execute_request_impl
// Description :  Execute the code.
//
// Notes       :  1. Stream output to io.StringIO when running codes.
//                2. Code will not be empty, it must contain characters other than newline or space.
//                3. Always return "xeus::create_successful_reply()", though I don't know what it is for.
//                4. Run the last statement using Py_single_input, so that it displays value.
//                5. libyt defined commands and magic commands (though currently not supported yet), should
//                   start with %, and should be single line.
//
// Arguments   :  int   execution_counter  : cell number
//                const std::string& code  : raw code, will not be empty
//                bool  silent             : (not used yet)
//                bool  store_history      : (not used yet)
//                nl::json user_expressions: (not used yet)
//                bool  allow_stdin        : (not used yet)
//
// Return      :  nl::json
//-------------------------------------------------------------------------------------------------------
nl::json LibytKernel::execute_request_impl(int execution_counter, const std::string& code, bool silent,
                                           bool store_history, nl::json user_expressions, bool allow_stdin) {
    SET_TIMER(__PRETTY_FUNCTION__);

    std::string cell_name = std::string("In [") + std::to_string(execution_counter) + std::string("]");

    // Find if '%' is the first non-space character, if so, redirect jobs to define command
    std::size_t found = code.find_first_not_of("\t\n\v\f\r ");
    if (found != std::string::npos && code.at(found) == '%') {
        // call magic command
#ifndef SERIAL_MODE
        int indicator = 2;
        MPI_Bcast(&indicator, 1, MPI_INT, g_myroot, MPI_COMM_WORLD);
#endif
        MagicCommand command;
        OutputData command_output = command.run(code.substr(found, code.length() - found));

        // publish result and error
        if (command_output.output.length() > 0) {
            nlohmann::json pub_data;
            pub_data[command_output.mimetype.c_str()] = command_output.output.c_str();
            publish_execution_result(execution_counter, std::move(pub_data), nl::json::object());
        }
        if (command_output.error.length() > 0) {
            publish_execution_error("LibytMagicCommandError", "", split(command_output.error, "\n"));
        }

        return xeus::create_successful_reply();
    }

    // Make sure code is valid before continue
    CodeValidity code_validity = LibytPythonShell::check_code_validity(code, false, cell_name.c_str());
    if (code_validity.is_valid.compare("complete") != 0) {
        publish_execution_error("", "", split(code_validity.error_msg, "\n"));
        return xeus::create_successful_reply();
    }

    // Parse the code using ast, and separate the last statement
    PyObject* py_module_ast = PyImport_ImportModule("ast");
    PyObject* py_ast_parse = PyObject_GetAttrString(py_module_ast, "parse");
    PyObject* py_result = PyObject_CallFunction(py_ast_parse, "s", code.c_str());
    PyObject* py_result_body = PyObject_GetAttrString(py_result, "body");

    // index guard, though this is unlikely happen
    Py_ssize_t num_statements = PyList_Size(py_result_body);
    if (num_statements <= 0) {
        return xeus::create_successful_reply();
    }
    long last_statement_lineno =
        PyLong_AsLong(PyObject_GetAttrString(PyList_GET_ITEM(py_result_body, num_statements - 1), "lineno"));
    std::array<std::string, 2> code_split = split_on_line(code, last_statement_lineno - 1);

    // Append newline at the end of the last statement, so that Python won't produce EOF error
    if (code_split[1].length() > 0) {
        code_split[1].append("\n");
    }

    // Append newline at the front of the last statement, so that Python error buffer can catch the correct lineno
    if (last_statement_lineno >= 1) {
        code_split[1].insert(0, std::string(last_statement_lineno - 1, '\n'));
    }

    Py_DECREF(py_module_ast);
    Py_DECREF(py_ast_parse);
    Py_DECREF(py_result);
    Py_DECREF(py_result_body);

    // Call execute cell, and concatenate the string
    // TODO: It is a bad practice to send execute signal msg to other ranks like this, should wrap in function.
#ifndef SERIAL_MODE
    int indicator = 1;
    MPI_Bcast(&indicator, 1, MPI_INT, g_myroot, MPI_COMM_WORLD);
#endif
    std::array<AccumulatedOutputString, 2> output = LibytPythonShell::execute_cell(code_split, cell_name);

    // Insert header to string
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

    // Publish results
    if (output[0].output_string.length() > 0) {
        nl::json pub_data;
        pub_data["text/plain"] = output[0].output_string.c_str();
        publish_execution_result(execution_counter, std::move(pub_data), nl::json::object());
    }
    if (output[1].output_string.length() > 0) {
        publish_execution_error("", "", split(output[1].output_string, "\n"));
    }

    return xeus::create_successful_reply();
}

//-------------------------------------------------------------------------------------------------------
// Class       :  LibytKernel
// Method      :  complete_request_impl
// Description :  Respond to <TAB> and create a list for auto-completion
//
// Notes       :  1. Route process to call jedi using Python C API:
//                   script = jedi.Interpreter(code, [namespaces])
//                   script.complete(lineno, columno)
//
// Arguments   :  const std::string &code : code to inspect for auto-completion
//                int          cursor_pos : cursor position in the code to inspect
//
// Return      :  nl::json
//-------------------------------------------------------------------------------------------------------
nl::json LibytKernel::complete_request_impl(const std::string& code, int cursor_pos) {
    SET_TIMER(__PRETTY_FUNCTION__);

    // Check if jedi has successfully import
    if (m_py_jedi_interpreter == NULL) {
        log_info(
            "Unable to import jedi, jedi auto-completion library is disabled (See https://jedi.readthedocs.io/)\n");
        return xeus::create_complete_reply({}, cursor_pos, cursor_pos);
    }

    PyObject* py_tuple_args = PyTuple_New(2);
    PyObject* py_list_scope = PyList_New(1);
    PyList_SET_ITEM(py_list_scope, 0, LibytPythonShell::get_script_namespace());  // steal ref
    Py_INCREF(LibytPythonShell::get_script_namespace());
    PyTuple_SET_ITEM(py_tuple_args, 0, Py_BuildValue("s", code.c_str()));  // steal ref
    PyTuple_SET_ITEM(py_tuple_args, 1, py_list_scope);                     // steal ref
    PyObject* py_script = PyObject_CallObject(m_py_jedi_interpreter, py_tuple_args);
    Py_DECREF(py_tuple_args);
    Py_XDECREF(py_list_scope);

    // find lineno and columno of the cursor position, and call script.complete(lineno, columno)
    std::array<int, 2> pos_no = find_lineno_columno(code, cursor_pos);
    PyObject* py_script_complete_callable = PyObject_GetAttrString(py_script, "complete");
    py_tuple_args = PyTuple_New(2);
    PyTuple_SET_ITEM(py_tuple_args, 0, PyLong_FromLong(pos_no[0]));
    PyTuple_SET_ITEM(py_tuple_args, 1, PyLong_FromLong(pos_no[1]));
    PyObject* py_complete_list = PyObject_CallObject(py_script_complete_callable, py_tuple_args);
    Py_DECREF(py_tuple_args);
    Py_DECREF(py_script_complete_callable);
    Py_DECREF(py_script);

    nl::json complete_list;
    for (Py_ssize_t i = 0; i < PyList_Size(py_complete_list); i++) {
        PyObject* py_name = PyObject_GetAttrString(PyList_GET_ITEM(py_complete_list, i), "name");
        complete_list.emplace_back(PyUnicode_AsUTF8(py_name));
        Py_DECREF(py_name);
    }

    int cursor_start = cursor_pos;
    if (PyList_Size(py_complete_list) > 0) {
        PyObject* py_name = PyObject_GetAttrString(PyList_GET_ITEM(py_complete_list, 0), "name");
        PyObject* py_complete = PyObject_GetAttrString(PyList_GET_ITEM(py_complete_list, 0), "complete");
        cursor_start = cursor_pos - ((int)(PyUnicode_GET_LENGTH(py_name) - PyUnicode_GET_LENGTH(py_complete)));
        Py_DECREF(py_name);
        Py_DECREF(py_complete);
    }
    Py_DECREF(py_complete_list);

    // publish result
    if (complete_list.size()) {
        return xeus::create_complete_reply(complete_list, cursor_start, cursor_pos);
    } else {
        return xeus::create_complete_reply({}, cursor_pos, cursor_pos);
    }
}

//-------------------------------------------------------------------------------------------------------
// Class       :  LibytKernel
// Method      :  inspect_request_impl
// Description :  Respond to ? inspection
//
// Notes       :  1. This function is currently idled, because I don't know why Xeus does not route
//                   inspections to this api when <python object>? occurred.
//
// Arguments   :  const std::string &code : code to inspect
//                int          cursor_pos : cursor position in the code where inspection is requested
//                int        detail_level : The level of detail desired. 0 for x?, 1 for x??
//
// Return      :  nl::json
//-------------------------------------------------------------------------------------------------------
nl::json LibytKernel::inspect_request_impl(const std::string& code, int cursor_pos, int detail_level) {
    SET_TIMER(__PRETTY_FUNCTION__);

    log_info("Code inspection is not supported yet\n");

    return xeus::create_inspect_reply();
}

//-------------------------------------------------------------------------------------------------------
// Class       :  LibytKernel
// Method      :  is_complete_request_impl
// Description :  Check if the code is complete in Jupyter console
//
// Notes       :  1. This request is never called from the Notebook or from JupyterLab clients,
//                   but it is called from the Jupyter console client.
//                2. Though this method is implemented, I still cannot access it through jupyter console,
//                   so this method is idled.
//
// Arguments   :  const std::string &code : code to check if it is complete
//
// Return      :  nl::json
//-------------------------------------------------------------------------------------------------------
nl::json LibytKernel::is_complete_request_impl(const std::string& code) {
    SET_TIMER(__PRETTY_FUNCTION__);

    CodeValidity code_validity = LibytPythonShell::check_code_validity(code, true);
    if (code_validity.is_valid.compare("complete") == 0) {
        return xeus::create_is_complete_reply("complete");
    } else {
        return xeus::create_is_complete_reply("incomplete");
    }
}

//-------------------------------------------------------------------------------------------------------
// Class       :  LibytKernel
// Method      :  kernel_info_request_impl
// Description :  Get libyt kernel information
//
// Notes       :  1. It needs PY_VERSION (defined in Python header).
//                2. Check https://jupyter-client.readthedocs.io/en/stable/messaging.html#kernel-info
//                3. Not sure what 'protocol_version' is for.
//                4. TODO: When there is a specific class for controlling libyt info, update helper links.
//
// Arguments   :  (None)
//
// Return      :  nl::json
//-------------------------------------------------------------------------------------------------------
nl::json LibytKernel::kernel_info_request_impl() {
    SET_TIMER(__PRETTY_FUNCTION__);

    nl::json libyt_kernel_info;

    // kernel implementation
    char libyt_version[20];
    sprintf(libyt_version, "%d.%d.%d", LIBYT_MAJOR_VERSION, LIBYT_MINOR_VERSION, LIBYT_MICRO_VERSION);
    libyt_kernel_info["implementation"] = "libyt_kernel";
    libyt_kernel_info["implementation_version"] = libyt_version;

    // protocol
    libyt_kernel_info["protocol_version"] = xeus::get_protocol_version().c_str();

    // language info
    libyt_kernel_info["language_info"]["name"] = "python";
    libyt_kernel_info["language_info"]["version"] = PY_VERSION;
    libyt_kernel_info["language_info"]["mimetype"] = "text/x-python";
    libyt_kernel_info["language_info"]["file_extension"] = ".py";

    // debugger, not supported
    libyt_kernel_info["debugger"] = false;

    // helper
    libyt_kernel_info["help_links"] = nl::json::array();
    libyt_kernel_info["help_links"][0] =
        nl::json::object({{"text", "libyt Kernel Documents"}, {"url", "https://libyt.readthedocs.io/en/latest/"}});

    // status
    libyt_kernel_info["status"] = "ok";

    return libyt_kernel_info;
}

//-------------------------------------------------------------------------------------------------------
// Class       :  LibytKernel
// Method      :  shutdown_request_impl
// Description :  Shutdown libyt kernel
//
// Notes       :  1. Dereference jedi interpreter python function.
//                2. TODO: It is a bad practice to send shutdown msg to other ranks, should wrap in function.
//
// Arguments   :  (None)
//
// Return      :  (None)
//-------------------------------------------------------------------------------------------------------
void LibytKernel::shutdown_request_impl() {
    SET_TIMER(__PRETTY_FUNCTION__);

#ifndef SERIAL_MODE
    int indicator = -1;
    MPI_Bcast(&indicator, 1, MPI_INT, g_myrank, MPI_COMM_WORLD);
#endif

    if (m_py_jedi_interpreter != NULL) {
        Py_DECREF(m_py_jedi_interpreter);
    }

    g_libyt_python_shell.clear_prompt_history();

    log_info("Shutting down libyt kernel ...\n");
}

//-------------------------------------------------------------------------------------------------------
// Method      :  split
// Description :  Split the string based on character
//
// Notes       :  1. It's a local method.
//                2. Find character c to split, and always include the line after last found character c,
//                   which is:
//                   "a\nb\nc\n" -> "a", "b", "c", ""
//
// Arguments   :  const std::string& code  : raw code
//                const char*        c     : split the string using character c
//
// Return      :  std::vector<std::string>
//-------------------------------------------------------------------------------------------------------
static std::vector<std::string> split(const std::string& code, const char* c) {
    SET_TIMER(__PRETTY_FUNCTION__);

    std::vector<std::string> code_split;
    std::size_t start_pos = 0, found;
    while (code.length() > 0) {
        found = code.find(c, start_pos);
        if (found != std::string::npos) {
            code_split.emplace_back(code.substr(start_pos, found - start_pos));
        } else {
            code_split.emplace_back(code.substr(start_pos, code.length() - start_pos));
            break;
        }
        start_pos = found + 1;
    }
    return code_split;
}

//-------------------------------------------------------------------------------------------------------
// Method      :  split_on_line
// Description :  Split the string to two parts on lineno.
//
// Notes       :  1. It's a local method.
//                2. Line count starts at 1.
//                3. code_split[0] contains line 1 ~ lineno, code_split[1] contains the rest.
//
// Arguments   :  const std::string& code  : raw code
//                unsigned int     lineno  : split on lineno, code_split[0] includes lineno
//
// Return      :  std::array<std::string, 2> code_split[0] : code from line 1 ~ lineno
//                                           code_split[1] : the rest of the code
//-------------------------------------------------------------------------------------------------------
static std::array<std::string, 2> split_on_line(const std::string& code, unsigned int lineno) {
    SET_TIMER(__PRETTY_FUNCTION__);

    std::array<std::string, 2> code_split = {std::string(""), std::string("")};
    std::size_t start_pos = 0, found;
    unsigned int line = 1;
    while (code.length() > 0) {
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
    return code_split;
}

//-------------------------------------------------------------------------------------------------------
// Method      :  find_lineno_columno
// Description :  Convert cursor position to lineno and columno, count starts from 1.
//
// Notes       :  1. Cursor position, lineno, and columno count start from 1.
//                2. Cursor position indicates how many characters are there from its position to the
//                   beginning of the string.
//
// Arguments   :  const std::string& code : raw code, contains newline characters.
//                int                 pos : cursor position
//
// Return      : std::array<int, 2> no[0] : lineno
//                                  no[1] : columno
//-------------------------------------------------------------------------------------------------------
static std::array<int, 2> find_lineno_columno(const std::string& code, int pos) {
    SET_TIMER(__PRETTY_FUNCTION__);

    if (code.length() < pos) return {-1, -1};

    int acc = pos;  // pos count start at 1.
    int lineno = 1, columno = 0;

    std::size_t start_pos = 0, found;
    while (code.length() > 0) {
        found = code.find('\n', start_pos);
        if (found != std::string::npos) {
            int len_in_line = found - start_pos;  // exclude "\n"
            if (acc - len_in_line <= 0) {
                columno = acc;
                break;
            } else {
                acc = acc - len_in_line - 1;  // remember to include "\n"
            }
        } else {
            columno = acc;
            break;
        }
        start_pos = found + 1;
        lineno += 1;
    }

    return {lineno, columno};
}

#endif  // #ifdef JUPYTER_KERNEL
