#if defined(INTERACTIVE_MODE) && defined(JUPYTER_KERNEL)
#include "libyt_kernel.h"

#include <Python.h>

#include <iostream>
#include <string>

#include "xeus/xhelper.hpp"
#include "yt_combo.h"

struct CodeValidity {
    bool is_valid;
    std::string error_msg;
};

static std::vector<std::string> split(const std::string& code, const char* c);
static std::array<std::string, 2> split_on_line(const std::string& code, unsigned int lineno);
static CodeValidity code_is_valid(const std::string& code);

//-------------------------------------------------------------------------------------------------------
// Class       :  LibytKernel
// Method      :  configure_impl
// Description :  Configure kernel before it runs anything
//
// Notes       :  1. Import io, sys. We need io.StringIO to get the output and error.
//                2. Get script's global python object.
//                3. TODO: initialize jedi auto-completion
//
// Arguments   :  (None)
//
// Return      :  (None)
//-------------------------------------------------------------------------------------------------------
void LibytKernel::configure_impl() {
    SET_TIMER(__PRETTY_FUNCTION__);

    PyRun_SimpleString("import io, sys\n");

    m_py_global = PyDict_GetItemString(g_py_interactive_mode, "script_globals");

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
//                2. TODO: Support Parallel
//                2. TODO: Support libyt defined commands.
//                3. TODO: Supprt Display
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

    // Make sure code is valid before continue
    CodeValidity code_validity = code_is_valid(code);
    if (!code_validity.is_valid) {
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

    // Append newline at the front of the last statement, so that Python error buffer can catch the correct lineno
    code_split[1].insert(0, std::string('\n', last_statement_lineno - 1));

    Py_DECREF(py_module_ast);
    Py_DECREF(py_ast_parse);
    Py_DECREF(py_result);
    Py_DECREF(py_result_body);

    // Clear the template buffer and redirect stdout, stderr
    PyRun_SimpleString("sys.OUTPUT_STDOUT=''\nstdout_buf=io.StringIO()\nsys.stdout=stdout_buf\n");
    PyRun_SimpleString("sys.OUTPUT_STDERR=''\nstderr_buf=io.StringIO()\nsys.stderr=stderr_buf\n");

    // Compile and execute code in script's namespace
    std::string cell_name = std::string("In [") + std::to_string(execution_counter) + std::string("]");

    // Execute upper half and lower half in serial, if error occurred, it will skip execute the lower half
    PyObject* py_src;
    bool has_error = false;
    for (int i = 0; i < 2; i++) {
        if (code_split[i].length() <= 0) continue;

        // Compile code
        if (i == 0) {
            py_src = Py_CompileString(code_split[i].c_str(), cell_name.c_str(), Py_file_input);
        } else if (i == 1) {
            py_src = Py_CompileString(code_split[i].c_str(), cell_name.c_str(), Py_single_input);
        }

        // Evaluate code
        if (py_src != NULL) {
            PyEval_EvalCode(py_src, m_py_global, m_py_global);
            if (PyErr_Occurred()) {
                has_error = true;
                PyErr_Print();
                Py_DECREF(py_src);
                break;
            }
            Py_DECREF(py_src);
        } else {
            has_error = true;
            PyErr_Print();
            break;
        }
    }

    // Collect stdout_buf/stderr_buf and store under sys.OUTPUT_STDOUT and sys.OUTPUT_STDERR
    PyRun_SimpleString("sys.stdout.flush()\n");
    PyRun_SimpleString("sys.stderr.flush()\n");
    PyRun_SimpleString("sys.OUTPUT_STDOUT=stdout_buf.getvalue()\nstdout_buf.close()\n");
    PyRun_SimpleString("sys.OUTPUT_STDERR=stderr_buf.getvalue()\nstderr_buf.close()\n");
    PyRun_SimpleString("sys.stdout=sys.__stdout__\n");
    PyRun_SimpleString("sys.stderr=sys.__stderr__\n");
    PyErr_Clear();

    // Parse the string
    PyObject* py_module_sys = PyImport_ImportModule("sys");
    PyObject* py_stdout_buf = PyObject_GetAttrString(py_module_sys, "OUTPUT_STDOUT");
    PyObject* py_stderr_buf = PyObject_GetAttrString(py_module_sys, "OUTPUT_STDERR");

    std::string output_stdout, output_stderr;
    output_stdout = std::string(PyUnicode_AsUTF8(py_stdout_buf));
    if (has_error) {
        output_stderr = std::string(PyUnicode_AsUTF8(py_stderr_buf));
    }

    Py_DECREF(py_module_sys);
    Py_DECREF(py_stdout_buf);
    Py_DECREF(py_stderr_buf);

    // Publish results
    nl::json pub_data;
    if (output_stdout.length() > 0) {
        pub_data["text/plain"] = output_stdout.c_str();
        publish_execution_result(execution_counter, std::move(pub_data), nl::json::object());
    }
    if (has_error) {
        publish_execution_error("", "", split(output_stderr, "\n"));
    }

    return xeus::create_successful_reply();
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
// Method      :  code_is_valid
// Description :  Check code validity.
//
// Notes       :  1. Test if it can compile based on Py_file_input.
//                2. I separated this function because code passed in can have multi-statement, and we
//                   want the last statement to use Py_single_input, which is different from here.
//
// Arguments   :  const std::string& code  : raw code
//
// Return      :  CodeValidity.is_valid : true if is valid, false if it failed to compile
//                             error_msg: error message from Python if it failed.
//-------------------------------------------------------------------------------------------------------
static CodeValidity code_is_valid(const std::string& code) {
    SET_TIMER(__PRETTY_FUNCTION__);

    PyObject* py_test_compile = Py_CompileString(code.c_str(), "<test-validity>", Py_file_input);

    if (py_test_compile != NULL) {
        Py_DECREF(py_test_compile);
        return {true, std::string("")};
    } else {
        // Get the error message
        PyRun_SimpleString("sys.OUTPUT_STDERR=''\nstderr_buf=io.StringIO()\nsys.stderr=stderr_buf\n");
        PyErr_Print();
        PyRun_SimpleString("sys.stderr.flush()\n");
        PyRun_SimpleString("sys.OUTPUT_STDERR=stderr_buf.getvalue()\nstderr_buf.close()\n");
        PyRun_SimpleString("sys.stderr=sys.__stderr__\n");
        PyObject* py_module_sys = PyImport_ImportModule("sys");
        PyObject* py_stderr_buf = PyObject_GetAttrString(py_module_sys, "OUTPUT_STDERR");
        CodeValidity code_validity = {false, std::string(PyUnicode_AsUTF8(py_stderr_buf))};

        // Clear buffer and dereference
        PyErr_Clear();
        Py_XDECREF(py_test_compile);
        Py_DECREF(py_module_sys);
        Py_DECREF(py_stderr_buf);

        return code_validity;
    }
}

#endif
