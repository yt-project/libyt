#ifndef __LIBYT_PYTHON_SHELL_H__
#define __LIBYT_PYTHON_SHELL_H__

#include <Python.h>

#include <array>
#include <string>
#include <vector>

enum class PythonStatus { kPythonUnknown = -1, kPythonFailed = 0, kPythonSuccess = 1 };

struct PythonOutput {
    PythonStatus status;
    std::string output;
    std::string error;
};

struct CodeValidity {
    std::string is_valid;
    std::string error_msg;
};

class LibytPythonShell {
private:
    static std::vector<std::string> s_Bracket_NotDoneErr;
    static std::vector<std::string> s_CompoundKeyword_NotDoneErr;

    std::string m_PromptHistory;
    int m_PromptHistoryCount;

    static PyObject* s_PyGlobals;
    static PyObject* function_body_dict_;

    static int mpi_size_;
    static int mpi_root_;
    static int mpi_rank_;

    PythonStatus AllExecute(int python_input_type, const std::string& code, const std::string& cell_base_name,
                            int src_rank, std::vector<PythonOutput>& output, int output_mpi_rank);
    static long GetLastStatementLineno(const std::string& code);

public:
    LibytPythonShell() : m_PromptHistory(""), m_PromptHistoryCount(0){};
    static void SetMPIInfo(const int mpi_size, const int mpi_root, const int mpi_rank) {
        mpi_size_ = mpi_size;
        mpi_root_ = mpi_root;
        mpi_rank_ = mpi_rank;
    }
    int update_prompt_history(const std::string& cmd_prompt);
    int clear_prompt_history();
    std::string& get_prompt_history() { return m_PromptHistory; };

    static int load_file_func_body(const char* filename);
    static int load_input_func_body(const char* code);
    static std::vector<std::string> get_funcname_defined(const char* filename);
    static int SetExceptionHook();
    static int InitializeNotDoneErrMsg();
    static int SetExecutionNamespace(PyObject* execution_namespace);
    static int SetFunctionBodyDict(PyObject* function_body_dict);
    static PyObject* GetExecutionNamespace() { return s_PyGlobals; }
    static PyObject* GetFunctionBodyDict() { return function_body_dict_; }
    static bool IsNotDoneErrMsg(const std::string& code);
    static CodeValidity CheckCodeValidity(const std::string& code, bool prompt_env = false,
                                          const char* cell_name = "<libyt-stdin>");
    PythonStatus AllExecutePrompt(const std::string& code, const std::string& cell_base_name, int src_rank,
                                  std::vector<PythonOutput>& output, int output_mpi_rank);
    PythonStatus AllExecuteFile(const std::string& code, const std::string& cell_base_name, int src_rank,
                                std::vector<PythonOutput>& output, int output_mpi_rank);
    PythonStatus AllExecuteCell(const std::string& code, const std::string& cell_base_name, int src_rank,
                                std::vector<PythonOutput>& output, int output_mpi_rank);
};

#endif  // __LIBYT_PYTHON_SHELL_H__
