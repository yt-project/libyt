#ifndef __LIBYT_FUNC_STATUS_LIST_H__
#define __LIBYT_FUNC_STATUS_LIST_H__

#include <Python.h>

#include <array>
#include <string>
#include <vector>

#include "func_status.h"

class func_status_list {
public:
    static const int s_NotDone_Num = 3;
    static std::array<std::string, s_NotDone_Num> s_NotDone_ErrMsg;
    static std::array<PyObject*, s_NotDone_Num> s_NotDone_PyErr;

private:
    std::vector<func_status> m_FuncStatusList;
    std::string m_PromptHistory;
    int m_PromptHistoryCount;

public:
    func_status_list(int capacity) : m_PromptHistory(""), m_PromptHistoryCount(0) {
        m_FuncStatusList.reserve(capacity);
    };
    ~func_status_list() { m_FuncStatusList.clear(); };
    func_status& operator[](int index) { return m_FuncStatusList[index]; };

    int reset();
    int size() { return (int)m_FuncStatusList.size(); };
    int print_summary();
    int get_func_index(const char* func_name);
    int add_new_func(const char* func_name, int run);
    int run_func();
    int update_prompt_history(const std::string& cmd_prompt);
    int clear_prompt_history();
    std::string& get_prompt_history() { return m_PromptHistory; };

    static int load_file_func_body(const char* filename);
    static int load_input_func_body(const char* code);
    static std::vector<std::string> get_funcname_defined(const char* filename);
    static int set_exception_hook();
    static int init_not_done_err_msg();
    static bool is_not_done_err_msg(const char* code);
};

#endif  // #ifndef __LIBYT_FUNC_STATUS_LIST_H__
