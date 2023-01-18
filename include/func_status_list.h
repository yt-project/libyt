#ifndef __LIBYT_FUNC_STATUS_LIST_H__
#define __LIBYT_FUNC_STATUS_LIST_H__

#include <vector>
#include <array>
#include <string>
#include "func_status.h"
#include <Python.h>

class func_status_list {
public:
    static const int s_NotDone_Num = 2;
    static std::array<std::string, s_NotDone_Num> s_NotDone_ErrMsg;
    static std::array<PyObject*, s_NotDone_Num>   s_NotDone_PyErr;

private:
    std::vector<func_status> m_FuncStatusList;

public:
    func_status_list(int capacity) { m_FuncStatusList.reserve(capacity); };
    ~func_status_list() { m_FuncStatusList.clear(); };
    func_status& operator[](int index) { return m_FuncStatusList[index]; };

    int reset();
    int size() { return (int) m_FuncStatusList.size(); };
    int print_summary();
    int get_func_index(const char *func_name);
    int add_new_func(const char *func_name, int run);
    int run_func();

    static int load_file_func_body(const char *filename);
    static int load_input_func_body(char *code);
    static std::vector<std::string> get_funcname_defined(const char *filename);
    static int set_exception_hook();
    static int init_not_done_err_msg();
    static bool is_not_done_err_msg();
};


#endif // #ifndef __LIBYT_FUNC_STATUS_LIST_H__
