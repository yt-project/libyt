#ifndef __LIBYT_FUNC_STATUS_LIST_H__
#define __LIBYT_FUNC_STATUS_LIST_H__

#include <vector>
#include "func_status.h"

class func_status_list {
private:
    std::vector<func_status> m_FuncStatusList;

public:
    func_status_list(int capacity) { m_FuncStatusList.reserve(capacity); };
    ~func_status_list() { m_FuncStatusList.clear(); };
    func_status& operator[](int index) {return m_FuncStatusList[index]; };

    int reset();
    int print_summary();
    int get_func_index(char *func_name);
    int add_new_func(char *func_name);
};


#endif // #ifndef __LIBYT_FUNC_STATUS_LIST_H__
