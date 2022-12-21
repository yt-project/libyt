#ifndef __FUNC_STATUS_H__
#define __FUNC_STATUS_H__

#include "yt_macro.h"
#include <string.h>
#include <string>

class func_status
{
private:
    char       *m_FuncName;
    std::string m_FuncBody;
    bool        m_Run;
    int         m_Status;

public:
    func_status(char *func_name);
    ~func_status();

    int update_func_body();
    int clear_func_body();
    int get_run(bool *run) { *run = m_Run; return YT_SUCCESS; };
    int set_run(bool run) { m_Run = run; return YT_SUCCESS; };
    int get_status(int *status) { *status = m_Status; return YT_SUCCESS; };
    int set_status(int status) { m_Status = status; return YT_SUCCESS; };
    int print_error();
};
#endif // #ifndef __FUNC_STATUS_H__
