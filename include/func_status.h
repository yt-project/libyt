#ifndef __FUNC_STATUS_H__
#define __FUNC_STATUS_H__

#include "yt_macro.h"

class func_status
{
private:
    char       *m_FuncName;
    bool        m_Run;
    int         m_Status;

public:
    func_status(char *func_name);
    func_status(char *func_name, char *code);
    ~func_status();
    func_status(const func_status& other);

    char* get_func_name() {return m_FuncName;};

    bool  get_run() const { return m_Run; };
    int   set_run(bool run) { m_Run = run; return YT_SUCCESS; };

    int get_status();
    int set_status(int status) { m_Status = status; return YT_SUCCESS; };

    int print_error();
    int update_func_body();
    int clear_func_body();
};
#endif // #ifndef __FUNC_STATUS_H__
