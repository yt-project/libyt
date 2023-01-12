#ifndef __FUNC_STATUS_H__
#define __FUNC_STATUS_H__

#include "yt_macro.h"

class func_status
{
private:
    char       *m_FuncName;
    int         m_Run;
    int         m_Status;

public:
    func_status(char *func_name, int run);
    ~func_status();
    func_status(const func_status& other);

    char* get_func_name() {return m_FuncName;};

    int  get_run() const { return m_Run; };
    int   set_run(int run) { m_Run = run; return YT_SUCCESS; };

    int get_status();
    int set_status(int status) { m_Status = status; return YT_SUCCESS; };

    int serial_print_error(int indent_size, int indent_level);
    int print_func_body(int indent_size, int indent_level);
};
#endif // #ifndef __FUNC_STATUS_H__
