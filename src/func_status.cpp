#include "func_status.h"

func_status::func_status(char *func_name)
: m_Run(1), m_Status(0)
{
    // copy func_name to m_FuncName
    int len = strlen(func_name);
    m_FuncName = new char [len+1];
    strcpy(m_FuncName, func_name);
}


func_status::~func_status()
{
    delete [] m_FuncName;
}




