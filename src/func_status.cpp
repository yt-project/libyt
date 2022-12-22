#ifdef INTERACTIVE_MODE

#include "func_status.h"
#include <string.h>


//-------------------------------------------------------------------------------------------------------
// Class       :  func_status
// Method      :  Constructor
//
// Notes       :  1. Initialize m_FuncName, m_Run, m_Status:
//                   (1) m_FuncName: function name, does not include argument
//                   (2)      m_Run: true  -> will run in next iteration (default)
//                                   false -> will idle in next iteration
//                   (3)   m_Status:     1 -> success
//                                       0 -> failed
//                                      -1 -> not execute by yt_inline/yt_inline_argument yet (default)
//
// Arguments   :  char    *func_name: inline function name
//-------------------------------------------------------------------------------------------------------
func_status::func_status(char *func_name)
: m_Run(true), m_Status(-1)
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

#endif // #ifdef INTERACTIVE_MODE
