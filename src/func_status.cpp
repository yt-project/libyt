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
    m_FuncName = new char [len + 1];
    strcpy(m_FuncName, func_name);
}


//-------------------------------------------------------------------------------------------------------
// Class       :  func_status
// Method      :  Copy Constructor
//
// Notes       :  1. It is unefficient to do it this way, but we are adding func_status class to
//                   g_func_status_list vector, which makes a copy.
//                   Although we can replace it to store class's pointer, I don't want to access through
//                   arrow.
//
// Arguments   :  const func_status& other
//-------------------------------------------------------------------------------------------------------
func_status::func_status(const func_status& other)
: m_Run(true), m_Status(-1)
{
    // copy m_FuncName;
    int len = strlen(other.m_FuncName);
    m_FuncName = new char [len + 1];
    strcpy(m_FuncName, other.m_FuncName);
}


func_status::~func_status()
{
    delete [] m_FuncName;
}

#endif // #ifdef INTERACTIVE_MODE
