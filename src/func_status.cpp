#ifdef INTERACTIVE_MODE

#include "func_status.h"
#include "yt_combo.h"
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
//                                      -2 -> running
//                2. Get function body using inspect.getsource store in libyt.interactive_mode["func_body"][<func_name>].
//                   If inspect.getsource gets error, store null string "" instead.
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

    // get function body using inspect.getsource
    int command_width = 200 + len * 3 + strlen(g_param_libyt.script);
    char *command = (char*) malloc(command_width * sizeof(char));
    sprintf(command, "try:\n"
                     "    libyt.interactive_mode[\"func_body\"][\"%s\"] = inspect.getsource(%s.%s)\n"
                     "except:\n"
                     "    libyt.interactive_mode[\"func_body\"][\"%s\"] = \"\"\n",
                     func_name, g_param_libyt.script, func_name, func_name);
    if (PyRun_SimpleString(command) == 0) log_debug("Loading inline function body %s ... done\n", func_name);
    else                                  log_debug("Loading inline function body %s ... failed\n", func_name);
    free(command);
}


func_status::func_status(char *func_name, char *code)
: m_Run(true), m_Status(-1)
{
    // copy func_name to m_FuncName
    int len = strlen(func_name);
    m_FuncName = new char [len + 1];
    strcpy(m_FuncName, func_name);

    // todo: convert code to python string and store under dict.
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


//-------------------------------------------------------------------------------------------------------
// Class       :  func_status
// Method      :  get_status
//
// Notes       :  1. Check if key exist in libyt.interactive_mode["func_err_msg"] dict, it means failed if
//                   func name in keys.
//                2. Return directly if this function is not running, which is m_Status != -2.
//
// Arguments   :  None
// Return      :  m_Status
//-------------------------------------------------------------------------------------------------------
int func_status::get_status() {
    // if it is not running (-2), we don't need to check if there is error msg.
    if (m_Status != -2) return m_Status;

    // check if key exist in libyt.interactive_mode["func_err_msg"] dict
    PyObject *py_func_name = PyUnicode_FromString(m_FuncName);
    if (PyDict_Contains(PyDict_GetItemString(g_py_interactive_mode, "func_err_msg"), py_func_name)) m_Status = 0;
    else m_Status = 1;
    Py_DECREF(py_func_name);

    return m_Status;
}

#endif // #ifdef INTERACTIVE_MODE
