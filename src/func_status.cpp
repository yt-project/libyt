#ifdef INTERACTIVE_MODE

#include "func_status.h"
#include "yt_combo.h"
#include <iostream>
#include <string>
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
// Notes       :  1. If m_Status = -2, which is running, then this is a collective call. After checking
//                   status at local, it will get other ranks status.
//                   Else, it will just return current m_Status.
//                2. Check if key exist in libyt.interactive_mode["func_err_msg"] dict, it means failed if
//                   func name in keys.
//                3. Return directly if this function is not running, which is m_Status != -2.
//                4. Definition of m_Status:
//                   1  -> success
//                   0  -> failed
//                   -1 -> not execute by yt_inline/yt_inline_argument yet (default)
//                   -2 -> running
//
// Arguments   :  None
// Return      :  m_Status
//-------------------------------------------------------------------------------------------------------
int func_status::get_status() {
    // if it is not running (-2), we don't need to check if there is error msg.
    if (m_Status != -2) return m_Status;

    // check if key exist in libyt.interactive_mode["func_err_msg"] dict, which is to get local status
    PyObject *py_func_name = PyUnicode_FromString(m_FuncName);
    if (PyDict_Contains(PyDict_GetItemString(g_py_interactive_mode, "func_err_msg"), py_func_name) == 1) m_Status = 0;
    else m_Status = 1;
    Py_DECREF(py_func_name);

    // mpi reduce, if sum(m_Status) != g_mysize, then some rank must have failed. Now m_Status is global status
    int tot_status = 0;
    MPI_Allreduce(&m_Status, &tot_status, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    if (tot_status != g_mysize) m_Status = 0;
    else m_Status = 1;

    return m_Status;
}


//-------------------------------------------------------------------------------------------------------
// Class       :  func_status
// Method      :  serial_print_error
//
// Notes       :  1. This is a collective call. Must call by every rank.
//                2. When it is this MPI rank's turn to print, invoke Python call to print error buffer
//                   in libyt.interactive_mode["func_err_msg"]. If this buffer is empty, print nothing.
//                3. Assert that every err msg line ends with newline \n.
//
// Arguments   :  int indent_size : indent size
//                int indent_level: how many times to indent
// Return      :  YT_SUCCESS or YT_FAIL
//-------------------------------------------------------------------------------------------------------
int func_status::serial_print_error(int indent_size, int indent_level) {
    // get my err msg at current rank
    PyObject *py_func_name = PyUnicode_FromString(m_FuncName);
    PyObject *py_err_msg = PyDict_GetItem(PyDict_GetItemString(g_py_interactive_mode, "func_err_msg"), py_func_name);
    const char *err_msg;
    if (py_err_msg != NULL) err_msg = PyUnicode_AsUTF8(py_err_msg);
    else                    err_msg = "";
    Py_DECREF(py_func_name);

    // serial print
    int root = 0;
    if (g_myrank == root) {
        for (int rank=0; rank<g_mysize; rank++) {
            printf("\033[1;36m");                               // set to bold cyan
            printf("%*s", indent_size * indent_level, "");      // indent
            printf("[ MPI %d ]\n", rank);
            printf("\033[0;37m");                               // set to white

            // get and print error msg, convert to string
            std::string *str_ptr;
            if (rank == g_myrank) {
                str_ptr = new std::string(err_msg);
            }
            else {
                int tag = rank;
                int err_msg_len;
                MPI_Recv(&err_msg_len, 1, MPI_INT, rank, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                char *err_msg_remote = (char*) malloc((err_msg_len + 1) * sizeof(char));
                MPI_Recv(err_msg_remote, err_msg_len, MPI_CHAR, rank, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                err_msg_remote[err_msg_len] = '\0';

                str_ptr = new std::string(err_msg_remote);
                free(err_msg_remote);
            }

            // indent
            std::size_t start_pos = 0;
            std::size_t found;
            while (true) {
                found = (*str_ptr).find("\n", start_pos);
                if (found != std::string::npos) {
                    printf("%*s", indent_size * indent_level, "");
                    printf("print range : [%ld, %ld), %s\n", start_pos, found, (*str_ptr).substr(start_pos, found).c_str());
                }
                else {
                    break;
                }
                start_pos = found + 1;
            }

            // clean up
            fflush(stdout);
            delete str_ptr;
        }
    }
    else {
        int tag = g_myrank;
        int err_msg_len = strlen(err_msg);
        MPI_Ssend(&err_msg_len, 1, MPI_INT, root, tag, MPI_COMM_WORLD);
        MPI_Ssend(err_msg, err_msg_len, MPI_CHAR, root, tag, MPI_COMM_WORLD);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    return YT_SUCCESS;
}

#endif // #ifdef INTERACTIVE_MODE
