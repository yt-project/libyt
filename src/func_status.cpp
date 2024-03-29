#if defined(INTERACTIVE_MODE) || defined(JUPYTER_KERNEL)

#include "func_status.h"

#include <string>

#include "yt_combo.h"

//-------------------------------------------------------------------------------------------------------
// Class       :  func_status
// Method      :  Constructor
//
// Notes       :  1. Initialize m_FuncName, m_Run, m_Status:
//                   (1) m_FuncName: function name, does not include argument
//                   (2)     m_Args: input arguments for new added functions. libyt uses m_Args when using
//                                   func_status_list::run_func to run the functions.
//                   (3)  m_Wrapper: wrapped with """ or ''' when executing python function inside exec.
//                                   this is only used when libyt runs functions not called by yt_run_Function*
//                                   yet in interactive mode.
//                                    true -> wrapped with """ (default)
//                                   false -> wrapped with '''
//                   (3)      m_Run:     1 -> will run in next iteration
//                                       0 -> will idle in next iteration
//                                      -1 -> not set yet
//                   (4)   m_Status:     1 -> success
//                                       0 -> failed
//                                      -1 -> not execute by yt_run_Function/yt_run_FunctionArguments yet (default)
//                                      -2 -> running
//                2. Separate storing actual python source code in libyt.interactive_mode["func_body"].
//
// Arguments   :  const char *func_name: inline function name
//                int         run      : will run in next iteration or not
//-------------------------------------------------------------------------------------------------------
func_status::func_status(const char* func_name, int run) : m_Args(""), m_Wrapper(true), m_Run(run), m_Status(-1) {
    SET_TIMER(__PRETTY_FUNCTION__);

    // copy func_name to m_FuncName
    int len = strlen(func_name);
    m_FuncName = new char[len + 1];
    strcpy(m_FuncName, func_name);
}

//-------------------------------------------------------------------------------------------------------
// Class       :  func_status
// Method      :  Copy Constructor
//
// Notes       :  1. It is inefficient to do it this way, but we are adding func_status class to
//                   g_func_status_list vector, which makes a copy.
//                   Although we can replace it to store class's pointer, I don't want to access through
//                   arrow.
//
// Arguments   :  const func_status& other
//-------------------------------------------------------------------------------------------------------
func_status::func_status(const func_status& other)
    : m_Args(other.m_Args),
      m_Wrapper(other.m_Wrapper),
      m_Run(other.m_Run),
      m_Status(-1),
      m_FullErrorMsg(std::move(other.m_FullErrorMsg)) {
    SET_TIMER(__PRETTY_FUNCTION__);

    // copy m_FuncName;
    int len = strlen(other.m_FuncName);
    m_FuncName = new char[len + 1];
    strcpy(m_FuncName, other.m_FuncName);
}

func_status::~func_status() { delete[] m_FuncName; }

//-------------------------------------------------------------------------------------------------------
// Class       :  func_status
// Method      :  get_full_func_name
//
// Notes       :  1. Return the string of how Python call this function, including arguments.
//
// Arguments   :  None
//
// Return      :  std::string function_call : how python will call this function, including args.
//-------------------------------------------------------------------------------------------------------
std::string func_status::get_full_func_name() {
    SET_TIMER(__PRETTY_FUNCTION__);

    std::string function_call = std::string(m_FuncName);
    function_call += std::string("(") + m_Args + std::string(")");

    return function_call;
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
//                   -1 -> not execute by yt_run_Function/yt_run_FunctionArguments yet (default)
//                   -2 -> running
//
// Arguments   :  None
// Return      :  m_Status
//-------------------------------------------------------------------------------------------------------
int func_status::get_status() {
    SET_TIMER(__PRETTY_FUNCTION__);

    // if it is not running (-2), we don't need to check if there is error msg.
    if (m_Status != -2) return m_Status;

    // check if key exist in libyt.interactive_mode["func_err_msg"] dict, which is to get local status
    PyObject* py_func_name = PyUnicode_FromString(m_FuncName);
    if (PyDict_Contains(PyDict_GetItemString(g_py_interactive_mode, "func_err_msg"), py_func_name) == 1) {
        m_Status = 0;
    } else {
        m_Status = 1;
    }
    Py_DECREF(py_func_name);

#ifndef SERIAL_MODE
    // mpi reduce, if sum(m_Status) != g_mysize, then some rank must have failed. Now m_Status is global status
    int tot_status = 0;
    MPI_Allreduce(&m_Status, &tot_status, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    if (tot_status != g_mysize) {
        m_Status = 0;
    } else {
        m_Status = 1;
    }
#endif

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
//
// Return      :  YT_SUCCESS or YT_FAIL
//-------------------------------------------------------------------------------------------------------
int func_status::serial_print_error(int indent_size, int indent_level) {
    SET_TIMER(__PRETTY_FUNCTION__);

    // get my err msg at current rank
    PyObject* py_func_name = PyUnicode_FromString(m_FuncName);
    PyObject* py_err_msg = PyDict_GetItem(PyDict_GetItemString(g_py_interactive_mode, "func_err_msg"), py_func_name);
    const char* err_msg;
    if (py_err_msg != NULL)
        err_msg = PyUnicode_AsUTF8(py_err_msg);
    else
        err_msg = "";
    Py_DECREF(py_func_name);

    // serial print
    int root = 0;
    if (g_myrank == root) {
        for (int rank = 0; rank < g_mysize; rank++) {
            printf("\033[1;36m");                           // set to bold cyan
            printf("%*s", indent_size * indent_level, "");  // indent
#ifndef SERIAL_MODE
            printf("[ MPI %d ]\n", rank);
#else
            printf("[ Process %d ]\n", rank);
#endif
            printf("\033[0;37m");  // set to white

            // get and print error msg, convert to string
            std::string str_err_msg;
            if (rank == g_myrank) {
                str_err_msg = std::string(err_msg);
            }
#ifndef SERIAL_MODE
            else {
                int tag = rank;
                int err_msg_len;
                MPI_Recv(&err_msg_len, 1, MPI_INT, rank, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                char* err_msg_remote = (char*)malloc((err_msg_len + 1) * sizeof(char));
                MPI_Recv(err_msg_remote, err_msg_len, MPI_CHAR, rank, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                err_msg_remote[err_msg_len] = '\0';

                str_err_msg = std::string(err_msg_remote);
                free(err_msg_remote);
            }
#endif

            // print out error msg with indent
            std::size_t start_pos = 0;
            std::size_t found;
            if (str_err_msg.length() == 0) {
                printf("%*s", indent_size * (indent_level + 1), "");
                printf("(none)\n");
            }
            while (str_err_msg.length() > 0) {
                found = str_err_msg.find("\n", start_pos);
                if (found != std::string::npos) {
                    printf("%*s", indent_size * (indent_level + 1), "");
                    for (std::size_t c = start_pos; c < found; c++) {
                        printf("%c", str_err_msg[c]);
                    }
                    printf("\n");
                } else {
                    break;
                }
                start_pos = found + 1;
            }

            // clean up
            fflush(stdout);
        }
    }
#ifndef SERIAL_MODE
    else {
        int tag = g_myrank;
        int err_msg_len = strlen(err_msg);
        MPI_Ssend(&err_msg_len, 1, MPI_INT, root, tag, MPI_COMM_WORLD);
        MPI_Ssend(err_msg, err_msg_len, MPI_CHAR, root, tag, MPI_COMM_WORLD);
    }

    MPI_Barrier(MPI_COMM_WORLD);
#endif

    return YT_SUCCESS;
}

//-------------------------------------------------------------------------------------------------------
// Class       :  func_status
// Method      :  get_error_msg
//
// Notes       :  1. This is a collective call. Must call by every rank.
//                2. Unlike serial_print_error, this stores the buffer, so it uses MPI_Gather/MPI_Gatherv
//                   to gather string.
//                3. Assert that every err msg line ends with newline \n.
//                4. Cache error message, every rank will cache it.
//
// Arguments   :  (None)
//
// Return      :  YT_SUCCESS or YT_FAIL
//-------------------------------------------------------------------------------------------------------
std::vector<std::string>& func_status::get_error_msg() {
    SET_TIMER(__PRETTY_FUNCTION__);

    if (!m_FullErrorMsg.empty()) {
        return m_FullErrorMsg;
    }

    // get my err msg at each process
    PyObject* py_func_name = PyUnicode_FromString(m_FuncName);
    PyObject* py_err_msg = PyDict_GetItem(PyDict_GetItemString(g_py_interactive_mode, "func_err_msg"), py_func_name);
    const char* err_msg;
    if (py_err_msg != NULL) {
        err_msg = PyUnicode_AsUTF8(py_err_msg);
    } else {
        err_msg = "";
    }
    Py_DECREF(py_func_name);

#ifndef SERIAL_MODE
    // all gather error, each rank will have full error buffer
    int error_len = strlen(err_msg);
    int* all_error_len = new int[g_mysize];
    MPI_Allgather(&error_len, 1, MPI_INT, all_error_len, 1, MPI_INT, MPI_COMM_WORLD);

    long sum_output_len = 0;
    int* displace = new int[g_mysize];
    for (int r = 0; r < g_mysize; r++) {
        displace[r] = 0;
        sum_output_len += all_error_len[r];
        for (int r1 = 0; r1 < r; r1++) {
            displace[r] += all_error_len[r1];
        }
    }
    char* all_error = new char[sum_output_len + 1];
    MPI_Allgatherv(err_msg, all_error_len[g_myrank], MPI_CHAR, all_error, all_error_len, displace, MPI_CHAR,
                   MPI_COMM_WORLD);

    for (int r = 0; r < g_mysize; r++) {
        m_FullErrorMsg.emplace_back(std::string(all_error).substr(displace[r], all_error_len[r]));
    }

    delete[] all_error_len;
    delete[] displace;
    delete[] all_error;
#else
    m_FullErrorMsg.emplace_back(std::string(err_msg));
#endif

    return m_FullErrorMsg;
}

//-------------------------------------------------------------------------------------------------------
// Class       :  func_status
// Method      :  clear_error_msg
//
// Notes       :  1. Must call by every rank. Since this cache is empty, then get_error_msg will gather
//                   errors from other ranks.
//                2. Cache error message, every root rank will cache it, so every rank must call it.
//
// Arguments   :  (None)
//
// Return      :  YT_SUCCESS or YT_FAIL
//-------------------------------------------------------------------------------------------------------
int func_status::clear_error_msg() {
    SET_TIMER(__PRETTY_FUNCTION__);

    m_FullErrorMsg.clear();

    return YT_SUCCESS;
}

//-------------------------------------------------------------------------------------------------------
// Class       :  func_status
// Method      :  print_func_body
//
// Notes       :  1. Only root will print funtion body.
//
// Arguments   :  int indent_size : indent size
//                int indent_level: how many times to indent
//
// Return      :  YT_SUCCESS or YT_FAIL
//-------------------------------------------------------------------------------------------------------
int func_status::print_func_body(int indent_size, int indent_level) {
    SET_TIMER(__PRETTY_FUNCTION__);

    int root = 0;
    if (g_myrank == root) {
        std::string func_body = get_func_body();

        // print function body with indent
        printf("\033[0;37m");
        if (func_body.length() <= 0) {
            printf("%*s", indent_size * (indent_level + 1), "");
            printf("(not defined)\n");
        } else {
            std::size_t start_pos = 0;
            std::size_t found;
            while (func_body.length() > 0) {
                found = func_body.find("\n", start_pos);
                if (found != std::string::npos) {
                    printf("%*s", indent_size * (indent_level + 1), "");
                    for (std::size_t c = start_pos; c < found; c++) {
                        printf("%c", func_body[c]);
                    }
                    printf("\n");
                } else
                    break;
                start_pos = found + 1;
            }
        }

        fflush(stdout);
    }

    return YT_SUCCESS;
}

//-------------------------------------------------------------------------------------------------------
// Class       :  func_status
// Method      :  get_func_body
//
// Notes       :  1. Get function body by querying libyt.interactive_mode["func_body"][m_FuncName]
//
// Arguments   :  (None)
//
// Return      :  std::string func_body : function body
//-------------------------------------------------------------------------------------------------------
std::string func_status::get_func_body() {
    SET_TIMER(__PRETTY_FUNCTION__);

    // get function body
    PyObject* py_func_name = PyUnicode_FromString(m_FuncName);
    PyObject* py_func_body = PyDict_GetItem(PyDict_GetItemString(g_py_interactive_mode, "func_body"), py_func_name);
    std::string func_body("");
    if (py_func_body != NULL) {
        func_body = std::string(PyUnicode_AsUTF8(py_func_body));
    }
    Py_DECREF(py_func_name);

    return func_body;
}

#endif  // #if defined(INTERACTIVE_MODE) || defined(JUPYTER_KERNEL)
