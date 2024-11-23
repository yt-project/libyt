#if defined(INTERACTIVE_MODE) || defined(JUPYTER_KERNEL)

#ifdef USE_PYBIND11
#include "pybind11/embed.h"
#else
#include <Python.h>
#endif

#include "LibytProcessControl.h"
#include "function_info.h"
#include "yt_combo.h"

int FunctionInfo::mpi_rank_;
int FunctionInfo::mpi_root_;
int FunctionInfo::mpi_size_;

//-------------------------------------------------------------------------------------------------------
// Class       :  FunctionInfo
// Method      :  Constructor
//
// Notes       :  1. Initialize:
//                   function_name_ does not include argument
//                   input_args_ : input arguments for new added functions.
//                                 It uses input_args_ when using FunctionInfoList::RunAllFunctions to run
//                                 all the functions.
//                2. Function body is stored in libyt.interactive_mode["func_body"], because we use Python to
//                   get the source code.
//
// Arguments   :  const char *func_name: inline function name
//                RunStatus   run      : will run in next iteration or not
//-------------------------------------------------------------------------------------------------------
FunctionInfo::FunctionInfo(const char* function_name, RunStatus run)
    : function_name_(function_name),
      input_args_(),
      run_(run),
      status_(kNotExecuteYet),
      all_status_(kNotExecuteYet),
      all_error_msg_() {
    SET_TIMER(__PRETTY_FUNCTION__);
    mpi_rank_ = LibytProcessControl::Get().mpi_rank_;
    mpi_root_ = LibytProcessControl::Get().mpi_root_;
    mpi_size_ = LibytProcessControl::Get().mpi_size_;
}

//-------------------------------------------------------------------------------------------------------
// Class       :  FunctionInfo
// Method      :  Copy Constructor
//
// Notes       :  1. It is inefficient to do it this way, but we are adding FunctionInfo class to
//                   g_func_status_list vector, which makes a copy.
//                   Although we can replace it to store class's pointer, I don't want to access through
//                   arrow.
//
// Arguments   :  const FunctionInfo& other
//-------------------------------------------------------------------------------------------------------
FunctionInfo::FunctionInfo(const FunctionInfo& other)
    : function_name_(other.function_name_),
      input_args_(other.input_args_),
      wrapper_(other.wrapper_),
      run_(other.run_),
      status_(other.status_),
      all_status_(other.all_status_),
      all_error_msg_(other.all_error_msg_) {
    SET_TIMER(__PRETTY_FUNCTION__);
}

//-------------------------------------------------------------------------------------------------------
// Class       :  FunctionInfo
// Method      :  GetFunctionNameWithInputArgs
//
// Notes       :  1. Return the string of how Python call this function, including arguments.
//
// Arguments   :  None
//
// Return      :  std::string function_call : how python will call this function, including args.
//-------------------------------------------------------------------------------------------------------
std::string FunctionInfo::GetFunctionNameWithInputArgs() {
    SET_TIMER(__PRETTY_FUNCTION__);

    std::string function_call = function_name_;
    function_call += "(" + input_args_ + ")";

    return function_call;
}

//-------------------------------------------------------------------------------------------------------
// Class       :  FunctionInfo
// Method      :  SetStatus
//
// Notes       :  1. Set the status of the function at current MPI process.
//                2. Because all_status_ represents the status of this function in every MPI process,
//                   we also need to set all_status_, and then make it update in GetAllStatus().
//
// Arguments   :  ExecuteStatus status : status to set
//-------------------------------------------------------------------------------------------------------
void FunctionInfo::SetStatus(FunctionInfo::ExecuteStatus status) {
    SET_TIMER(__PRETTY_FUNCTION__);

    status_ = status;
    all_status_ = kNeedUpdate;
}

//-------------------------------------------------------------------------------------------------------
// Class       :  FunctionInfo
// Method      :  SetStatusUsingPythonResult
//
// Notes       :  1. Set the status of this function at current MPI process.
//                2. It will check if function name exist in libyt.interactive_mode["func_err_msg"] keys,
//                   is so, then this MPI process has failed.
//                3. Also update all_status_ to need update, so that it will sync with other ranks.
//
// Arguments   :  None
//-------------------------------------------------------------------------------------------------------
void FunctionInfo::SetStatusUsingPythonResult() {
    SET_TIMER(__PRETTY_FUNCTION__);

    all_status_ = kNeedUpdate;

    // Check if libyt.interactive_mode["func_err_msg"] contains function name as its key, failed if there is
#ifdef USE_PYBIND11
    pybind11::module_ libyt = pybind11::module_::import("libyt");
    pybind11::dict py_func_err_msg = libyt.attr("interactive_mode")["func_err_msg"];
    status_ = py_func_err_msg.contains(function_name_) ? kFailed : kSuccess;
#else
    PyObject* py_func_name = PyUnicode_FromString(function_name_.c_str());
    status_ = (PyDict_Contains(PyDict_GetItemString(g_py_interactive_mode, "func_err_msg"), py_func_name) == 1)
                  ? kFailed
                  : kSuccess;
    Py_DECREF(py_func_name);
#endif
}

//-------------------------------------------------------------------------------------------------------
// Class       :  FunctionInfo
// Method      :  GetAllStatus
//
// Notes       :  1. Return the overall status of this function, if one rank failed, then it is failed.
//                2. If all_status_ = kNeedUpdate, then this is a collective call.
//                   After checking status at local, it will sync the other ranks' status.
//                3. Returned cached all_status_ if it is not kNeedUpdate.
//                4. SetStatus/SetStatusUsingPythonResult/GetAllStatus are in a group when running function.
//
// Arguments   :  None
//
// Return      :  all_status_
//-------------------------------------------------------------------------------------------------------
FunctionInfo::ExecuteStatus FunctionInfo::GetAllStatus() {
    SET_TIMER(__PRETTY_FUNCTION__);

    if (all_status_ != kNeedUpdate) {
        return all_status_;
    }

#ifndef SERIAL_MODE
    // Sync status_ to other ranks
    int total_status = 0;
    int my_status = (status_ == kSuccess) ? 1 : 0;
    MPI_Allreduce(&my_status, &total_status, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    all_status_ = (total_status == mpi_size_) ? kSuccess : kFailed;
#else
    all_status_ = status_;
#endif

    return all_status_;
}

//-------------------------------------------------------------------------------------------------------
// Class       :  FunctionInfo
// Method      :  GetAllErrorMsg
//
// Notes       :  1. This is a collective call. Must call by every rank.
//                2. Assert that every err msg line ends with newline \n.
//                3. Cache error message, every MPI process will cache it.
//
// Arguments   :  (None)
//
// Return      :  std::vector<std::string>& all_error_msg_ : error messages from every MPI process
//-------------------------------------------------------------------------------------------------------
std::vector<std::string>& FunctionInfo::GetAllErrorMsg() {
    SET_TIMER(__PRETTY_FUNCTION__);

    if (!all_error_msg_.empty()) {
        return all_error_msg_;
    }

    // Get error message from libyt.interactive_mode["func_err_msg"]
    const char* err_msg;
#ifdef USE_PYBIND11
    pybind11::module_ libyt = pybind11::module_::import("libyt");
    pybind11::dict py_func_err_msg = libyt.attr("interactive_mode")["func_err_msg"];
    std::string err_msg_str;
    if (py_func_err_msg.contains(function_name_)) {
        err_msg_str = std::string(py_func_err_msg[function_name_.c_str()].cast<std::string>());
        err_msg = err_msg_str.c_str();
    } else {
        err_msg = "";
    }
#else
    PyObject* py_func_name = PyUnicode_FromString(function_name_.c_str());
    PyObject* py_err_msg = PyDict_GetItem(PyDict_GetItemString(g_py_interactive_mode, "func_err_msg"), py_func_name);
    if (py_err_msg != NULL) {
        err_msg = PyUnicode_AsUTF8(py_err_msg);
    } else {
        err_msg = "";
    }
    Py_DECREF(py_func_name);
#endif

#ifndef SERIAL_MODE
    // Gather all error messages from all ranks
    int error_len = (int)strlen(err_msg);
    int* all_error_len = new int[mpi_size_];
    MPI_Allgather(&error_len, 1, MPI_INT, all_error_len, 1, MPI_INT, MPI_COMM_WORLD);

    long sum_output_len = 0;
    int* displace = new int[mpi_size_];
    for (int r = 0; r < mpi_size_; r++) {
        displace[r] = 0;
        sum_output_len += all_error_len[r];
        for (int r1 = 0; r1 < r; r1++) {
            displace[r] += all_error_len[r1];
        }
    }
    char* all_error = new char[sum_output_len + 1];
    MPI_Allgatherv(err_msg, all_error_len[mpi_rank_], MPI_CHAR, all_error, all_error_len, displace, MPI_CHAR,
                   MPI_COMM_WORLD);

    for (int r = 0; r < mpi_size_; r++) {
        all_error_msg_.emplace_back(std::string(all_error).substr(displace[r], all_error_len[r]));
    }

    delete[] all_error_len;
    delete[] displace;
    delete[] all_error;
#else
    all_error_msg_.emplace_back(std::string(err_msg));
#endif

    return all_error_msg_;
}

//-------------------------------------------------------------------------------------------------------
// Class       :  FunctionInfo
// Method      :  GetFunctionBody
//
// Notes       :  1. Get function body by getting libyt.interactive_mode["func_body"][function_name_],
//                   or else, return default "(Function body unknown)\n"
//
// Arguments   :  (None)
//
// Return      :  std::string func_body : function body
//-------------------------------------------------------------------------------------------------------
std::string FunctionInfo::GetFunctionBody() {
    SET_TIMER(__PRETTY_FUNCTION__);

    std::string func_body("(Function body unknown)\n");

#ifdef USE_PYBIND11
    pybind11::module_ libyt = pybind11::module_::import("libyt");
    pybind11::dict py_func_body = libyt.attr("interactive_mode")["func_body"];
    if (py_func_body.contains(function_name_)) {
        func_body = py_func_body[function_name_.c_str()].cast<std::string>();
    }
#else
    // get function body
    PyObject* py_func_name = PyUnicode_FromString(function_name_.c_str());
    PyObject* py_func_body = PyDict_GetItem(PyDict_GetItemString(g_py_interactive_mode, "func_body"), py_func_name);
    if (py_func_body != NULL) {
        func_body = std::string(PyUnicode_AsUTF8(py_func_body));
    }
    Py_DECREF(py_func_name);
#endif

    return func_body;
}

//-------------------------------------------------------------------------------------------------------
// Class       :  FunctionInfo
// Method      :  ClearAllErrorMsg
//
// Notes       :  1. Must call by every rank.
//                2. GetAllErrorMsg() will initiate a collective call if the cache is empty, so every
//                   rank must call it.
//
// Arguments   :  (None)
//-------------------------------------------------------------------------------------------------------
void FunctionInfo::ClearAllErrorMsg() {
    SET_TIMER(__PRETTY_FUNCTION__);

    all_error_msg_.clear();
}

//-------------------------------------------------------------------------------------------------------
// Class       :  FunctionInfoList
// Method      :  ResetEveryFunctionStatus
//
// Notes       :  1. Reset every function's status to FunctionInfo::kNotExecuteYet.
//                2. Clear the error buffer.
//                3. SetAllStatus() should come after SetStatus(), because SetStatus() will set all_status_,
//                   which is bad.
//
// Arguments   :  None
//-------------------------------------------------------------------------------------------------------
void FunctionInfoList::ResetEveryFunctionStatus() {
    SET_TIMER(__PRETTY_FUNCTION__);

    for (auto& func : function_list_) {
        func.SetStatus(FunctionInfo::kNotExecuteYet);
        func.SetAllStatus(FunctionInfo::kNotExecuteYet);
        func.ClearAllErrorMsg();
    }
}

//-------------------------------------------------------------------------------------------------------
// Class       :  FunctionInfoList
// Method      :  GetFunctionIndex
//
// Notes       :  1. Look up index of function name in the list. If the function doesn't exist, return -1.
//
// Arguments   :  std::string *func_name: inline function name
//
// Return      :  index : index of func_name in list, return -1 if it doesn't exist.
//-------------------------------------------------------------------------------------------------------
int FunctionInfoList::GetFunctionIndex(const std::string& function_name) {
    SET_TIMER(__PRETTY_FUNCTION__);

    int index = -1;
    for (size_t i = 0; i < function_list_.size(); i++) {
        if (function_list_[i].GetFunctionName() == function_name) {
            index = (int)i;
            break;
        }
    }

    return index;
}

//-------------------------------------------------------------------------------------------------------
// Class       :  FunctionInfoList
// Method      :  AddNewFunction
//
// Notes       :  1. Check if function name is defined inside the vector, if not create one.
//                2. Return function index.
//
// Arguments   :  std::string   *function_name: inline function name
//                RunStatus      run          : run in next inline analysis or not.
//
// Return      : int index : Function index in list.
//-------------------------------------------------------------------------------------------------------
int FunctionInfoList::AddNewFunction(const std::string& function_name, FunctionInfo::RunStatus run) {
    SET_TIMER(__PRETTY_FUNCTION__);

    int index = GetFunctionIndex(function_name);
    if (index < 0) {
        index = GetSize();
        function_list_.emplace_back(function_name.c_str(), run);
    }

    return index;
}

//-------------------------------------------------------------------------------------------------------
// Class       :  FunctionInfoList
// Method      :  RunEveryFunction
//
// Notes       :  1. This is a collective call. It executes new added functions that haven't run by
//                   yt_run_Function/yt_run_FunctionArguments yet.
//                2. How this method runs python function is identical to yt_run_Function*. It uses
//                   PyRun_SimpleString.
//                3. libyt uses either """ or ''' to wrap the code to execute in exec().
//                4. function_name_(input_args_) is how Python calls the function:
//                   try:
//                       exec(<wrapper><function_name_with_args><wrapper>, sys.modules["<script_name>"].__dict__)
//                   except Exception as e:
//                       libyt.interactive_mode["func_err_msg"]["<function_name>"] = traceback.format_exc()
//
// Arguments   :  (None)
//-------------------------------------------------------------------------------------------------------
void FunctionInfoList::RunEveryFunction() {
    SET_TIMER(__PRETTY_FUNCTION__);

    for (auto& function : function_list_) {
        FunctionInfo::RunStatus run = function.GetRun();
        FunctionInfo::ExecuteStatus status = function.GetStatus();
        if (run == FunctionInfo::kWillRun && status == FunctionInfo::kNotExecuteYet) {
            std::string command = std::string("try:\n"
                                              "    exec(") +
                                  function.GetWrapper() + function.GetFunctionNameWithInputArgs() +
                                  function.GetWrapper() + std::string(", sys.modules[\"") + g_param_libyt.script +
                                  std::string("\"].__dict__)\n"
                                              "except Exception as e:\n"
                                              "    libyt.interactive_mode[\"func_err_msg\"][\"") +
                                  function.GetFunctionName() + std::string("\"] = traceback.format_exc()\n");

            log_info("Performing YT inline analysis %s ...\n", function.GetFunctionNameWithInputArgs().c_str());
            function.SetStatus(FunctionInfo::kNeedUpdate);
            if (PyRun_SimpleString(command.c_str()) != 0) {
                // We set the status to failed even though this should never happen,
                // because the status is set based on if an error msg is set or not.
                function.SetStatus(FunctionInfo::kFailed);
                log_error("Unexpected error occurred when running PyRun_SimpleString\n");
            } else {
                function.SetStatusUsingPythonResult();
            }
            FunctionInfo::ExecuteStatus all_status = function.GetAllStatus();
            log_info("Performing YT inline analysis %s ... %s\n", function.GetFunctionNameWithInputArgs().c_str(),
                     (all_status == FunctionInfo::kSuccess) ? "done" : "failed");
        }
    }
}

#endif  // #if defined(INTERACTIVE_MODE) || defined(JUPYTER_KERNEL)
