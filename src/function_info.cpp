#if defined(INTERACTIVE_MODE) || defined(JUPYTER_KERNEL)

#ifdef USE_PYBIND11
#include "pybind11/embed.h"
#else
#include <Python.h>
#endif

#include "function_info.h"
#include "yt_combo.h"

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
}

//-------------------------------------------------------------------------------------------------------
// Class       :  FunctionInfo
// Method      :  Copy Constructor
//
// Notes       :  1. It is inefficient to do it this way, but we are adding func_status class to
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
// Method      :  GetStatus
//
// Notes       :  1. Return the status of this function at current MPI process
//                2. If status_ = kNeedUpdate, it will check if function name exist in
//                   libyt.interactive_mode["func_err_msg"] keys, is so, then this MPI process has failed.
//                3. Returned cached status_ if it is not kNeedUpdate.
//
// Arguments   :  None
//
// Return      :  all_status_
//-------------------------------------------------------------------------------------------------------
FunctionInfo::ExecuteStatus FunctionInfo::GetStatus() {
    SET_TIMER(__PRETTY_FUNCTION__);

    if (status_ != kNeedUpdate) {
        return status_;
    }

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

    return status_;
}

//-------------------------------------------------------------------------------------------------------
// Class       :  FunctionInfo
// Method      :  GetAllStatus
//
// Notes       :  1. Return the overall status of this function, if one rank failed, then it is failed.
//                2. If all_status_ = kNeedUpdate, then this is a collective call.
//                   After checking status at local, it will sync the other ranks' status.
//                3. Returned cached all_status_ if it is not kNeedUpdate.
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

    if (status_ == kNeedUpdate) {
        status_ = GetStatus();
    }

#ifndef SERIAL_MODE
    // Sync status_ to other ranks
    int total_status = 0;
    int my_status = (status_ == kSuccess) ? 1 : 0;
    MPI_Allreduce(&my_status, &total_status, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    all_status_ = (total_status == g_mysize) ? kSuccess : kFailed;
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
#ifdef USE_PYBIND11
    // TODO: (START HERE)
#else
#endif

    return all_error_msg_;
}
std::string FunctionInfo::GetFunctionBody() { return std::string(); }
void FunctionInfo::ClearAllErrorMsg() {}

FunctionInfoList::FunctionInfoList(int capacity) {}
FunctionInfoList::~FunctionInfoList() {}
FunctionInfo& FunctionInfoList::operator[](int index) { return <#initializer #>; }
void FunctionInfoList::Reset() {}
size_t FunctionInfoList::GetSize() { return 0; }
int FunctionInfoList::GetFunctionIndex(const std::string& function_name) { return 0; }
void FunctionInfoList::AddNewFunction(const std::string& function_name, FunctionInfo::RunStatus run) {}
void FunctionInfoList::RunAllFunctions() {}

#endif  // #if defined(INTERACTIVE_MODE) || defined(JUPYTER_KERNEL)
