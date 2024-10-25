#ifndef LIBYT_PROJECT_INCLUDE_FUNCTION_INFO_H_
#define LIBYT_PROJECT_INCLUDE_FUNCTION_INFO_H_

#include <string>
#include <vector>

class FunctionInfo {
public:
    enum RunStatus { kNotSetYet = -1, kWillIdle = 0, kWillRun = 1 };
    enum ExecuteStatus { kNeedUpdate = -2, kNotExecuteYet = -1, kFailed = 0, kSuccess = 1 };

private:
    std::string function_name_;
    std::string input_args_;
    const char* wrapper_ = "\"\"\"";
    RunStatus run_;
    ExecuteStatus status_;
    ExecuteStatus all_status_;
    std::vector<std::string> all_error_msg_;

public:
    FunctionInfo(const char* function_name, RunStatus run);
    FunctionInfo(const FunctionInfo& other);

    std::string& GetFunctionName() { return function_name_; }
    std::string& GetInputArgs() { return input_args_; }
    void SetInputArgs(const std::string& args) { input_args_ = args; }
    const char* GetWrapper() { return wrapper_; }
    void SetWrapper(const char* wrapper) { wrapper_ = wrapper; }
    RunStatus GetRun() { return run_; }
    void SetRun(RunStatus run) { run_ = run; }
    ExecuteStatus GetStatus() { return status_; }

    void SetStatus(ExecuteStatus status);
    void SetStatusUsingPythonResult();
    std::string GetFunctionNameWithInputArgs();
    ExecuteStatus GetAllStatus();
    std::vector<std::string>& GetAllErrorMsg();
    std::string GetFunctionBody();
    void ClearAllErrorMsg();
};

class FunctionInfoList {
private:
    std::vector<FunctionInfo> function_list_;

public:
    explicit FunctionInfoList(int capacity) { function_list_.reserve(capacity); }
    ~FunctionInfoList() { function_list_.clear(); }
    FunctionInfo& operator[](int index) { return function_list_[index]; }

    int GetSize() { return (int)function_list_.size(); }

    void ResetEveryFunctionStatus();
    int GetFunctionIndex(const std::string& function_name);
    int AddNewFunction(const std::string& function_name, FunctionInfo::RunStatus run);
    void RunEveryFunction();
};

#endif  // LIBYT_PROJECT_INCLUDE_FUNCTION_INFO_H_
