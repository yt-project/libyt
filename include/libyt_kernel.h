#ifndef LIBYT_PROJECT_INCLUDE_LIBYT_KERNEL_H_
#define LIBYT_PROJECT_INCLUDE_LIBYT_KERNEL_H_

#include <Python.h>

#include <nlohmann/json.hpp>
#include <xeus/xinterpreter.hpp>

class LibytKernel : public xeus::xinterpreter {
 public:
  LibytKernel() = default;
  virtual ~LibytKernel() = default;

 private:
  PyObject* m_py_jedi_interpreter;

  void configure_impl() override;
  nl::json execute_request_impl(int execution_counter, const std::string& code,
                                bool silent, bool store_history,
                                nl::json user_expressions, bool allow_stdin) override;
  nl::json complete_request_impl(const std::string& code, int cursor_pos) override;
  nl::json inspect_request_impl(const std::string& code, int cursor_pos,
                                int detail_level) override;
  nl::json is_complete_request_impl(const std::string& code) override;
  nl::json kernel_info_request_impl() override;
  void shutdown_request_impl() override;
};

#endif  // LIBYT_PROJECT_INCLUDE_LIBYT_KERNEL_H_
