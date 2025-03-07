#ifndef LIBYT_PROJECT_INCLUDE_MAGIC_COMMAND_H_
#define LIBYT_PROJECT_INCLUDE_MAGIC_COMMAND_H_

#include <string>
#include <vector>

struct MagicCommandOutput {
  bool exit_entry_point;
  std::string status;
  std::string mimetype;
  std::string output;
  std::string error;

  MagicCommandOutput()
      : exit_entry_point(false), status("Unknown"), mimetype("text/plain") {};
};

class MagicCommand {
 public:
  enum EntryPoint {
    kLibytInteractiveMode = 0,
    kLibytReloadScript = 1,
    kLibytJupyterKernel = 2
  };

 private:
  std::string command_;
  MagicCommandOutput output_;
  EntryPoint entry_point_;
  bool command_undefined_;
  static int mpi_root_;
  static int mpi_rank_;
  static int mpi_size_;

  int Exit();
  int GetStatusHtml();
  int GetStatusText();
  int GetHelpMsgMarkdown();
  int GetHelpMsgText();
  int LoadScript(const std::vector<std::string>& args);
  int ExportScript(const std::vector<std::string>& args);
  int SetFunctionRun(const std::vector<std::string>& args);
  int SetFunctionIdle(const std::vector<std::string>& args);
  int GetFunctionStatusMarkdown(const std::vector<std::string>& args);
  int GetFunctionStatusText(const std::vector<std::string>& args);

 public:
  explicit MagicCommand(EntryPoint entry_point);
  MagicCommandOutput& Run(const std::string& command = std::string(""));
};

#endif  // LIBYT_PROJECT_INCLUDE_MAGIC_COMMAND_H_
