#ifndef __MAGIC_COMMAND_H__
#define __MAGIC_COMMAND_H__

#include <string>
#include <vector>

struct OutputData {
    std::string status;
    std::string mimetype;
    std::string output;
    std::string error;
};

class MagicCommand {
private:
    std::string m_Command;
    bool m_Undefine;
    OutputData m_OutputData;
    static int s_Root;

public:
    MagicCommand() : m_Command(""), m_Undefine(true), m_OutputData(){};
    OutputData& run(const std::string& command = std::string(""));

    int exit();
    int get_status();
    int get_help_msg();
    int load_script(const std::string& filename);
    int export_script(const std::string& filename);
    int set_func_run(const std::string& funcname, bool run);
    int set_func_run(const std::string& funcname, bool run, std::vector<std::string>& arg_list);
    int get_func_status(const char* funcname);
};

#endif  //__MAGIC_COMMAND_H__
