#ifndef __MAGIC_COMMAND_H__
#define __MAGIC_COMMAND_H__

#include <string>
#include <vector>

class MagicCommand {
private:
    std::string m_Command;
    bool m_Undefine;
    static int s_Root;

public:
    MagicCommand() : m_Command(""), m_Undefine(true){};
    bool run(const std::string& command = std::string(""));

    int print_status();
    int print_help_msg();
    int load_script(const char* filename);
    int export_script(const char* filename);
    int set_func_run(const char* funcname, bool run);
    int set_func_run(const char* funcname, bool run, std::vector<std::string>& arg_list);
    int get_func_status(const char* funcname);
};

#endif  //__MAGIC_COMMAND_H__
