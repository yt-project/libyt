#ifndef __DEFINE_COMMAND_H__
#define __DEFINE_COMMAND_H__

#include <array>
#include <string>
#include <vector>

class define_command {
private:
    std::string m_Command;
    std::string m_OutputFileName;
    bool m_Undefine;
    static int s_Root;

    int print_status();
    int print_help_msg();
    int load_script(const char* filename);
    int export_script(const char* filename);
    int set_func_run(const char* funcname, bool run);
    int set_func_run(const char* funcname, bool run, std::vector<std::string>& arg_list);
    int get_func_status(const char* funcname);
    void write_to_file(const char* format, ...);

public:
    define_command() : m_Undefine(true){};
    define_command(const std::string& output_filename) : m_OutputFileName(output_filename), m_Undefine(true){};
    std::array<bool, 2> run(const std::string& command = std::string(""));
};

#endif  //__DEFINE_COMMAND_H__
