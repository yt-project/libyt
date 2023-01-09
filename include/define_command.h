#ifndef __DEFINE_COMMAND_H__
#define __DEFINE_COMMAND_H__

#include <string>

class define_command {
private:
    std::string m_Command;
    static int s_Root;

public:
    define_command(char *command) : m_Command(command) {};
    int run();
    bool is_exit();

    static int print_status();
    static int print_help_msg();
    static int load_script(const char *filename);
    static int export_script(const char *filename);
};


#endif //__DEFINE_COMMAND_H__
