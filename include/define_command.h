#ifndef __DEFINE_COMMAND_H__
#define __DEFINE_COMMAND_H__

#include <string>

class define_command {
private:
    std::string m_Command;
    bool m_Undefine;
    static int s_Root;

public:
    define_command(char *command) : m_Command(command), m_Undefine(true) {};
    bool run();

    int print_status();
    int print_help_msg();
    int load_script(const char *filename);
    int export_script(const char *filename);
};


#endif //__DEFINE_COMMAND_H__
