#ifndef __DEFINE_COMMAND_H__
#define __DEFINE_COMMAND_H__

#include <string>

class define_command {
private:
    std::string m_Command;

public:
    define_command(char *command) : m_Command(command) {};
    int run();
    bool is_exit();


};


#endif //__DEFINE_COMMAND_H__
