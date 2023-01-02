#ifdef INTERACTIVE_MODE

#include "define_command.h"
#include "yt_combo.h"

int define_command::run() {
    
    return YT_SUCCESS;
}

bool define_command::is_exit() {
    std::size_t start_pos = 0;
    std::size_t found = m_Command.find("exit", start_pos);
    if (found != std::string::npos) return true;
    else return false;
}

#endif // #ifdef INTERACTIVE_MODE
