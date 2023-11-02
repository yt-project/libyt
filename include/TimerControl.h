#ifndef __TIMERCONTROL_H__
#define __TIMERCONTROL_H__

#ifdef SUPPORT_TIMER
#include <mutex>
#include <string>

class TimerControl {
public:
    TimerControl(){};
    void CreateFile(const char* filename, int rank);
    void WriteProfile(const char* func_name, long long start, long long end, uint32_t thread_id);

private:
    std::string m_FileName;
    int m_MPIRank;
    bool m_FirstLine;
    std::mutex m_Lock;
};
#endif  // #ifdef SUPPORT_TIMER

#endif  // #ifndef __TIMERCONTROL_H__
