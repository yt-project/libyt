#ifndef __TIMER_H__
#define __TIMER_H__

#ifdef SUPPORT_TIMER
#include <chrono>

class Timer {
public:
    Timer(const char* func_name);
    ~Timer();
    void Stop();

private:
    const char* m_FuncName;
    std::chrono::time_point<std::chrono::high_resolution_clock> m_StartTime;
    bool m_Stopped;
};
#endif  // #ifdef SUPPORT_TIMER

#ifdef SUPPORT_TIMER
#define SET_TIMER(x) Timer Timer(x)
#else
#define SET_TIMER(x)
#endif  // #ifdef SUPPORT_TIMER

#endif  //__TIMER_H__
