#ifdef SUPPORT_TIMER

#include "timer.h"

#include <iostream>
#include <thread>

#include "libyt_process_control.h"

//-------------------------------------------------------------------------------------------------------
// Class       :  Timer
// Method      :  Constructor
// Description :  Record start time
//-------------------------------------------------------------------------------------------------------
Timer::Timer(const char* func_name) : m_FuncName(func_name), m_Stopped(false) {
    m_StartTime = std::chrono::high_resolution_clock::now();
}

//-------------------------------------------------------------------------------------------------------
// Class       :  Timer
// Method      :  Destructor
// Description :  Stop the clock
//-------------------------------------------------------------------------------------------------------
Timer::~Timer() {
    if (!m_Stopped) Stop();
}

//-------------------------------------------------------------------------------------------------------
// Class       :  Timer
// Method      :  Stop
// Description :  Write profile
//-------------------------------------------------------------------------------------------------------
void Timer::Stop() {
    std::chrono::time_point<std::chrono::high_resolution_clock> endTime = std::chrono::high_resolution_clock::now();

    long long start = std::chrono::time_point_cast<std::chrono::microseconds>(m_StartTime).time_since_epoch().count();
    long long end = std::chrono::time_point_cast<std::chrono::microseconds>(endTime).time_since_epoch().count();
    uint32_t threadID = std::hash<std::thread::id>{}(std::this_thread::get_id());

    LibytProcessControl::Get().timer_control.WriteProfile(m_FuncName, start, end, threadID);

    m_Stopped = true;
}

#endif  // #ifdef SUPPORT_TIMER
