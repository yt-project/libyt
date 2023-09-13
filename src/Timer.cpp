#ifdef SUPPORT_TIMER

#include <mpi.h>
#include <iostream>
#include "Timer.h"

Timer::Timer(const char *func_name)
: m_FuncName(func_name), m_Stopped(false) {
    m_StartTime = std::chrono::high_resolution_clock::now();
}

Timer::~Timer() {
    if (!m_Stopped)
        Stop();
}

void Timer::Stop() {
    std::chrono::time_point<std::chrono::high_resolution_clock> endTime = std::chrono::high_resolution_clock::now();

    long long start = std::chrono::time_point_cast<std::chrono::microseconds>(m_StartTime).time_since_epoch().count();
    long long end = std::chrono::time_point_cast<std::chrono::microseconds>(endTime).time_since_epoch().count();
    uint32_t threadID = std::hash<std::thread::id>{}(std::this_thread::get_id());
    int myrank;
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

    // TODO: Make it print in TimerControl
    std::cout << "[MPI " << myrank << ", threadID " << threadID << "] ";
    std::cout << "T( " << m_FuncName << ") = " << end - start;

    m_Stopped = true;
}

#endif // #ifdef SUPPORT_TIMER

