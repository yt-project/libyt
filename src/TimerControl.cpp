#ifdef SUPPORT_TIMER

#include "TimerControl.h"

#include <string.h>

#include <algorithm>
#include <fstream>
#include <iostream>

#include "libyt.h"

//-------------------------------------------------------------------------------------------------------
// Class       :  TimerControl
// Method      :  CreateFile
// Description :  Create profile result file and write headings
//-------------------------------------------------------------------------------------------------------
void TimerControl::CreateFile(const char* filename, int rank) {
    // Initialize
    m_FileName = std::string(filename);
    m_MPIRank = rank;
    m_FirstLine = true;

    // Overwrite and create profile file
    std::ofstream file_out;
    file_out.open(m_FileName.c_str(), std::ofstream::out);

    // Write heading and basic info
    if (m_MPIRank == 0) {
        file_out << "{\"otherData\": {"
                 << "\"version\": \"" << LIBYT_MAJOR_VERSION << "." << LIBYT_MINOR_VERSION << "." << LIBYT_MICRO_VERSION
                 << "\","
                 << "\"mode\": "
#if defined(INTERACTIVE_MODE)
                 << "\"interactive_mode\""
#elif defined(JUPYTER_KERNEL)
                 << "\"jupyter_kernel_mode\""
#else
                 << "\"normal_mode\""
#endif
                 << "},";
        file_out << "\"traceEvents\":[";
    }

    file_out.close();
}

//-------------------------------------------------------------------------------------------------------
// Class       :  TimerControl
// Method      :  WriteProfile
// Description :  Write profile to file
//
// Notes       :  1. Please refer to chrome tracing format
//                   (https://docs.google.com/document/d/1CvAClvFfyA5R-PhYUmn5OOQtYMH4h6I0nSsKchNAySU/preview#heading=h.uxpopqvbjezh)
//                2. This is thread-safe.
//                3. Function name cannot contain " double-quote, it will be replaced to '.
//
// Parameters  :  func_name : function name
//                start     : start time
//                end       : end time
//                thread_id : thread id
//-------------------------------------------------------------------------------------------------------
void TimerControl::WriteProfile(const char* func_name, long long start, long long end, uint32_t thread_id) {
    std::lock_guard<std::mutex> lock(m_Lock);

    // replace " to ' in func_name;
    std::string func_name_str = std::string(func_name);
    std::replace(func_name_str.begin(), func_name_str.end(), '"', '\'');

    // Set profile string, write to file
    char profile[1000];
    sprintf(profile,
            "%s{\"name\":\"%s\","
            "\"cat\":\"function\","
            "\"dur\":%lld,"
            "\"ph\":\"X\","
            "\"pid\":%d,"
            "\"tid\":%llu,"
            "\"ts\":%lld"
            "}",
            m_FirstLine ? "" : ",", func_name_str.c_str(), end - start, m_MPIRank, (long long int)thread_id, start);

    std::ofstream file_out;
    file_out.open(m_FileName.c_str(), std::ofstream::out | std::ofstream::app);
    file_out.write(profile, strlen(profile));
    m_FirstLine = false;

    file_out.close();
}

#endif  // #ifdef SUPPORT_TIMER
