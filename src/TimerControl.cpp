#ifdef SUPPORT_TIMER

#include <fstream>
#include <string.h>
#include <iostream>
#include "TimerControl.h"
#include "libyt.h"


//-------------------------------------------------------------------------------------------------------
// Class       :  TimerControl
// Method      :  CreateFile
// Description :  Create profile result file and write headings
//-------------------------------------------------------------------------------------------------------
void TimerControl::CreateFile(const char *filename, int rank) {
    // Initialize
    m_FileName = std::string(filename);
    m_MPIRank = rank;
    m_FirstLine = true;

    // Overwrite and create profile file
    std::ofstream file_out;
    file_out.open(m_FileName.c_str(), std::ofstream::out);

    // Write heading and basic info
    file_out << "{\"otherData\": {"
             << "\"version\": \"" << LIBYT_MAJOR_VERSION << "." << LIBYT_MINOR_VERSION << "." << LIBYT_MICRO_VERSION << "\","
#ifdef INTERACTIVE_MODE
             << "\"mode\": " << "\"interactive_mode\""
#else
             << "\"mode\": " << "\"normal_mode\""
#endif
             << "},";
    file_out << "\"traceEvents\":[";

    file_out.close();
}


//-------------------------------------------------------------------------------------------------------
// Class       :  TimerControl
// Method      :  WriteProfile
// Description :  Write profile to file
//
// Notes       :  1. Please refer to chrome tracing format
//                   (https://docs.google.com/document/d/1CvAClvFfyA5R-PhYUmn5OOQtYMH4h6I0nSsKchNAySU/preview#heading=h.uxpopqvbjezh)
//
// Parameters  :  func_name : function name
//                start     : start time
//                end       : end time
//                thread_id : thread id
//-------------------------------------------------------------------------------------------------------
void TimerControl::WriteProfile(const char *func_name, long long start, long long end, uint32_t thread_id) {
    // Set profile string, write to file
    char profile[1000];
    sprintf(profile, "%s{\"name\":\"%s\","
                       "\"cat\":\"function\","
                       "\"dur\":%lld,"
                       "\"ph\":\"X\","
                       "\"pid\":%d,"
                       "\"tid\":%llu,"
                       "\"ts\":%lld"
                       "}",
                       m_FirstLine ? "" : ",", func_name, end - start, m_MPIRank, (long long int)thread_id, start);

    std::ofstream  file_out;
    file_out.open(m_FileName.c_str(), std::ofstream::out | std::ofstream::app);
    file_out.write(profile, strlen(profile));
    m_FirstLine = false;

    file_out.close();
}

#endif // #ifdef SUPPORT_TIMER
