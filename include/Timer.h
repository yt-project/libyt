#ifndef __TIMER_H__
#define __TIMER_H__

#include <vector>
#include <string>

class Timer
{
private:
    char  m_FileName[50];
    char  m_TempFileName[60];
    long  m_Step;

    std::vector<double> m_RecordTime;
    std::vector<bool>   m_CheckRecordTime;
    std::vector<std::string> m_Column;

public:
    Timer(char *filename);
    ~Timer();

    void print_header();
    void record_time(char *Column, int tag);
    void print_all_time();
};


#endif //__TIMER_H__
