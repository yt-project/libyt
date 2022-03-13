#ifdef SUPPORT_TIMER

#include "Timer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include <ctype.h>

Timer::Timer(char *filename)
: m_PrintHeader(false)
{
    // Save write-to filename and path.
    strcpy(m_FileName, filename);
    sprintf(m_TempFileName, "%s_temp", filename);
}

Timer::~Timer()
{
    // Clear vector.
    m_RecordTime.clear();
    m_CheckRecordTime.clear();
    m_Column.clear();
}

void Timer::print_header()
{
    // Read original file, and create a temporary new file.
    FILE *original, *output;
    original = fopen(m_FileName, "r");
    output   = fopen(m_TempFileName, "w");

    for(int i = 0; i < m_Column.size(); i++){
        fprintf(output, "%s,", m_Column[i].c_str());
    }
    fprintf(output, "\n");

    // Move original record time to temp.
    // If original exists, then it must have contain headers. So we skip first line.
    if( original ){
        char ch;
        bool first_line = true;
        while( (ch = fgetc(original)) != EOF ){
            if( !first_line ){
                fputc(ch, output);
            }
            else if( ch == '\n' ){
                first_line = false;
            }
        }
        fclose(original);
    }
    fclose(output);

    // Remove original file and rename temp file to replace origin.
    char cmd[60];
    sprintf(cmd, "rm -f %s; mv %s %s", m_FileName, m_TempFileName, m_FileName);
    system(cmd);
}

void Timer::record_time(char *Column, int tag)
{
    // Get wall time.
    double time = MPI_Wtime();

    // Search column index.
    int column_index = -1;
    for(int i = 0; i < m_Column.size(); i++){
        if( strcmp(m_Column[i].c_str(), Column) == 0){
            column_index = i;
            break;
        }
    }

    // If no such column, add a new column.
    if( column_index < 0 ){
        column_index = m_Column.size();
        std::string s(Column);
        m_Column.push_back(s);
        m_RecordTime.push_back(0);
        m_CheckRecordTime.push_back(false);
        m_PrintHeader = true;
    }

    // Record time, if tag = 0. Calculate time pass, if tag = 1.
    if( tag == 0 ){
        m_RecordTime[column_index] = time;
        m_CheckRecordTime[column_index] = true;
    }
    else if( tag == 1 && m_CheckRecordTime[column_index]){
        m_RecordTime[column_index] = time - m_RecordTime[column_index];
    }
}

void Timer::print_all_time()
{
    if( m_PrintHeader ){
        print_header();
        m_PrintHeader = false;
    }

    // Open file
    FILE *output;
    output = fopen(m_FileName, "a");

    // Flush all saved time, and reset check m_CheckRecordTime.
    for(int i = 0; i < m_Column.size(); i++){
        if(m_CheckRecordTime[i]){
            fprintf(output, "%lf,", m_RecordTime[i]);
        }
        else{
            fprintf(output, "NaN,");
        }
        m_CheckRecordTime[i] = false;
    }
    fprintf(output, "\n");
    fclose(output);
}

#endif // #ifdef SUPPORT_TIMER
