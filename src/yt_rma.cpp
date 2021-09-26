#include "yt_rma.h"
#include <string.h>

//-------------------------------------------------------------------------------------------------------
// Class       :  yt_rma
// Method      :  Constructor
//
// Notes       :  1. Initialize m_Window, which used inside OpenMPI RMA operation.
//                2. Copy the input fname to m_FieldName, in case it is freed.
//
// Arguments   :  char* fname:
// Return      :  YT_SUCCESS or YT_FAIL
//-------------------------------------------------------------------------------------------------------
yt_rma::yt_rma(char* fname)
{
    // Initialize m_Window
    MPI_Win_create_dynamic(MPI_INFO_NULL, MPI_COMM_WORLD, &m_Window);

    // Copy input fname
    int len = strlen(fname);
    m_FieldName = new char [len+1];
    strcpy(m_FieldName, fname);
    printf("yt_rma: Field Name = %s\n", m_FieldName);
}

//-------------------------------------------------------------------------------------------------------
// Class       :  yt_rma
// Method      :  Destructor
//
// Notes       :  1. Freed m_Window, m_FieldName.
//
// Arguments   :  char* fname:
// Return      :  YT_SUCCESS or YT_FAIL
//-------------------------------------------------------------------------------------------------------
yt_rma::~yt_rma()
{
    MPI_Win_free(m_Window);
    delete [] m_FieldName;
    printf("yt_rma: Destructor called\n");
}