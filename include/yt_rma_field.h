#ifndef __YT_RMA_FIELD_H__
#define __YT_RMA_FIELD_H__

/*******************************************************************************
/
/  yt_rma_field class.
/
/  ==> included by init_libyt_module.cpp.
/
********************************************************************************/

#include <vector>
#include "yt_combo.h"

//-------------------------------------------------------------------------------------------------------
// Class       :  yt_rma_field
// Description :  Class to deal with RMA operation, to get non-local grids.
// 
// Notes       :  1. This class deals with OpenMPI RMA operation, should be called by every rank.
//                2. This class isn't aim to deal with generic RMA operation, it is meant to work with 
//                   libyt frontend in YT when getting non-local grids. 
//                3. First assume that each rank has the same fname_list, and their fname order are the 
//                   same.
//                4. One instance with one field and one window.
//
//-------------------------------------------------------------------------------------------------------
class yt_rma_field
{
private:

	MPI_Win  m_Window;
	char    *m_FieldName;
    char    *m_FieldDefineType;
    int      m_FieldIndex;
    bool     m_FieldSwapAxes;

	std::vector<yt_rma_grid_info> m_Prepare;
    std::vector<void*> m_PrepareData;

	long m_LenAllPrepare;
	yt_rma_grid_info  *m_AllPrepare;

    std::vector<yt_rma_grid_info> m_Fetched;
    std::vector<void*> m_FetchedData;

public:
    yt_rma_field(char* fname, int len_prepare, long len_get_grid);
	~yt_rma_field();

    // OpenMPI RMA operation
	int prepare_data(long& gid);
    int gather_all_prepare_data(int root);
    int fetch_remote_data(long& gid, int& rank);
    int clean_up();
    int get_fetched_data(long *gid, char **fname, yt_dtype *data_dtype, int (*data_dim)[3], void **data_ptr);
};

#endif // #ifndef __YT_RMA_FIELD_H__