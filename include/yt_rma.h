#ifndef __YT_RMA_H__
#define __YT_RMA_H__

/*******************************************************************************
/
/  yt_rma class and all the related struct.
/
/  ==> included by init_libyt_module.cpp.
/
********************************************************************************/

#include <mpi.h>
#include <vector>
#include "yt_combo.h"

struct yt_rma_grid_info
{	
	long     id;
	MPI_Aint address;	
	int      rank;
	yt_dtype grid_dtype;
	int      grid_dimensions[3];
};


//-------------------------------------------------------------------------------------------------------
// Class       :  yt_rma
// Description :  Class to deal with RMA operation, to get non-local grids.
// 
// Notes       :  1. This class deals with OpenMPI RMA operation, should be called by every rank.
//                2. This class isn't aim to deal with generic RMA operation, it is meant to work with 
//                   libyt frontend in YT when getting non-local grids. 
//                3. First assume that each rank has the same fname_list, and their fname order are the 
//                   same.
//                4. One instance with one field and one window.
// 
// Data Member :  //TODO
// 
// Method      :  //TODO
//-------------------------------------------------------------------------------------------------------
class yt_rma
{
private:

	MPI_Win  m_Window;
	char    *m_FieldName;

	std::vector<yt_rma_grid_info> m_Prepare;
    std::vector<void*> m_PrepareData;

	long m_LenAllPrepare;
	yt_rma_grid_info  *m_AllPrepare;

    std::vector<yt_rma_grid_info> m_Fetched;
    std::vector<void*> m_FetchedData;

public:
	yt_rma(char *fname);
	~yt_rma();

    // OpenMPI RMA operation
	int prepare_data(long& gid);
    int fetch_remote_data(long& gid, int& rank);
    int clean_up();
    int get_fetched_data(long& gid);

private:
    // Field and particle dependent
    static int get_size(int& dim0, int& dim1, int& dim2);
    // Field and particle independent
    static int get_mpi_type(yt_dtype& grid_dtype);
};

#endif // #ifndef __YT_RMA_H__