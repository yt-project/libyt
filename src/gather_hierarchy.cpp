#include "yt_combo.h"
#include "libyt.h"

//-------------------------------------------------------------------------------------------------------
// Structure   :  yt_hierarchy
// Description :  Data structure for pass struct data in MPI process, it is meant to be temperary.
//
// Data Member :  dimensions     : Number of cells along each direction
//                left_edge      : Grid left  edge in code units
//                right_edge     : Grid right edge in code units
//                particle_count : Nunber of particles in this grid
//                level          : AMR level (0 for the root level)
//                proc_num       : An array of MPI rank that the grid belongs
//                id             : Grid ID (0-indexed ==> must be in the range 0 <= id < total number of grids)
//                parent_id      : Parent grid ID (0-indexed, -1 for grids on the root level)
//                proc_num       : Process number, grid belong to which MPI rank
//-------------------------------------------------------------------------------------------------------
typedef struct yt_hierarchy{
	double left_edge[3];
   	double right_edge[3];

   	long   particle_count;
   	long   id;
   	long   parent_id;

   	int    dimensions[3];
   	int    level;
   	int    proc_num;
};

//-------------------------------------------------------------------------------------------------------
// Function    :  gather_hierarchy.cpp
// Description :  Gather the hierarchy in each rank.
//
// Note        :  1. Call by yt_add_grid()
//                2. Gather the hierarchy to RootRank, then boardcast to the other ranks. 
//                   (TODO: Should be tested to see if boardcast is needed.)
//                3. 
//
// Parameter   : 
//
// Return      :  YT_SUCCESS or YT_FAIL
//-------------------------------------------------------------------------------------------------------

int gather_hierarchy() {

	int MyRank;
	int NRank;
	int RootRank = 0;

	MPI_Comm_size(MPI_COMM_WORLD, &NRank);
	MPI_Comm_rank(MPI_COMM_WORLD, &MyRank);

	// Create MPI_Datatype for struct yt_hierarchy
	MPI_Datatype yt_hierarchy_mpi_type;
	int lengths[8] = { 3, 3, 1, 1, 1, 3, 1, 1 };
	const MPI_Aint displacements[8] = { 0, 3 * sizeof(double), 6 * sizeof(double),
		                               	6 * sizeof(double) + sizeof(long), 6 * sizeof(double) + 2 * sizeof(long), 
		                               	6 * sizeof(double) + 3 * sizeof(long),
		                               	6 * sizeof(double) + 3 * sizeof(long) + 3 * sizeof(int), 
		                               	6 * sizeof(double) + 3 * sizeof(long) + 4 * sizeof(int)};
	MPI_Datatype types[8] = { MPI_DOUBLE, MPI_DOUBLE, MPI_LONG, MPI_LONG, MPI_LONG, MPI_INT, MPI_INT, MPI_INT };
	MPI_Type_create_struct(8, lengths, displacements, types, &yt_hierarchy_mpi_type);
	MPI_Type_commit(&yt_hierarchy_mpi_type);





	return YT_SUCCESS;
}