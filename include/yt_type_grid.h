#ifndef __YT_TYPE_GRID_H__
#define __YT_TYPE_GRID_H__



/*******************************************************************************
/
/  yt_grid structure
/
/  ==> included by yt_type.h
/
********************************************************************************/

//-------------------------------------------------------------------------------------------------------
// Structure   :  yt_data
// Description :  Data structure to store a field data's pointer and its array dimensions.
// 
// Notes       :  1. This struct will be use in yt_grid data member field_data and particle_data.
// 
// Data Member :  data_ptr           : data pointer
//                data_dimensions[3] : dimension of the data to be passed to python for wrapping, which is the actual
//                                     size of the array.
//                                     Def => fieldData[ dim[0] ][ dim[1] ][ dim[2] ]
//                data_dtype         : Data type of the field in specific grid. If this is set as YT_DTYPE_UNKNOWN,
//                                     then we will use field_dtype define in field_list as input field data type.
//
// Methods     : yt_data            : Constructor
//               ~yt_data           : Destructor, does nothing for now.
//-------------------------------------------------------------------------------------------------------
struct yt_data
{
    void     *data_ptr;
    int       data_dimensions[3];
    yt_dtype  data_dtype;

    //===================================================================================
    // Method      :  yt_data
    // Description :  Constructor of the structure "yt_data"
    //
    // Note        :  Initialize all data members
    //
    // Parameter   :  None
    //===================================================================================
    yt_data()
    {
        data_ptr = NULL;
        for(int d=0; d<3; d++){ data_dimensions[d] = 0; }
        data_dtype = YT_DTYPE_UNKNOWN;
    }
    ~yt_data()
    {

    }
};

//-------------------------------------------------------------------------------------------------------
// Structure   :  yt_grid
// Description :  Data structure to store a full single grid with data pointer
// 
// Notes       :  1. We assume that each element in array[3] are all in use, which is we only supports 
//                   dim 3 for now.
//
// Data Member :  grid_dimensions : Number of cells along each direction in [x][y][z] coordinate.
//                left_edge       : Grid left  edge in code units
//                right_edge      : Grid right edge in code units
//                par_count_list  : Array that records number of particles in each species, the input order
//                                  should be the same as the input particle_list.
//                level           : AMR level (0 for the root level)
//                id              : Grid ID
//                parent_id       : Parent grid ID
//                proc_num        : Process number, grid belong to which MPI rank
//                field_data      : Pointer pointing to yt_data array, each element stores an info for
//                                  field array to be wrapped.
//                particle_data   : Pointer to pointer pointing to yt_data array (yt_data** array),
//                                  Ex: particle_data[0][1] represents particle data for
//                                  particle_list[0].attr_list[1].
//
// Method      :  yt_grid  : Constructor
//               ~yt_grid  : Destructor
//                validate : Check if all data members have been set properly by users
//-------------------------------------------------------------------------------------------------------
struct yt_grid
{

// data members
// ===================================================================================
    double     left_edge[3];
    double     right_edge[3];

    long      *par_count_list;
    long       id;
    long       parent_id;

    int        grid_dimensions[3];
    int        level;
    int        proc_num;

    yt_data   *field_data;
    yt_data  **particle_data;

    //===================================================================================
    // Method      :  yt_grid
    // Description :  Constructor of the structure "yt_grid"
    //
    // Note        :  Initialize all data members
    //
    // Parameter   :  None
    //===================================================================================
    yt_grid()
    {

//    set defaults
        for (int d=0; d<3; d++) {
            left_edge [d]  = DBL_UNDEFINED;
            right_edge[d]  = DBL_UNDEFINED; }

        for (int d=0; d<3; d++) {
            grid_dimensions[d]  = INT_UNDEFINED; }

        par_count_list = NULL;
        id             = LNG_UNDEFINED;
        parent_id      = LNG_UNDEFINED;
        level          = INT_UNDEFINED;

        proc_num       = INT_UNDEFINED;

        field_data     = NULL;
        particle_data  = NULL;

    } // METHOD : yt_grid


    //===================================================================================
    // Method      :  ~yt_grid
    // Description :  Destructor of the structure "yt_grid"
    //
    // Note        :  1. Not used currently
    //                2. We do not free the pointer arrays field_data, particle_data
    //
    // Parameter   :  None
    //===================================================================================
    ~yt_grid()
    {

    } // METHOD : ~yt_grid


    //===================================================================================
    // Method      :  validate
    // Description :  Check if all data members have been set properly by users
    //
    // Note        :  1. This function does not perform checks that depend on the input
    //                   YT parameters (e.g., whether left_edge lies within the simulation domain)
    //                   ==> These checks are performed in check_grid()
    //                2. If check needs information other than grid info, we will do it elsewhere.
    //
    // Parameter   :  None
    //
    // Return      :  YT_SUCCESS or YT_FAIL
    //===================================================================================
    int validate() const
    {

        for (int d=0; d<3; d++) {
            if ( left_edge [d]  == DBL_UNDEFINED    )   YT_ABORT( "\"%s[%d]\" has not been set for grid id [%ld]!\n", "left_edge",  d,  id );
            if ( right_edge[d]  == DBL_UNDEFINED    )   YT_ABORT( "\"%s[%d]\" has not been set for grid id [%ld]!\n", "right_edge", d,  id ); }

        for (int d=0; d<3; d++) {
            if ( grid_dimensions[d]  == INT_UNDEFINED )   YT_ABORT( "\"%s[%d]\" has not been set for grid id [%ld]!\n", "grid_dimensions", d,  id ); }
        if ( id             == LNG_UNDEFINED    )   YT_ABORT(     "\"%s\" has not been set for grid id [%ld]!\n", "id",             id );
        if ( parent_id      == LNG_UNDEFINED    )   YT_ABORT(     "\"%s\" has not been set for grid id [%ld]!\n", "parent_id",      id );
        if ( level          == INT_UNDEFINED    )   YT_ABORT(     "\"%s\" has not been set for grid id [%ld]!\n", "level",          id );
        if ( proc_num       == INT_UNDEFINED    )   YT_ABORT(     "\"%s\" has not been set for grid id [%ld]!\n", "proc_num",       id );

//    additional checks
        for (int d=0; d<3; d++) {
            if ( grid_dimensions[d] <= 0 )   YT_ABORT( "\"%s[%d]\" == %d <= 0 for grid [%ld]!\n", "grid_dimensions", d, grid_dimensions[d], id ); }
        if ( level < 0 )            YT_ABORT( "\"%s\" == %d < 0 for grid [%ld]!\n", "level", level, id );

        return YT_SUCCESS;

    } // METHOD : validate

}; // struct yt_grid



#endif // #ifndef __YT_TYPE_GRID_H__
