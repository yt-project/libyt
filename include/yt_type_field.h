#ifndef __YT_TYPE_FIELD_H__
#define __YT_TYPE_FIELD_H__

/*******************************************************************************
/
/  yt_field structure
/
/  ==> included by yt_type.h
/
********************************************************************************/

// include relevant headers/prototypes
#include <string.h>
#include "yt_type_array.h"
void log_debug( const char *Format, ... );
void log_warning(const char *Format, ...);

//-------------------------------------------------------------------------------------------------------
// Structure   :  yt_field
// Description :  Data structure to store a field's label and its definition of data representation.
// 
// Notes       :  1. The data representation type will be initialized as "cell-centered".
//                2. "field_unit", "field_name_alias", "field_display_name", are set corresponding to yt 
//                   ( "name", ("units", ["fields", "to", "alias"], "display_name"))
//
// Data Member :  char    *field_name           : Field name
//                char    *field_type           : Define type, for now, we have these types, define in
//                                                validate():
//                                                  (1) "cell-centered"
//                                                  (2) "face-centered"
//                                                  (3) "derived_func"
//                yt_dtype field_dtype          : Field type of the grid. Can be YT_FLOAT, YT_DOUBLE, YT_INT.
//                bool     contiguous_in_x      : true  ==> [z][y][x], x address alter-first, default value.
//                                                false ==> [x][y][z], z address alter-first
//                short    field_ghost_cell[6]  : Number of cell to ignore at the beginning and the end of each dimension.
//                                                The dimensions are in the point of view of the field data, it has
//                                                nothing to do with coordinates.
//
//                char    *field_unit           : Set field_unit if needed.
//                int      num_field_name_alias : Set fields to alias, number of the aliases.
//                char   **field_name_alias     : Aliases.
//                char    *field_display_name   : Set display name on the plottings, if not set, yt will 
//                                                use field_name as display name.
//
//                (func pointer) derived_func   : pointer to function that has argument
//                                                (const int, const long*, const char*, yt_array*) with no return.
//
// Method      :  yt_field  : Constructor
//               ~yt_field  : Destructor
//                show      : Print data member value
//                validate  : Validate data member in struct
//-------------------------------------------------------------------------------------------------------
struct yt_field
{
// data members
// ======================================================================================================
	char     *field_name;
	char     *field_type;
	yt_dtype  field_dtype;
	bool      contiguous_in_x;
    short     field_ghost_cell[6];
	char     *field_unit;
	int       num_field_name_alias;
	char    **field_name_alias;
	char     *field_display_name;

	void (*derived_func) (const int, const long *, const char*, yt_array*);


//=======================================================================================================
// Method      : yt_field
// Description : Constructor of the structure "yt_field"
// 
// Note        : 1. Initialize field_type as "cell-centered"
//               2. Initialize field_unit as "". If it is not set by user, then yt will use the particle 
//                  unit set by the frontend in yt_set_parameter(). If there still isn't one, then it 
//                  will use "". 
// Parameter   : None
// ======================================================================================================
	yt_field()
	{
		field_name = NULL;
        field_type = "cell-centered";
		field_dtype = YT_DTYPE_UNKNOWN;
        contiguous_in_x = true;
        for(int d=0; d<6; d++){ field_ghost_cell[d] = 0; }
		field_unit = "";
		num_field_name_alias = 0;
		field_name_alias = NULL;
		field_display_name = NULL;

		derived_func = NULL;
	} // METHOD : yt_field

//=======================================================================================================
// Method      : show
// Description : Print out all data members
// 
// Note        : 1. Print out "NOT SET" if the pointers are NULL.
// 
// Parameter   : None
// ======================================================================================================
   int show() const {
   // TODO: Pretty Print, show the yt_field
      
      return YT_SUCCESS;
   }

//=======================================================================================================
// Method      : validate
// Description : Validate data member in the struct.
// 
// Note        : 1. Validate data member value in one yt_field struct.
//                  (1) field_name is set != NULL.
//                  (2) field_type can only be : "cell-centered", "face-centered", "derived_func".
//                  (3) Check if field_dtype is set.
//                  (4) Raise warning if derived_func == NULL and field_type is set to "derived_func".
//                  (5) field_ghost_cell cannot be smaller than 0.
//               2. Used in check_field_list()
// 
// Parameter   : None
// ======================================================================================================
    int validate() const {
        // field name is set.
        if ( field_name == NULL ){
            YT_ABORT("field_name is not set!\n");
        }

        // field_type can only be : "cell-centered", "face-centered", "derived_func".
        bool  check1 = false;
        int   num_type = 3;
        char *type[3]  = {"cell-centered", "face-centered", "derived_func"};
        for ( int i = 0; i < num_type; i++ ){
            if ( strcmp(field_type, type[i]) == 0 ) {
                check1 = true;
                break;
            }
        }
        if ( check1 == false ){
            YT_ABORT("In field [%s], unknown field_type [%s]!\n", field_name, field_type);
        }

        // if field_dtype is set.
        bool check2 = false;
        for ( int yt_dtypeInt = YT_FLOAT; yt_dtypeInt < YT_DTYPE_UNKNOWN; yt_dtypeInt++ ){
            yt_dtype dtype = static_cast<yt_dtype>(yt_dtypeInt);
            if ( field_dtype == dtype ){
                check2 = true;
                break;
            }
        }
        if ( check2 == false ){
            YT_ABORT("In field [%s], field_dtype not set!\n", field_name);
        }

        // Raise warning if derived_func == NULL and field_type is set to "derived_func".
        if ( strcmp(field_type, "derived_func") == 0 && derived_func == NULL ){
            YT_ABORT("In field [%s], field_type == %s, derived_func not set!\n",
                     field_name, field_type);
        }

        // field_ghost_cell cannot be smaller than 0.
        for ( int d = 0; d < 6; d++ ){
            if( field_ghost_cell[d] < 0 ){
                YT_ABORT("In field [%s], field_ghost_cell[%d] < 0. This parameter means number of cells to ignore and should be >= 0!\n",
                         field_name, d);
            }
        }
      
        return YT_SUCCESS;
    }


//=======================================================================================================
// Method      : ~yt_field
// Description : Destructor of the structure "yt_field"
// 
// Note        : 1. Not used currently
// 
// Parameter   : None
//=======================================================================================================
	~yt_field()
	{

	} // METHOD : ~yt_field

};

#endif // #ifndef __YT_TYPE_FIELD_H__
