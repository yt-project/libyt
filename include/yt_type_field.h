#ifndef __YT_TYPE_FIELD_H__
#define __YT_TYPE_FIELD_H__

/*******************************************************************************
/
/  yt_field structure
/
/  ==> included by yt_type.h
/
********************************************************************************/

//-------------------------------------------------------------------------------------------------------
// Structure   :  yt_field
// Description :  Data structure to store a field's label and its definition of data representation.
// 
// Notes       :  1. The data representation type will be initialize as "cell-centered".
//                2. "field_dimension" is used for fields like MHD, they did not have the same dimension
//                   as in the other field, though they are in the same patch. This is used in append_grid.cpp.
//                3. "field_unit", "field_name_alias", "field_display_name", are set corresponding to yt 
//                   ( "name", ("units", ["fields", "to", "alias"], # "display_name"))
//
// Data Member :  char  *field_name           : Field name
//                char  *field_define_type    : Define type, for now, we have these types
//                                              (1) "cell-centered"
//                                              (2) "face-centered"
//                                              (3) "derived_func"
//                int    field_dimension[3]   : Field dimension, use to pass in array to yt, set as default 
//                                              value if undefined.
//                char  *field_unit           : Set field_unit if needed.
//                int    num_field_name_alias : Set fields to alias, number of the aliases.
//                char **field_name_alias     : Aliases.
//                char  *field_display_name   : Set display name on the plottings.
//
//                (func pointer) derived_func : pointer to function that has argument int, and double **
//
// Method      :  yt_field  : Constructor
//               ~yt_field  : Destructor
//-------------------------------------------------------------------------------------------------------
struct yt_field
{
// data members
// ======================================================================================================
	char  *field_name;
	char  *field_define_type;
	int    field_dimension[3];
	char  *field_unit;
	int    num_field_name_alias;
	char **field_name_alias;
	char  *field_display_name;

	void (*derived_func) (long, double *);


//=======================================================================================================
// Method      : yt_field
// Description : Constructor of the structure "yt_field"
// 
// Note        : 1. Initialize field_define_type as "cell-centered"
// 
// Parameter   : None
// ======================================================================================================
	yt_field()
	{
		field_name = "NOT SET";
		field_define_type = "cell-centered";
		for ( int d=0; d<3; d++ ){
			field_dimension[d] = INT_UNDEFINED;
		}
		field_unit = "NOT SET";
		num_field_name_alias = 0;
		field_name_alias = NULL;
		field_display_name = "NOT SET";

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