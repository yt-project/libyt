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
void log_debug( const char *Format, ... );
void log_warning(const char *Format, ...);

//-------------------------------------------------------------------------------------------------------
// Structure   :  yt_field
// Description :  Data structure to store a field's label and its definition of data representation.
// 
// Notes       :  1. The data representation type will be initialize as "cell-centered".
//                2. "field_unit", "field_name_alias", "field_display_name", are set corresponding to yt 
//                   ( "name", ("units", ["fields", "to", "alias"], # "display_name"))
//
// Data Member :  char  *field_name           : Field name
//                char  *field_define_type    : Define type, for now, we have these types, define in 
//                                              validate():
//                                              (1) "cell-centered"
//                                              (2) "face-centered"
//                                              (3) "derived_func"
//                bool   swap_axes            : true  ==> [z][y][x], x address alter-first, default value.
//                                              false ==> [x][y][z], z address alter-first
//                char  *field_unit           : Set field_unit if needed.
//                int    num_field_name_alias : Set fields to alias, number of the aliases.
//                char **field_name_alias     : Aliases.
//                char  *field_display_name   : Set display name on the plottings, if not set, yt will 
//                                              use field_name as display name.
//
//                (func pointer) derived_func : pointer to function that has argument (int, double *)
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
	char  *field_name;
	char  *field_define_type;
	bool   swap_axes;
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
		field_name = NULL;
		field_define_type = "cell-centered";
		swap_axes = true;
		field_unit = "NOT SET";
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
//                  (2) field_define_type can only be : "cell-centered", "face-centered", "derived_func".
//                  (3) Raise warning if derived_func == NULL and field_define_type is set to "derived_func".
//               2. Used in yt_commit_grids()
// 
// Parameter   : None
// ======================================================================================================
   int validate() const {
   	// field name is set.
   	if ( field_name == NULL ){
   		YT_ABORT("field_name is not set!\n");
   		return YT_FAIL;
   	}

   	// field_define_type can only be : "cell-centered", "face-centered", "derived_func".
   	bool  check1 = false;
   	int   num_type = 3;
   	char *type[3]  = {"cell-centered", "face-centered", "derived_func"};
   	for ( int i = 0; i < num_type; i++ ){
   		if ( strcmp(field_define_type, type[i]) == 0 ) {
   			check1 = true;
   			break;
   		}
   	}
   	if ( check1 == false ){
   		YT_ABORT("In field [%s], unknown field_define_type [%s]!\n", field_name, field_define_type);
   		return YT_FAIL;
   	}

   	// Raise warning if derived_func == NULL and field_define_type is set to "derived_func".
   	if ( strcmp(field_define_type, "derived_func") == 0 && derived_func == NULL ){
   		log_warning("In field [%s], field_define_type == %s, but derived_func not set!\n", 
   			          field_name, field_define_type);
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