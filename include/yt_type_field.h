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
void log_debug( const char *Format, ... );

//-------------------------------------------------------------------------------------------------------
// Structure   :  yt_field
// Description :  Data structure to store a field's label and its definition of data representation.
// 
// Notes       :  1. The data representation type will be initialize as "cell-centered".
//
// Data Member :  char *field_name        : Field name
//                char *field_define_type : Define type, ex: cell-centered
//
// Method      :  yt_field  : Constructor
//               ~yt_field  : Destructor
//-------------------------------------------------------------------------------------------------------
struct yt_field
{
// data members
// ======================================================================================================
	char *field_name;
	char *field_define_type;

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
	} // METHOD : yt_field

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