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
// Notes       :  1. The data representation type will be initialize as cell-centered.
//
// Data Member :  
//
// Method      :  yt_field  : Constructor
//               ~yt_field  : Destructor
//                show      : Show the value
//-------------------------------------------------------------------------------------------------------
struct yt_field
{
// data members
// ======================================================================================================
	char *field_name;
	char *field_type;

//=======================================================================================================
// Method      : yt_field
// Description : Constructor of the structure "yt_field"
// 
// Note        : 1. Initialize field_type as "cell-centered"
// 
// Parameter   : None
// ======================================================================================================
	yt_field()
	{
		field_name = "NOT SET";
		field_type = "cell-centered";
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

//=======================================================================================================
// Method      : show
// Description : Print out all data members if the verbose level >= YT_VERBOSE_DEBUG
// 
// Note        : 1. Will be used in yt_type_param_yt.h (struct yt_param_yt show())
// 
// Parameter   : None
// 
// Return      : YT_SUCCESS
//=======================================================================================================
	int show() const
	{
		log_debug("(%s, %s)", field_name, field_type);
	}

};
