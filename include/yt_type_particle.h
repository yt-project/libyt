#ifndef __YT_TYPE_PARTICLE_H__
#define __YT_TYPE_PARTICLE_H__

/*******************************************************************************
/
/  yt_par_type, yt_attribute, and yt_particle structure
/
/  ==> included by yt_type.h
/
********************************************************************************/

#include <string.h>
#include "yt_type_array.h"
void log_debug( const char *Format, ... );
void log_warning(const char *Format, ...);

//-------------------------------------------------------------------------------------------------------
// Structure   :  yt_par_type
// Description :  Data structure to store each species names and their number of attributes.
// 
// Notes       :  1. Some data are overlap with yt_particle. We need this first be input by user through
//                   yt_set_parameter(), so that we can set up and initialize particle_list properly.
//
// Data Member :  char  *par_type  : Particle type name (ptype in yt-term).
//                int    num_attr      : Number of attributes in this species.
//-------------------------------------------------------------------------------------------------------
struct yt_par_type
{
	char *par_type;
	int   num_attr;

	yt_par_type()
	{
		par_type = NULL;
		num_attr     = INT_UNDEFINED;
	}
};

//-------------------------------------------------------------------------------------------------------
// Structure   :  yt_attribute
// Description :  Data structure to store particle attributes.
// 
// Notes       :  1. "attr_unit", "attr_name_alias", "attr_display_name", are set corresponding to yt 
//                   ( "name", ("units", ["alias1", "alias2"], "display_name"))
//
// Data Member :  char     *attr_name             : Particle label name, which in yt, it is its attribute.
//                yt_dtype  attr_dtype            : Attribute's data type. Should be yt_dtype.
//                char     *attr_unit             : Set attr_unit if needed, if not set, it will search 
//                                               for XXXFieldInfo. Where XXX is set by g_param_yt.frontend.
//                int       num_attr_name_alias   : Set attribute name to alias, number of the aliases.
//                char    **attr_name_alias       : Aliases.
//                char     *attr_display_name     : Set display name on the plottings, if not set, yt will 
//                                                  use attr_name as display name.
//
// Method      :  yt_attribute  : Constructor
//               ~yt_attribute  : Destructor
//                show          : Print data member value
//                validate      : Validate data member in struct
//-------------------------------------------------------------------------------------------------------
struct yt_attribute
{
	char     *attr_name;
	yt_dtype  attr_dtype;
	char     *attr_unit;
	int       num_attr_name_alias;
	char    **attr_name_alias;
	char     *attr_display_name;


//=======================================================================================================
// Method      : yt_attribute
// Description : Constructor of the structure "yt_attribute"
// 
// Note        : 1. Initialize attr_unit as "". If it is not set by user, then yt will use the particle 
//                  unit set by the frontend in yt_set_parameter(). If there still isn't one, then it 
//                  will use "". 
//               2. Initialize attr_dtype as YT_DOUBLE.
// 
// Parameter   : None
// ======================================================================================================
	yt_attribute()
	{
		attr_name = NULL;
		attr_dtype = YT_DOUBLE;
		attr_unit = "";
		num_attr_name_alias = 0;
		attr_name_alias = NULL;
		attr_display_name = NULL;
	} // METHOD : yt_attribute


//=======================================================================================================
// Method      : show
// Description : Print out all data members
// 
// Note        : 1. Print out "NOT SET" if the pointers are NULL.
// 
// Parameter   : None
// ======================================================================================================
   	int show() const {
    	// TODO: Pretty Print, show the yt_attribute
      
      	return YT_SUCCESS;
   	}


//=======================================================================================================
// Method      : validate
// Description : Validate data member in the struct.
// 
// Note        : 1. Validate data member value in one yt_attribute struct.
//                  (1) attr_name is set, and != NULL.
//                  (2) attr_dtype is one of yt_dtype.
// 
// Parameter   : None
// ======================================================================================================
   	int validate() const {
   		// attr_name is set
   		if ( attr_name == NULL ){
   			YT_ABORT("attr_name is not set!\n");
   		}

   		// attr_dtype is one of yt_dtype
   		bool valid = false;
   		for ( int yt_dtypeInt = YT_FLOAT; yt_dtypeInt < YT_DTYPE_UNKNOWN; yt_dtypeInt++ ){
   			yt_dtype dtype = static_cast<yt_dtype>(yt_dtypeInt);
   			if ( attr_dtype == dtype ){
   				valid = true;
   				break;
   			}
   		}
   		if ( valid == false ){
   			YT_ABORT("In attr [%s], unknown attr_dtype!\n", attr_name);
   		}

   		return YT_SUCCESS;
   	}


//=======================================================================================================
// Method      : ~yt_attribute
// Description : Destructor of the structure "yt_attribute"
// 
// Note        : 1. Not used currently
// 
// Parameter   : None
//=======================================================================================================
   	~yt_attribute()
   	{

   	}
};


//-------------------------------------------------------------------------------------------------------
// Structure   :  yt_particle
// Description :  Data structure to store particle info and function to get them.
// 
// Notes       :  1. Particle type is "par_type", which is "ptype" in YT-term.
//                2. attr_list must only contain attributes that can get by get_attr.
//
// Data Member :  char         *par_type  : Particle type.
//                int           num_attr      : Length of the attr_list.
//                yt_attribute *attr_list     : Attribute list, contains a list of attributes name, and 
//                                              function pointer get_attr knows how to get these data.
//                char         *coor_x        : Attribute name of coordinate x.
//                char         *coor_y        : Attribute name of coordinate y.
//                char         *coor_z        : Attribute name of coordinate z.
//                
//                (func pointer) get_attr     : pointer to function with arguments (int, long*, char*, yt_array*)
//                                              that gets particle attribute.
//
// Method      :  yt_particle  : Constructor
//               ~yt_particle  : Destructor
//                show         : Print data member value
//                validate     : Validate data member in struct
//-------------------------------------------------------------------------------------------------------
struct yt_particle
{
// data members
// ======================================================================================================
	char         *par_type;
	int           num_attr;
	yt_attribute *attr_list;

	char         *coor_x;
	char         *coor_y;
	char         *coor_z;

	void        (*get_attr) (int, long*, char*, yt_array*);


//=======================================================================================================
// Method      : yt_particle
// Description : Constructor of the structure "yt_particle"
// 
// Note        : 1. 
// 
// Parameter   : None
// ======================================================================================================
	yt_particle()
	{
		par_type = NULL;
		num_attr = INT_UNDEFINED;
		attr_list = NULL;

		coor_x = NULL;
		coor_y = NULL;
		coor_z = NULL;

		get_attr = NULL;
	}


//=======================================================================================================
// Method      : show
// Description : Print out all data members
// 
// Note        : 1. Print out "NOT SET" if the pointers are NULL.
// 
// Parameter   : None
// ======================================================================================================
   	int show() const {
    	// TODO: Pretty Print, show the yt_particle
      
      	return YT_SUCCESS;
   	}


//=======================================================================================================
// Method      : validate
// Description : Validate data member in the struct.
// 
// Note        : 1. Validate data member value in one yt_particle struct.
//                  (1) par_type is set != NULL
//                  (2) attr_list is set != NULL
//                  (3) num_attr should > 0
//                  (4) attr_name in attr_list should be unique 
//                  (5) call yt_attribute validate for each attr_list elements.
//                  (6) raise log_warning if coor_x, coor_y, coor_z is not set.
//                  (7) raise log_warning if get_attr not set.
//               2. Used inside check_particle_list().
// 
// Parameter   : None
// ======================================================================================================
   	int validate() const {
   		// par_type should be set
   		if ( par_type == NULL ){
   			YT_ABORT("par_type is not set!\n");
   		}

   		// attr_list != NULL
   		if ( attr_list == NULL ){
   			YT_ABORT("Particle type [ %s ], attr_list not set properly!\n", par_type);
   		}
		// num_attr should > 0
   		if ( num_attr < 0 ){
   			YT_ABORT("Particle type [ %s ], num_attr not set properly!\n", par_type);
   		}
   		
   		// call yt_attribute validate for each attr_list elements.
   		for ( int i = 0; i < num_attr; i++ ){
   			if ( !(attr_list[i].validate()) ){
   				YT_ABORT("Particle type [ %s ], attr_list element [ %d ] not set properly!\n", par_type, i);
   			}
   		}

   		// attr_name in attr_list should be unique
   		for ( int i = 0; i < num_attr; i++ ){
   			for ( int j = i+1; j < num_attr; j++ ){
   				if ( strcmp(attr_list[i].attr_name, attr_list[j].attr_name) == 0 ){
   					YT_ABORT("Particle type [ %s ], attr_list element [ %d ] and [ %d ] have same attr_name, expect them to be unique!\n",
   						      par_type, i, j);
   				}
   			}
   		}

   		// if didn't input coor_x/y/z, yt cannot function properly for this particle.
   		if ( coor_x == NULL ){
   			YT_ABORT("Particle type [ %s ], attribute name of coordinate x coor_x not set!\n", par_type);
   		}
   		if ( coor_y == NULL ){
   			YT_ABORT("Particle type [ %s ], attribute name of coordinate y coor_y not set!\n", par_type);
   		}
   		if ( coor_z == NULL ){
   			YT_ABORT("Particle type [ %s ], attribute name of coordinate z coor_z not set!\n", par_type);
   		}

   		// if didn't input get_attr, yt cannot function properly for this particle.
   		if ( get_attr == NULL ){
   			YT_ABORT("Particle type [ %s ], function that gets particle attribute get_attr not set!\n", par_type);
   		}

      	return YT_SUCCESS;
   	}


//=======================================================================================================
// Method      : ~yt_particle
// Description : Destructor of the structure "yt_particle"
// 
// Note        : 1. Not used currently
// 
// Parameter   : None
//=======================================================================================================
	~yt_particle()
	{

	} // METHOD : ~yt_particle

};

#endif // #ifndef __YT_TYPE_PARTICLE_H__
