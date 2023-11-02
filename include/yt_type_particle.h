#ifndef __YT_TYPE_PARTICLE_H__
#define __YT_TYPE_PARTICLE_H__

#include "yt_type_array.h"

//-------------------------------------------------------------------------------------------------------
// Structure   :  yt_par_type
// Description :  Data structure to store each species names and their number of attributes.
//
// Notes       :  1. Some data are overlap with yt_particle. We need this first be input by user through
//                   yt_set_Parameters(), so that we can set up and initialize particle_list properly.
//                2. For now, libyt only borrows the particle type par_type from simulation. The lifetime
//                   of par_type should cover the whole in situ process.
//
// Data Member :  const char  *par_type  : Particle type name (ptype in yt-term).
//                int          num_attr  : Number of attributes in this species.
//-------------------------------------------------------------------------------------------------------
typedef struct yt_par_type {
    const char* par_type;
    int num_attr;

#ifdef __cplusplus
    yt_par_type() {
        par_type = nullptr;
        num_attr = INT_UNDEFINED;
    }
#endif  // #ifdef __cplusplus

} yt_par_type;

//-------------------------------------------------------------------------------------------------------
// Structure   :  yt_attribute
// Description :  Data structure to store particle attributes.
//
// Notes       :  1. The lifetime of attr_name should cover the whole in situ analysis process.
//                2. The lifetime of attr_unit, attr_name_alias, attr_display_name should cover yt_commit
//                3. "attr_unit", "attr_name_alias", "attr_display_name", are set corresponding to yt
//                   ( "name", ("units", ["alias1", "alias2"], "display_name"))
//
// Data Member :  const char   *attr_name           : Particle label name, which in yt, it is its attribute.
//                yt_dtype      attr_dtype          : Attribute's data type. Should be yt_dtype.
//                const char   *attr_unit           : Set attr_unit if needed, if not set, it will search
//                                                    for XXXFieldInfo. Where XXX is set by g_param_yt.frontend.
//                int           num_attr_name_alias : Set attribute name to alias, number of the aliases.
//                const char  **attr_name_alias     : Aliases.
//                const char   *attr_display_name   : Set display name on the plottings, if not set, yt will
//                                                    use attr_name as display name.
//
// Method      :  yt_attribute  : Constructor
//-------------------------------------------------------------------------------------------------------
typedef struct yt_attribute {
    const char* attr_name;
    yt_dtype attr_dtype;
    const char* attr_unit;
    int num_attr_name_alias;
    const char** attr_name_alias;
    const char* attr_display_name;

#ifdef __cplusplus
    //=======================================================================================================
    // Method      : yt_attribute
    // Description : Constructor of the structure "yt_attribute"
    //
    // Note        : 1. Initialize attr_unit as "". If it is not set by user, then yt will use the particle
    //                  unit set by the frontend in yt_set_Parameters(). If there still isn't one, then it
    //                  will use "".
    //               2. Initialize attr_dtype as YT_DTYPE_UNKNOWN.
    //
    // Parameter   : None
    // ======================================================================================================
    yt_attribute() {
        attr_name = nullptr;
        attr_dtype = YT_DTYPE_UNKNOWN;
        attr_unit = "";
        num_attr_name_alias = 0;
        attr_name_alias = nullptr;
        attr_display_name = nullptr;
    }
#endif  // #ifdef __cplusplus

} yt_attribute;

//-------------------------------------------------------------------------------------------------------
// Structure   :  yt_particle
// Description :  Data structure to store particle info and function to get them.
//
// Notes       :  1. Particle type is "par_type", which is "ptype" in YT-term.
//                2. For now, libyt only borrows the particle type par_type from simulation. The lifetime
//                   of par_type should cover the whole in situ process.
//                3. attr_list must only contain attributes that can get by get_par_attr.
//
// Data Member :  const char   *par_type  : Particle type.
//                int           num_attr  : Length of the attr_list.
//                yt_attribute *attr_list : Attribute list, contains a list of attributes name, and
//                                          function get_par_attr knows how to get these data.
//                const char   *coor_x    : Attribute name of coordinate x.
//                const char   *coor_y    : Attribute name of coordinate y.
//                const char   *coor_z    : Attribute name of coordinate z.
//
//                (func ptr) get_par_attr : pointer to function with input arguments
//                                          (const int, const long*, const char*, const char*, yt_array*)
//                                          that gets particle attribute.
//
// Method      :  yt_particle  : Constructor
//-------------------------------------------------------------------------------------------------------
typedef struct yt_particle {
    const char* par_type;
    int num_attr;
    yt_attribute* attr_list;
    const char* coor_x;
    const char* coor_y;
    const char* coor_z;
    void (*get_par_attr)(const int, const long*, const char*, const char*, yt_array*);

#ifdef __cplusplus
    //=======================================================================================================
    // Method      : yt_particle
    // Description : Constructor of the structure "yt_particle"
    //
    // Note        : 1. Used in yt_set_Parameters.cpp
    //
    // Parameter   : None
    // ======================================================================================================
    yt_particle() {
        par_type = nullptr;
        num_attr = INT_UNDEFINED;
        attr_list = nullptr;
        coor_x = nullptr;
        coor_y = nullptr;
        coor_z = nullptr;
        get_par_attr = nullptr;
    }
#endif  // #ifdef __cplusplus

} yt_particle;

#endif  // #ifndef __YT_TYPE_PARTICLE_H__
