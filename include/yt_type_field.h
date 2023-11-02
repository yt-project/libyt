#ifndef __YT_TYPE_FIELD_H__
#define __YT_TYPE_FIELD_H__

#ifndef __cplusplus
#include <stdbool.h>
#endif
#include "yt_type_array.h"

//-------------------------------------------------------------------------------------------------------
// Structure   :  yt_field
// Description :  Data structure to store a field's label and its definition of data representation.
//
// Notes       :  1. The data representation type will be initialized as "cell-centered".
//                2. The lifetime of field_name and field_type should cover the whole in situ process.
//                3. The lifetime of field_unit, field_name_alias, field_display_name should cover yt_commit.
//                3. "field_unit", "field_name_alias", "field_display_name", are set corresponding to yt
//                   ( "name", ("units", ["fields", "to", "alias"], "display_name"))
//
// Data Member :  const char   *field_name           : Field name
//                const char   *field_type           : Define type, for now, we have these types,
//                                                     (1) "cell-centered"
//                                                     (2) "face-centered"
//                                                     (3) "derived_func"
//                yt_dtype      field_dtype          : Field type of the grid.
//                bool          contiguous_in_x      : true  ==> [z][y][x], x address alter-first, default value.
//                                                     false ==> [x][y][z], z address alter-first
//                short         field_ghost_cell[6]  : Number of cell to ignore at the beginning and the end of each
//                                                     dimension.
//                                                     The dimensions are in the point of view of the field data, it has
//                                                     nothing to do with x, y, z coordinates.
//
//                const char   *field_unit           : Set field_unit if needed.
//                int           num_field_name_alias : Set field to alias names, number of the aliases.
//                const char  **field_name_alias     : Aliases.
//                const char   *field_display_name   : Set display name on the figure, if not set, yt will use field
//                                                     name as display name.
//
//                (func pointer) derived_func        : pointer to function that has prototype
//                                                     void (const int, const long*, const char*, yt_array*).
//
// Method      :  yt_field  : Constructor
//-------------------------------------------------------------------------------------------------------
typedef struct yt_field {
    const char* field_name;
    const char* field_type;
    yt_dtype field_dtype;
    bool contiguous_in_x;
    short field_ghost_cell[6];
    const char* field_unit;
    int num_field_name_alias;
    const char** field_name_alias;
    const char* field_display_name;
    void (*derived_func)(const int, const long*, const char*, yt_array*);

#ifdef __cplusplus
    //=======================================================================================================
    // Method      : yt_field
    // Description : Constructor of the structure "yt_field"
    //
    // Note        : 1. Initialize field_type as "cell-centered"
    //               2. Initialize field_unit as "". If it is not set by user, then yt will use the particle
    //                  unit set at yt frontend. If there still isn't one, then it will use "".
    //
    // Parameter   : None
    // ======================================================================================================
    yt_field() {
        field_name = nullptr;
        field_type = "cell-centered";
        field_dtype = YT_DTYPE_UNKNOWN;
        contiguous_in_x = true;
        for (int d = 0; d < 6; d++) {
            field_ghost_cell[d] = 0;
        }
        field_unit = "";
        num_field_name_alias = 0;
        field_name_alias = nullptr;
        field_display_name = nullptr;
        derived_func = nullptr;
    }
#endif  // #ifdef __cplusplus

} yt_field;

#endif  // #ifndef __YT_TYPE_FIELD_H__
