#ifndef LIBYT_PROJECT_INCLUDE_YT_TYPE_ARRAY_H_
#define LIBYT_PROJECT_INCLUDE_YT_TYPE_ARRAY_H_

#include "yt_macro.h"

//-------------------------------------------------------------------------------------------------------
// Structure   :  yt_array
// Description :  Data structure to store derived fields and particle attributes generated
// by derived
//                functions or get particle attribute functions.
//
// Notes       :  1. This struct is used in yt_field data member derived_func.
//                2. This struct is used in yt_particle data member get_par_attr.
//
// Data Member : gid                : gid of the field.
//               data_length        : size of the data_ptr.
//               data_ptr           : data pointer, where function should fill in the
//               required data.
//
// Methods     : yt_array            : Constructor
//-------------------------------------------------------------------------------------------------------
typedef struct yt_array {
  long gid;
  long data_length;
  void* data_ptr;

#ifdef __cplusplus
  //===================================================================================
  // Method      :  yt_array
  // Description :  Constructor of the structure "yt_array"
  //
  // Note        :  Initialize all data members
  //
  // Parameter   :  None
  //===================================================================================
  yt_array() {
    gid = LNG_UNDEFINED;
    data_length = 0;
    data_ptr = nullptr;
  }
#endif  // #ifdef __cplusplus

} yt_array;

#endif  // LIBYT_PROJECT_INCLUDE_YT_TYPE_ARRAY_H_
