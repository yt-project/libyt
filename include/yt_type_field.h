#ifndef LIBYT_PROJECT_INCLUDE_YT_TYPE_FIELD_H_
#define LIBYT_PROJECT_INCLUDE_YT_TYPE_FIELD_H_

#ifndef __cplusplus
#include <stdbool.h>
#endif
#include "yt_type_array.h"

/**
 * \struct yt_field
 * \brief Data structure to store a field's label and its definition of data
 * representation.
 * \details
 * 1. `field_unit`, `field_name_alias`, `field_display_name`, are set corresponding to yt
 *    frontend `( "name", ("units", ["fields", "to", "alias"], "display_name"))`.
 *
 * \rst
 * .. caution::
 *    The lifetime of ``field_name`` and ``field_type`` should cover the whole in
 *    situ process.
 *
 *    The lifetime of ``field_unit``, ``field_name_alias``, ``field_display_name``
 *    should cover \ref yt_commit.
 * \endrst
 */
typedef struct yt_field {
  const char* field_name;         /*!< Field name */
  const char* field_type;         /*!< Define type
                                   *   (`"cell-centered"`, `"face-centered"`,
                                   *   `"derived_func"`) */
  yt_dtype field_dtype;           /*!< Field data type */
  bool contiguous_in_x;           /*!< true for x address alter-first (`[z][y][x]`);
                                   *   false for z address alter-first (`[x][y][z]`)*/
  short field_ghost_cell[6];      /*!< Number of cell to ignore at the beginning and the
                                   *   end of each dimension of a data pointer. */
  const char* field_unit;         /*!< Field unit */
  int num_field_name_alias;       /*!< Number of field name alias */
  const char** field_name_alias;  /*!< A list of field name alias */
  const char* field_display_name; /*!< Field display name */

  /** Derived function */
  void (*derived_func)(const int, const long*, const char*, yt_array*);

#ifdef __cplusplus
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

#endif  // LIBYT_PROJECT_INCLUDE_YT_TYPE_FIELD_H_
