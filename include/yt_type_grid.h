#ifndef LIBYT_PROJECT_INCLUDE_YT_TYPE_GRID_H_
#define LIBYT_PROJECT_INCLUDE_YT_TYPE_GRID_H_

#include "yt_macro.h"

/**
 * \struct yt_data
 * \brief Data structure to store a field and particle data's pointer and its array
 * dimensions.
 * \details
 * 1. This struct will be used in \ref yt_grid data member `field_data` and
 *    `particle_data`.
 * 2. If `data_dtype` is set as `YT_DTYPE_UNKNOWN`, then libyt will use `field_dtype`
 *    defined in \ref yt_field as input field data type.
 */
typedef struct yt_data {
  void* data_ptr;         /*!< Data pointer */
  int data_dimensions[3]; /*!< Dimension of the data to be passed to Python for wrapping,
                           *   which is the actual dimension of the array in its pov. */
  yt_dtype data_dtype;    /*!< Data type */

#ifdef __cplusplus
  yt_data() {
    data_ptr = nullptr;
    for (int d = 0; d < 3; d++) {
      data_dimensions[d] = 0;
    }
    data_dtype = YT_DTYPE_UNKNOWN;
  }
#endif  // #ifdef __cplusplus
} yt_data;

/**
 * \struct yt_grid
 * \brief Data structure to store a full single AMR grid with data pointers
 *
 * \rst
 * .. note::
 *
 *    Only support dim 3 AMR grid for now.
 * \endrst
 */
typedef struct yt_grid {
  double left_edge[3];     /*!< Grid left edge in code units */
  double right_edge[3];    /*!< Grid right edge in code units */
  long* par_count_list;    /*!< Array that records number of particles in each species,
                            *   the input order should be the same as the input
                            *   `particle_list` */
  long id;                 /*!< Grid id */
  long parent_id;          /*!< Parent grid id (if no parent grid, set to `-1`) */
  int grid_dimensions[3];  /*!< Number of cells along each direction in `[x][y][z]` */
  int level;               /*!< AMR level (0 for the root level) */
  int proc_num;            /*!< Process number, grid belongs to which MPI rank */
  yt_data* field_data;     /*!< Each element stores an info of field data to be wrapped */
  yt_data** particle_data; /*!< Ex: `particle_data[0][1]` represents particle data for
                            * `particle_list[0].attr_list[1]` */

#ifdef __cplusplus
  yt_grid() {
    for (int d = 0; d < 3; d++) {
      left_edge[d] = DBL_UNDEFINED;
      right_edge[d] = DBL_UNDEFINED;
    }
    for (int d = 0; d < 3; d++) {
      grid_dimensions[d] = INT_UNDEFINED;
    }
    par_count_list = nullptr;
    id = LNG_UNDEFINED;
    parent_id = LNG_UNDEFINED;
    level = INT_UNDEFINED;
    proc_num = INT_UNDEFINED;
    field_data = nullptr;
    particle_data = nullptr;
  }
#endif  // #ifdef __cplusplus

} yt_grid;

#endif  // LIBYT_PROJECT_INCLUDE_YT_TYPE_GRID_H_
