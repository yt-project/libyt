#ifndef LIBYT_PROJECT_INCLUDE_YT_TYPE_PARTICLE_H_
#define LIBYT_PROJECT_INCLUDE_YT_TYPE_PARTICLE_H_

#include "yt_type_array.h"

/**
 * \struct yt_par_type
 * \brief Data structure to store particle type names and their number of attributes.
 *
 * \rst
 * .. caution::
 *    For now, libyt only borrows the particle type ``par_type`` from simulation.
 *    The lifetime of ``par_type`` should cover the whole in situ process.
 * \endrst
 */
typedef struct yt_par_type {
  const char* par_type; /*!< Particle type name */
  int num_attr;         /*!< Number of attributes in this type */

#ifdef __cplusplus
  yt_par_type() {
    par_type = nullptr;
    num_attr = INT_UNDEFINED;
  }
#endif  // #ifdef __cplusplus

} yt_par_type;

/**
 * \struct yt_attribute
 * \brief Data structure to store particle attributes.
 * \details
 * 1. `attr_unit`, `attr_name_alias`, `attr_display_name`, are set corresponding to yt
 *    `( "name", ("units", ["alias1", "alias2"], "display_name"))`
 *
 * \rst
 * .. caution::
 *    The lifetime of ``attr_name`` should cover the whole in situ analysis process.
 *
 *    The lifetime of `attr_unit`, ``attr_name_alias``, ``attr_display_name`` should cover
 *    \ref yt_commit
 * \endrst
 */
typedef struct yt_attribute {
  const char* attr_name;         /*!< Attribute name */
  yt_dtype attr_dtype;           /*!< Attribute data type */
  const char* attr_unit;         /*!< Attribute unit */
  int num_attr_name_alias;       /*!< Number of attribute name alias */
  const char** attr_name_alias;  /*!< A list of attribute name alias */
  const char* attr_display_name; /*!< Attribute display name */

#ifdef __cplusplus
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

//
// Data Member :  const char   *par_type  : Particle type.
//                int           num_attr  : Length of the attr_list.
//                yt_attribute *attr_list : Attribute list, contains a list of attributes
//                name, and
//                                          function get_par_attr knows how to get these
//                                          data.
//                const char   *coor_x    : Attribute name of coordinate x.
//                const char   *coor_y    : Attribute name of coordinate y.
//                const char   *coor_z    : Attribute name of coordinate z.
//
//                (func ptr) get_par_attr : pointer to function with input arguments
//                                          (const int, const long*, const char*, const
//                                          char*, yt_array*) that gets particle
//                                          attribute.
//
// Method      :  yt_particle  : Constructor
//-------------------------------------------------------------------------------------------------------
/**
 * \struct yt_particle
 * \brief Data structure to store particle info and get particle function.
 *
 * \rst
 * .. caution::
 *    libyt only borrows the particle type ``par_type`` from simulation. The lifetime of
 *    ``par_type`` should cover the whole in situ process.
 * \endrst
 */
typedef struct yt_particle {
  const char* par_type;    /*!< Particle type */
  int num_attr;            /*!< Number of attributes */
  yt_attribute* attr_list; /*!< Attribute list */
  const char* coor_x;      /*!< Attribute name of coordinate x */
  const char* coor_y;      /*!< Attribute name of coordinate y */
  const char* coor_z;      /*!< Attribute name of coordinate z */

  /** Get particle function */
  void (*get_par_attr)(const int, const long*, const char*, const char*, yt_array*);

#ifdef __cplusplus
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

#endif  // LIBYT_PROJECT_INCLUDE_YT_TYPE_PARTICLE_H_
