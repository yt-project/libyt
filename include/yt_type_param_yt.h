#ifndef LIBYT_PROJECT_INCLUDE_YT_TYPE_PARAM_YT_H_
#define LIBYT_PROJECT_INCLUDE_YT_TYPE_PARAM_YT_H_

#include "yt_macro.h"
#include "yt_type_particle.h"

/**
 * \struct yt_param_yt
 * \brief Data structure to store YT-specific parameters and parameters to initialize the
 * storage for AMR structure.
 *
 * \rst
 * .. note::
 *
 *    Only support dim 3 AMR grid for now.
 * \endrst
 */
typedef struct yt_param_yt {
  const char* frontend;        /*!< Name of the simulation code frontend in yt */
  const char* fig_basename;    /*!< Base name of the output figures */
  double domain_left_edge[3];  /*!< Left edge of the simulation domain in code units */
  double domain_right_edge[3]; /*!< Right edge of the simulation domain in code units */
  double current_time;         /*!< Simulation time in code units */
  double current_redshift;     /*!< Redshift */
  double omega_lambda;         /*!< Dark energy mass density */
  double omega_matter;         /*!< Dark matter mass density */
  double hubble_constant;      /*!< Dimensionless Hubble parameter at the present day */
  double length_unit;          /*!< Simulation length unit in cm (CGS unit) */
  double mass_unit;            /*!< Simulation mass unit in g (CGS unit) */
  double time_unit;            /*!< Simulation time unit in s (CGS unit) */
  double velocity_unit;        /*!< Simulation velocity unit in cm / s (CGS unit) */
  double magnetic_unit;        /*!< Simulation magnetic unit in gauss */
  int periodicity[3];          /*!< Periodicity along each x,y,z dimension.
                                *   (0/1 ==> No/Yes) */
  int cosmological_simulation; /*!< Cosmological simulation dataset (0/1 ==> No/Yes) */
  int dimensionality;          /*!< Dimensionality (1/2/3), only support 3. */
  int domain_dimensions[3];    /*!< Number of cells along each x,y,z dimension on the
                                *   root AMR level */
  int refine_by;               /*!< Refinement factor between a grid and its subgrid */
  int index_offset;            /*!< Offset of the index */
  long num_grids;              /*!< Total number of grids */
  int num_fields;              /*!< Number of fields */
  int num_par_types;           /*!< Number of particle types */
  yt_par_type* par_type_list;  /*!< Particle type list, the array has length
                                *   `num_par_types` */
  int num_grids_local;         /*!< Number of local grids in current rank */

#ifdef __cplusplus
  yt_param_yt() {
    frontend = nullptr;
    fig_basename = nullptr;
    for (int d = 0; d < 3; d++) {
      domain_left_edge[d] = DBL_UNDEFINED;
      domain_right_edge[d] = DBL_UNDEFINED;
    }
    current_time = DBL_UNDEFINED;
    current_redshift = DBL_UNDEFINED;
    omega_lambda = DBL_UNDEFINED;
    omega_matter = DBL_UNDEFINED;
    hubble_constant = DBL_UNDEFINED;
    length_unit = DBL_UNDEFINED;
    mass_unit = DBL_UNDEFINED;
    time_unit = DBL_UNDEFINED;
    velocity_unit = DBL_UNDEFINED;
    magnetic_unit = DBL_UNDEFINED;

    for (int d = 0; d < 3; d++) {
      periodicity[d] = INT_UNDEFINED;
      domain_dimensions[d] = INT_UNDEFINED;
    }
    cosmological_simulation = INT_UNDEFINED;
    dimensionality = INT_UNDEFINED;
    refine_by = INT_UNDEFINED;
    index_offset = 0;
    num_grids = LNG_UNDEFINED;

    num_fields = 0;
    num_par_types = 0;
    num_grids_local = 0;
    par_type_list = nullptr;
  }
#endif  // #ifdef __cplusplus

} yt_param_yt;

#endif  // LIBYT_PROJECT_INCLUDE_YT_TYPE_PARAM_YT_H_
