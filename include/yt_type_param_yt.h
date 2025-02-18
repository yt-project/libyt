#ifndef LIBYT_PROJECT_INCLUDE_YT_TYPE_PARAM_YT_H_
#define LIBYT_PROJECT_INCLUDE_YT_TYPE_PARAM_YT_H_

#include "yt_macro.h"
#include "yt_type_particle.h"

//-------------------------------------------------------------------------------------------------------
// Structure   :  yt_param_yt
// Description :  Data structure to store YT-specific parameters
//
// Notes       :  1. We assume that each element in array_name[3] are all in use.
//                2. Included by yt_type.h.
//
// Data Member :  frontend                : Name of the target simulation code
//                fig_basename            : Base name of the output figures
//                domain_left_edge        : Simulation left edge in code units
//                domain_right_edge       : Simulation right edge in code units
//                dimensionality          : Dimensionality (1/2/3), this has nothing to do
//                with array. domain_dimensions       : Number of cells along each
//                dimension on the root AMR level periodicity             : Periodicity
//                along each dimension (0/1 ==> No/Yes) current_time            :
//                Simulation time in code units cosmological_simulation : Cosmological
//                simulation dataset (0/1 ==> No/Yes) current_redshift        : Redshift
//                omega_lambda            : Dark energy mass density
//                omega_matter            : Dark matter mass density
//                hubble_constant         : Dimensionless Hubble parameter at the present
//                day length_unit             : Simulation length unit in cm (CGS)
//                mass_unit               : Simulation mass   unit in g  (CGS)
//                time_unit               : Simulation time   unit in s  (CGS)
//                velocity_unit           : Simulation velocity unit in cm / s (CGS)
//                magnetic_unit           : Simulation magnetic unit in gauss
//                refine_by               : Refinement factor between a grid and its
//                subgrid index_offset            : Offset of the index. num_grids : Total
//                number of grids num_fields              : Number of fields num_par_types
//                : Number of particle types, initialized as 0. num_grids_local         :
//                Number of local grids in each rank par_type_list           : particle
//                type list, including only {par_type, num_attr},
//                                          elements here will be copied into
//                                          particle_list. This is because we want libyt
//                                          to handle initialization of particle_list.
//
// Method      :  yt_param_yt : Constructor
//-------------------------------------------------------------------------------------------------------
typedef struct yt_param_yt {
  const char* frontend;
  const char* fig_basename;
  double domain_left_edge[3];
  double domain_right_edge[3];
  double current_time;
  double current_redshift;
  double omega_lambda;
  double omega_matter;
  double hubble_constant;
  double length_unit;
  double mass_unit;
  double time_unit;
  double velocity_unit;
  double magnetic_unit;
  int periodicity[3];
  int cosmological_simulation;
  int dimensionality;
  int domain_dimensions[3];
  int refine_by;
  int index_offset;
  long num_grids;
  int num_fields;
  int num_par_types;
  yt_par_type* par_type_list;
  int num_grids_local;

#ifdef __cplusplus
  //===================================================================================
  // Method      :  yt_param_yt
  // Description :  Constructor of the structure "yt_param_yt"
  //
  // Note        :  Initialize all data members
  //
  // Parameter   :  None
  //===================================================================================
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
