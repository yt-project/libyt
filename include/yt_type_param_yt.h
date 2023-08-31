#ifndef __YT_TYPE_PARAM_YT_H__
#define __YT_TYPE_PARAM_YT_H__



/*******************************************************************************
/
/  yt_param_yt structure
/
/  ==> included by yt_type.h
/
********************************************************************************/


// include relevant headers/prototypes
#include "yt_macro.h"
#include "yt_type_grid.h"
#include "yt_type_field.h"
#include "yt_type_particle.h"
void log_debug( const char *Format, ... );
void log_warning(const char *Format, ...);


//-------------------------------------------------------------------------------------------------------
// Structure   :  yt_param_yt
// Description :  Data structure to store YT-specific parameters (e.g., *Dataset.periodicity)
// 
// Notes       :  1. We assume that each element in array_name[3] are all in use.
//
// Data Member :  frontend                : Name of the target simulation code
//                fig_basename            : Base name of the output figures
//                domain_left_edge        : Simulation left edge in code units
//                domain_right_edge       : Simulation right edge in code units
//                dimensionality          : Dimensionality (1/2/3), this has nothing to do with array.
//                domain_dimensions       : Number of cells along each dimension on the root AMR level
//                periodicity             : Periodicity along each dimension (0/1 ==> No/Yes)
//                current_time            : Simulation time in code units
//                cosmological_simulation : Cosmological simulation dataset (0/1 ==> No/Yes)
//                current_redshift        : Redshift
//                omega_lambda            : Dark energy mass density
//                omega_matter            : Dark matter mass density
//                hubble_constant         : Dimensionless Hubble parameter at the present day
//                length_unit             : Simulation length unit in cm (CGS)
//                mass_unit               : Simulation mass   unit in g  (CGS)
//                time_unit               : Simulation time   unit in s  (CGS)
//                magnetic_unit           : Simulation magnetic unit in gauss
//                refine_by               : Refinement factor between a grid and its subgrid
//                index_offset            : Offset of the index.
//                num_grids               : Total number of grids
//                num_fields              : Number of fields
//                num_par_types           : Number of particle types, initialized as 0.
//                num_grids_local         : Number of local grids in each rank
//                par_type_list           : particle type list, including only {par_type, num_attr},
//                                          elements here will be copied into particle_list. This is because
//                                          we want libyt to handle initialization of particle_list.
//
// Method      :  yt_param_yt : Constructor
//                validate    : Check if all data members have been set properly by users
//                show        : Print out all data members
//-------------------------------------------------------------------------------------------------------
struct yt_param_yt
{
   const char *frontend;
   const char *fig_basename;

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
   double magnetic_unit;
   int    periodicity[3];
   int    cosmological_simulation;
   int    dimensionality;
   int    domain_dimensions[3];
   int    refine_by;
   int    index_offset;
   long   num_grids;
   int           num_fields;
   int           num_par_types;
   yt_par_type  *par_type_list;
   int           num_grids_local;

#ifdef __cplusplus
   //===================================================================================
   // Method      :  yt_param_yt
   // Description :  Constructor of the structure "yt_param_yt"
   //
   // Note        :  Initialize all data members
   //
   // Parameter   :  None
   //===================================================================================
   yt_param_yt()
   {
       frontend      = NULL;
       fig_basename  = NULL;
       for (int d=0; d<3; d++)
       {
           domain_left_edge [d] = DBL_UNDEFINED;
           domain_right_edge[d] = DBL_UNDEFINED;
       }
       current_time            = DBL_UNDEFINED;
       current_redshift        = DBL_UNDEFINED;
       omega_lambda            = DBL_UNDEFINED;
       omega_matter            = DBL_UNDEFINED;
       hubble_constant         = DBL_UNDEFINED;
       length_unit             = DBL_UNDEFINED;
       mass_unit               = DBL_UNDEFINED;
       time_unit               = DBL_UNDEFINED;
       magnetic_unit           = DBL_UNDEFINED;

       for (int d=0; d<3; d++)
       {
           periodicity      [d] = INT_UNDEFINED;
           domain_dimensions[d] = INT_UNDEFINED;
       }
       cosmological_simulation  = INT_UNDEFINED;
       dimensionality           = INT_UNDEFINED;
       refine_by                = INT_UNDEFINED;
       index_offset             = 0;
       num_grids                = LNG_UNDEFINED;

       num_fields              = 0;
       num_par_types           = 0;
       num_grids_local         = 0;
       par_type_list           = NULL;
   }
#endif // #ifdef __cplusplus

#ifdef __cplusplus
   //===================================================================================
   // Method      :  validate
   // Description :  Check if all data members have been set properly by users
   //
   // Note        :  1. Cosmological parameters are checked only if cosmological_simulation == 1
   //                2. Only check if data are used, does not alter them.
   //                3. TODO: (C-library) This method should be moved to elsewhere.
   //
   // Parameter   :  None
   //
   // Return      :  YT_SUCCESS or YT_FAIL
   //===================================================================================
   int validate() const
   {

      if ( frontend                == NULL          )   YT_ABORT( "\"%s\" has not been set!\n",     "frontend" );
      for (int d=0; d<3; d++) {
      if ( domain_left_edge [d]    == DBL_UNDEFINED )   YT_ABORT( "\"%s[%d]\" has not been set!\n", "domain_left_edge",  d );
      if ( domain_right_edge[d]    == DBL_UNDEFINED )   YT_ABORT( "\"%s[%d]\" has not been set!\n", "domain_right_edge", d ); }
      if ( current_time            == DBL_UNDEFINED )   YT_ABORT( "\"%s\" has not been set!\n",     "current_time" );
      if ( cosmological_simulation == INT_UNDEFINED )   YT_ABORT( "\"%s\" has not been set!\n",     "cosmological_simulation" );
      if ( cosmological_simulation ) {
      if ( current_redshift        == DBL_UNDEFINED )   YT_ABORT( "\"%s\" has not been set!\n",     "current_redshift" );
      if ( omega_lambda            == DBL_UNDEFINED )   YT_ABORT( "\"%s\" has not been set!\n",     "omega_lambda" );
      if ( omega_matter            == DBL_UNDEFINED )   YT_ABORT( "\"%s\" has not been set!\n",     "omega_matter" );
      if ( hubble_constant         == DBL_UNDEFINED )   YT_ABORT( "\"%s\" has not been set!\n",     "hubble_constant" ); }
      if ( length_unit             == DBL_UNDEFINED )   YT_ABORT( "\"%s\" has not been set!\n",     "length_unit" );
      if ( mass_unit               == DBL_UNDEFINED )   YT_ABORT( "\"%s\" has not been set!\n",     "mass_unit" );
      if ( time_unit               == DBL_UNDEFINED )   YT_ABORT( "\"%s\" has not been set!\n",     "time_unit" );
      if ( magnetic_unit           == DBL_UNDEFINED )   log_warning( "\"%s\" has not been set!\n",  "magnetic_unit" );
      
      for (int d=0; d<3; d++) {
      if ( periodicity      [d]    == INT_UNDEFINED )   YT_ABORT( "\"%s[%d]\" has not been set!\n", "periodicity", d );
      if ( domain_dimensions[d]    == INT_UNDEFINED )   YT_ABORT( "\"%s[%d]\" has not been set!\n", "domain_dimensions", d ); }
      if ( dimensionality          == INT_UNDEFINED )   YT_ABORT( "\"%s\" has not been set!\n",     "dimensionality" );
      if ( refine_by               == INT_UNDEFINED )   YT_ABORT( "\"%s\" has not been set!\n",     "refine_by" );
      if ( num_grids               == LNG_UNDEFINED )   YT_ABORT( "\"%s\" has not been set!\n",     "num_grids" );
      if ( num_par_types > 0 && par_type_list == NULL  )   YT_ABORT( "Particle species info par_type_list has not been set!\n");
      if ( num_par_types < 0 && par_type_list != NULL  )   YT_ABORT( "Particle species info num_par_types has not been set!\n");
      for (int s=0; s<num_par_types; s++) {
      if ( par_type_list[s].par_type == NULL || par_type_list[s].num_attr < 0 ) YT_ABORT( "par_type_list element [ %d ] is not set properly!\n", s);
      }

      return YT_SUCCESS;
   }

   //===================================================================================
   // Method      :  show
   // Description :  Print out all data members if the verbose level >= YT_VERBOSE_DEBUG
   //
   // Note        :  None
   //
   // Parameter   :  None
   //
   // Return      :  always YT_SUCCESS
   //===================================================================================
   int show() const
   {
      if ( validate() != YT_SUCCESS ) {
         YT_ABORT("yt_param_yt has not been set correctly.\n");
      }

      const int width_scalar = 25;
      const int width_vector = width_scalar - 3;

      log_debug( "   %-*s = %s\n",         width_scalar, "frontend",                frontend                );
      log_debug( "   %-*s = %s\n",         width_scalar, "fig_basename",            fig_basename            );
      for (int d=0; d<3; d++) {
      log_debug( "   %-*s[%d] = %13.7e\n", width_vector, "domain_left_edge",  d,    domain_left_edge [d]    ); }
      for (int d=0; d<3; d++) {
      log_debug( "   %-*s[%d] = %13.7e\n", width_vector, "domain_right_edge", d,    domain_right_edge[d]    ); }
      log_debug( "   %-*s = %13.7e\n",     width_scalar, "current_time",            current_time            );
      log_debug( "   %-*s = %d\n",         width_scalar, "cosmological_simulation", cosmological_simulation );
      if ( cosmological_simulation ) {
      log_debug( "   %-*s = %13.7e\n",     width_scalar, "current_redshift",        current_redshift        );
      log_debug( "   %-*s = %13.7e\n",     width_scalar, "omega_lambda",            omega_lambda            );
      log_debug( "   %-*s = %13.7e\n",     width_scalar, "omega_matter",            omega_matter            );
      log_debug( "   %-*s = %13.7e\n",     width_scalar, "hubble_constant",         hubble_constant         ); }

      log_debug( "   %-*s = %13.7e\n",     width_scalar, "length_unit",             length_unit             );
      log_debug( "   %-*s = %13.7e\n",     width_scalar, "mass_unit",               mass_unit               );
      log_debug( "   %-*s = %13.7e\n",     width_scalar, "time_unit",               time_unit               );
      if ( magnetic_unit == DBL_UNDEFINED ){
         log_debug( "   %-*s = %s\n",      width_scalar, "magnetic_unit",           "NOT SET, and will be set to 1.");
      }
      else{
         log_debug( "   %-*s = %13.7e\n",     width_scalar, "magnetic_unit",        magnetic_unit           );
      }
      
      for (int d=0; d<3; d++) {
      log_debug( "   %-*s[%d] = %d\n",     width_vector, "periodicity", d,          periodicity[d]          ); }
      for (int d=0; d<3; d++) {
      log_debug( "   %-*s[%d] = %d\n",     width_vector, "domain_dimensions", d,    domain_dimensions[d]    ); }
      log_debug( "   %-*s = %d\n",         width_scalar, "dimensionality",          dimensionality          );
      log_debug( "   %-*s = %d\n",         width_scalar, "refine_by",               refine_by               );
      log_debug( "   %-*s = %d\n",         width_scalar, "index_offset",            index_offset            );
      log_debug( "   %-*s = %ld\n",        width_scalar, "num_grids",               num_grids               );
      
      log_debug( "   %-*s = %ld\n",        width_scalar, "num_fields",              num_fields              );
      log_debug( "   %-*s = %ld\n",        width_scalar, "num_par_types",           num_par_types           );
      for (int s=0; s<num_par_types; s++){
      log_debug( "   %-*s[%d] = (type=\"%s\", num_attr=%d)\n", width_vector, "par_type_list", s, par_type_list[s].par_type, par_type_list[s].num_attr);
      }
      log_debug( "   %-*s = %ld\n",        width_scalar, "num_grids_local",         num_grids_local         );

      return YT_SUCCESS;
   }
#endif // #ifdef __cplusplus

};

#endif // #ifndef __YT_TYPE_PARAM_YT_H__
