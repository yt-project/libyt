#ifndef __YT_TYPE_H__
#define __YT_TYPE_H__



/*******************************************************************************
/
/  Data types used by libyt
/
********************************************************************************/


// include relevant headers/prototypes
#include "yt_macro.h"
void log_debug( const char *Format, ... );


// short names for unsigned types
// ===================================================================================
typedef unsigned int       uint;
typedef unsigned long int  ulong;


// enumerate types
// ===================================================================================
enum yt_verbose { YT_VERBOSE_NONE=0, YT_VERBOSE_INFO=1, YT_VERBOSE_WARNING=2, YT_VERBOSE_DEBUG=3 };


// structures
// ===================================================================================

//-------------------------------------------------------------------------------------------------------
// Structure   :  yt_param_libyt
// Description :  Data structure of libyt runtime parameters
//
// Data Member :  verbose : Verbose level
//                script  : Name of the YT inline analysis script (without the .py extension)
//
// Method      :  yt_param_libyt : Constructor
//               ~yt_param_libyt : Destructor
//-------------------------------------------------------------------------------------------------------
struct yt_param_libyt
{

// data members
// ===================================================================================
   yt_verbose verbose;
   const char *script;


   //===================================================================================
   // Constructor :  yt_param_libyt
   // Description :  Constructor of the structure "yt_param_libyt"
   //
   // Note        :  Initialize all data members
   //
   // Parameter   :  None
   //===================================================================================
   yt_param_libyt()
   {

//    set defaults
      verbose = YT_VERBOSE_WARNING;
      script  = "yt_script.py";

   } // METHOD : yt_param_libyt


   //===================================================================================
   // Destructor  :  ~yt_param_libyt
   // Description :  Destructor of the structure "yt_param"_libyt
   //
   // Note        :  Free memory
   //===================================================================================
   ~yt_param_libyt()
   {

   } // METHOD : ~yt_param_libyt

}; // struct yt_param_libyt



//-------------------------------------------------------------------------------------------------------
// Structure   :  yt_param_yt
// Description :  Data structure to store YT-specific parameters (e.g., *Dataset.periodicity)
//
// Data Member :  domain_left_edge        : Simulation left edge in code units
//                domain_right_edge       : Simulation right edge in code units
//                dimensionality          : Dimensionality (1/2/3)
//                domain_dimensions       : Number of cells along each dimension on the root AMR level
//                periodicity             : Periodicity along each dimension (0/1 ==> No/Yes)
//                current_time            : Simulation time in code units
//                cosmological_simulation : Cosmological simulation dataset (0/1 ==> No/Yes)
//                current_redshift        : Redshift
//                omega_lambda            : Dark energy mass density
//                omega_matter            : Dark matter mass density
//                hubble_constant         : Dimensionless Hubble parameter at the present day
//                length_unit             : Simulation length unit in CGS
//                mass_unit               : Simulation mass   unit in CGS
//                time_unit               : Simulation time   unit in CGS
//
// Method      :  yt_param_yt : Constructor
//               ~yt_param_yt : Destructor
//                validate    : Check if all data members have been set properly by users
//                show        : Print out all data members
//-------------------------------------------------------------------------------------------------------
struct yt_param_yt
{

// data members
// ===================================================================================
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

// declare all boolean variables as int so that we can check whether they have been set by users
   int    periodicity[3];
   int    cosmological_simulation;
   int    dimensionality;
   int    domain_dimensions[3];


   //===================================================================================
   // Constructor :  yt_param_yt
   // Description :  Constructor of the structure "yt_param_yt"
   //
   // Note        :  Initialize all data members
   //
   // Parameter   :  None
   //===================================================================================
   yt_param_yt()
   {

//    set defaults
      for (int d=0; d<3; d++)
      {
         domain_left_edge [d] = FLT_UNDEFINED;
         domain_right_edge[d] = FLT_UNDEFINED;
      }
      current_time            = FLT_UNDEFINED;
      current_redshift        = FLT_UNDEFINED;
      omega_lambda            = FLT_UNDEFINED;
      omega_matter            = FLT_UNDEFINED;
      hubble_constant         = FLT_UNDEFINED;
      length_unit             = FLT_UNDEFINED;
      mass_unit               = FLT_UNDEFINED;
      time_unit               = FLT_UNDEFINED;

      for (int d=0; d<3; d++)
      {
         periodicity      [d] = INT_UNDEFINED;
         domain_dimensions[d] = INT_UNDEFINED;
      }
      cosmological_simulation = INT_UNDEFINED;
      dimensionality          = INT_UNDEFINED;

   } // METHOD : yt_param_yt


   //===================================================================================
   // Destructor  :  ~yt_param_yt
   // Description :  Destructor of the structure "yt_param"_yt
   //
   // Note        :  Free memory
   //
   // Parameter   :  None
   //===================================================================================
   ~yt_param_yt()
   {

   } // METHOD : ~yt_param_yt


   //===================================================================================
   // Destructor  :  validate
   // Description :  Check if all data members have been set properly by users
   //
   // Note        :  1. Cosmological parameters are checked only if cosmological_simulation == 1
   //
   // Parameter   :  None
   //
   // Return      :  YT_SUCCESS or YT_FAIL
   //===================================================================================
   int validate() const
   {

      for (int d=0; d<3; d++) {
      if ( domain_left_edge [d]    == FLT_UNDEFINED )   YT_ABORT( "\"%s[%d]\" has not been set!\n", "domain_left_edge",  d );
      if ( domain_right_edge[d]    == FLT_UNDEFINED )   YT_ABORT( "\"%s[%d]\" has not been set!\n", "domain_right_edge", d ); }
      if ( current_time            == FLT_UNDEFINED )   YT_ABORT( "\"%s\" has not been set!\n",     "current_time" );
      if ( cosmological_simulation == INT_UNDEFINED )   YT_ABORT( "\"%s\" has not been set!\n",     "cosmological_simulation" );
      if ( cosmological_simulation ) {
      if ( current_redshift        == FLT_UNDEFINED )   YT_ABORT( "\"%s\" has not been set!\n",     "current_redshift" );
      if ( omega_lambda            == FLT_UNDEFINED )   YT_ABORT( "\"%s\" has not been set!\n",     "omega_lambda" );
      if ( omega_matter            == FLT_UNDEFINED )   YT_ABORT( "\"%s\" has not been set!\n",     "omega_matter" );
      if ( hubble_constant         == FLT_UNDEFINED )   YT_ABORT( "\"%s\" has not been set!\n",     "hubble_constant" ); }
      if ( length_unit             == FLT_UNDEFINED )   YT_ABORT( "\"%s\" has not been set!\n",     "length_unit" );
      if ( mass_unit               == FLT_UNDEFINED )   YT_ABORT( "\"%s\" has not been set!\n",     "mass_unit" );
      if ( time_unit               == FLT_UNDEFINED )   YT_ABORT( "\"%s\" has not been set!\n",     "time_unit" );

      for (int d=0; d<3; d++) {
      if ( periodicity      [d]    == INT_UNDEFINED )   YT_ABORT( "\"%s[%d]\" has not been set!\n", "periodicity", d );
      if ( domain_dimensions[d]    == INT_UNDEFINED )   YT_ABORT( "\"%s[%d]\" has not been set!\n", "domain_dimensions", d ); }
      if ( dimensionality          == INT_UNDEFINED )   YT_ABORT( "\"%s\" has not been set!\n",     "dimensionality" );

      return YT_SUCCESS;

   } // METHOD : validate


   //===================================================================================
   // Destructor  :  show
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

      const int width_scalar = 25;
      const int width_vector = width_scalar - 3;

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
      for (int d=0; d<3; d++) {
      log_debug( "   %-*s[%d] = %d\n",     width_vector, "periodicity", d,          periodicity[d]          ); }
      for (int d=0; d<3; d++) {
      log_debug( "   %-*s[%d] = %d\n",     width_vector, "domain_dimensions", d,    domain_dimensions[d]    ); }
      log_debug( "   %-*s = %d\n",         width_scalar, "dimensionality",          dimensionality          );

      return YT_SUCCESS;

   } // METHOD : show

}; // struct yt_param_yt



#endif // #ifndef __YT_TYPE_H__
