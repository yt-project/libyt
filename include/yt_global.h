#ifndef __YT_GLOBAL_H__
#define __YT_GLOBAL_H__



/*******************************************************************************
/
/  All libyt global variables are defined here
/
********************************************************************************/


// convenient macros for defining and declaring global variables
// ==> predefine DEFINE_GLOBAL in the file actually **defines** these global variables (e.g., yt_init.cpp)
// ==> there should be one and only one file that defines DEFINE_GLOBAL

// SET_GLOBAL will invoke SET_GLOBAL_INIT or SET_GLOBAL_NOINIT depending on the number of arguments
// ==> http://stackoverflow.com/questions/11761703/overloading-macro-on-number-of-arguments
#define GET_MACRO( _1, _2, _3, TARGET_MACRO, ... )   TARGET_MACRO
#define SET_GLOBAL( ... )   GET_MACRO( __VA_ARGS__, SET_GLOBAL_INIT, SET_GLOBAL_NOINIT ) ( __VA_ARGS__ )

// SET_GLOBAL_INIT/NOINIT are for global variables with/without initialization
#ifdef DEFINE_GLOBAL
# define SET_GLOBAL_INIT( type, name, init_value )   type name = init_value
# define SET_GLOBAL_NOINIT( type, name )             type name
#else
# define SET_GLOBAL_INIT( type, name, init_value )   extern type name
# define SET_GLOBAL_NOINIT( type, name )             extern type name
#endif


// include relevant headers
#include "yt_type.h"


// add the prefix "g_" for all global C variables
SET_GLOBAL( yt_param_libyt, g_param_libyt           );   // libyt runtime parameters
                                                         // ==> Do not defined it as a pointer so that it is
                                                         //     initialized during compilation
SET_GLOBAL( yt_param_yt,    g_param_yt              );   // YT parameters

SET_GLOBAL( int,            g_myrank                );   // My current MPI rank

SET_GLOBAL( int,            g_mysize                );   // My current MPI size

// user-defined MPI data type
SET_GLOBAL( MPI_Datatype,   yt_long_mpi_type              );

SET_GLOBAL( MPI_Datatype,   yt_hierarchy_mpi_type         );

SET_GLOBAL( MPI_Datatype,   yt_rma_grid_info_mpi_type     );

SET_GLOBAL( MPI_Datatype,   yt_rma_particle_info_mpi_type );

#ifdef SUPPORT_TIMER
#include "Timer.h"
SET_GLOBAL( Timer,         *g_timer,          NULL  );   // Timer for recording performance.
#endif // #ifdef SUPPORT_TIMER

// add the prefix "g_py_" for all global Python objects
#ifndef NO_PYTHON
SET_GLOBAL( PyObject,      *g_py_grid_data,   NULL  );   // Python dictionary to store grid data
SET_GLOBAL( PyObject,      *g_py_hierarchy,   NULL  );   // Python dictionary to store hierachy information
SET_GLOBAL( PyObject,      *g_py_param_yt,    NULL  );   // Python dictionary to store YT parameters
SET_GLOBAL( PyObject,      *g_py_param_user,  NULL  );   // Python dictionary to store code-specific parameters
#endif


// remove macros used locally
#undef GET_MACRO
#undef SET_GLOBAL
#undef SET_GLOBAL_INIT
#undef SET_GLOBAL_NOINIT



#endif // #ifndef __YT_GLOBAL_H__
