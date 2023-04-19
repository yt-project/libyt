#include "yt_combo.h"
#include "libyt.h"
#include <typeinfo>


template <typename T>
static int add_nonstring( const char *key, const int n, const T *input );
static int add_string( const char *key, const char *input );


// maximum string width of a key (for outputting debug information only)
static const int MaxParamNameWidth = 15;




//-------------------------------------------------------------------------------------------------------
// Function    :  yt_set_UserParameter*
// Description :  Add code-specific parameters
//
// Note        :  1. All code-specific parameters are stored in "libyt.param_user"
//                2. Overloaded with various data types: float, double, int, long, uint, ulong, char*
//                   ==> But do not use c++ template since I don't know how to instantiating template
//                       without function name mangling ...
//
// Parameter   :  key   : Dictionary key
//                n     : Number of elements in the input array
//                        ==> Currently it must be 1 or 3 (or arbitrary if input is a string)
//                input : Input array containing "n" elements or a single string
//
// Return      :  YT_SUCCESS or YT_FAIL
//-------------------------------------------------------------------------------------------------------

int yt_set_UserParameterInt     ( const char *key, const int n, const int       *input ) { return add_nonstring( key, n, input ); }
int yt_set_UserParameterLong    ( const char *key, const int n, const long      *input ) { return add_nonstring( key, n, input ); }
int yt_set_UserParameterLongLong( const char *key, const int n, const long long *input ) { return add_nonstring( key, n, input ); }
int yt_set_UserParameterUint    ( const char *key, const int n, const uint      *input ) { return add_nonstring( key, n, input ); }
int yt_set_UserParameterUlong   ( const char *key, const int n, const ulong     *input ) { return add_nonstring( key, n, input ); }
int yt_set_UserParameterFloat   ( const char *key, const int n, const float     *input ) { return add_nonstring( key, n, input ); }
int yt_set_UserParameterDouble  ( const char *key, const int n, const double    *input ) { return add_nonstring( key, n, input ); }
int yt_set_UserParameterString  ( const char *key,              const char      *input ) { return add_string   ( key,    input ); }



//***********************************************
// template for various input types except string
//***********************************************
template <typename T>
static int add_nonstring( const char *key, const int n, const T *input )
{

// check if libyt has been initialized
   if ( !g_param_libyt.libyt_initialized )
      YT_ABORT( "Please invoke yt_initialize() before calling %s()!\n", __FUNCTION__ );


// export data to libyt.param_user
   if (  typeid(T) == typeid(float)  ||  typeid(T) == typeid(double)  ||
         typeid(T) == typeid(  int)  ||  typeid(T) == typeid(  long)  ||
         typeid(T) == typeid( uint)  ||  typeid(T) == typeid( ulong)  ||
         typeid(T) == typeid(long long)                                 )
   {
//    scalar and 3-element array
      if      ( n == 1 ) {   if ( add_dict_scalar ( g_py_param_user, key, *input ) == YT_FAIL )   return YT_FAIL;   }
      else if ( n == 3 ) {   if ( add_dict_vector3( g_py_param_user, key,  input ) == YT_FAIL )   return YT_FAIL;   }
      else
         YT_ABORT( "Currently %s() only supports loading a single scalar or a three-element array!\n",
                   __FUNCTION__ );
   }

   else
      YT_ABORT( "Unsupported data type (only support char*, float*, double*, int*, long*, long long*, uint*, ulong*)!\n" );



   log_debug( "Inserting code-specific parameter \"%-*s\" ... done\n", MaxParamNameWidth, key );

   return YT_SUCCESS;

} // FUNCTION : add_nonstring



//***********************************************
// treat string input separately ...
//***********************************************
static int add_string( const char *key, const char *input )
{

// check if libyt has been initialized
   if ( !g_param_libyt.libyt_initialized )
      YT_ABORT( "Please invoke yt_initialize() before calling %s()!\n", __FUNCTION__ );


// export data to libyt.param_user
   if ( add_dict_string( g_py_param_user, key, input ) == YT_FAIL )   return YT_FAIL;


   log_debug( "Inserting code-specific parameter \"%-*s\" ... done\n", MaxParamNameWidth, key );

   return YT_SUCCESS;

} // FUNCTION : add_string

