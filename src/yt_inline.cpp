#include "yt_combo.h"
#include <stdarg.h>
#include "libyt.h"


//-------------------------------------------------------------------------------------------------------
// Function    :  yt_inline_argument
// Description :  Execute the YT inline analysis script
//
// Note        :  1. Python script name is stored in "g_param_libyt.script"
//                2. This python script must contain function of <function_name> you called.
//                3. Must give argc (argument count), even if there are no arguments.
//
// Parameter   :  char *function_name : function name in python script 
//                int  argc           : input arguments count
//                ...                 : list of arguments, should be input as (char*)
//
// Return      :  YT_SUCCESS or YT_FAIL
//-------------------------------------------------------------------------------------------------------
int yt_inline_argument( char *function_name, int argc, ... ){

#ifdef SUPPORT_TIMER
    g_timer->record_time(function_name, 0);
#endif

// check if libyt has been initialized
   if ( !g_param_libyt.libyt_initialized ){
      YT_ABORT( "Please invoke yt_init() before calling %s()!\n", __FUNCTION__ );
   }

// Not sure if we need this MPI_Barrier
   MPI_Barrier(MPI_COMM_WORLD);

   log_info( "Performing YT inline analysis ...\n");

   va_list Args, Args_len;
   va_start(Args, argc);
   va_copy(Args_len, Args);

   // Count inline function width = .<function_name>() + '\0'
   int InlineFunctionWidth = strlen(function_name) + 4; 
   for(int i = 0; i < argc; i++){

      if ( i != 0 ) InlineFunctionWidth++; // comma "," in called function
      
      InlineFunctionWidth = InlineFunctionWidth + strlen(va_arg(Args_len, char*));
   }

   // Allocate command, and connect input arguments 
   const int CallYT_CommandWidth = strlen( g_param_libyt.script ) + InlineFunctionWidth;
   char *CallYT = (char*) malloc( CallYT_CommandWidth*sizeof(char) );
   strcpy( CallYT, g_param_libyt.script );
   strcat( CallYT, ".");
   strcat( CallYT, function_name);
   strcat( CallYT, "(");
   for(int i = 0; i < argc; i++){

      if ( i != 0 ) strcat( CallYT, ",");

      strcat( CallYT, va_arg(Args, char*));
   }
   strcat( CallYT, ")");

   va_end(Args_len);
   va_end(Args);

   if ( PyRun_SimpleString( CallYT ) == 0 ){
      log_debug( "Invoking \"%s\" ... done\n", CallYT );
   }
   else{
      YT_ABORT(  "Invoking \"%s\" ... failed\n", CallYT );
   }
   
   log_info( "Performing YT inline analysis <%s> ... done.\n", CallYT);
   free( CallYT );

#ifdef SUPPORT_TIMER
    g_timer->record_time(function_name, 1);
#endif

   return YT_SUCCESS;
}

//-------------------------------------------------------------------------------------------------------
// Function    :  yt_inline
// Description :  Execute the YT inline analysis script
//
// Note        :  1. Python script name is stored in "g_param_libyt.script"
//                2. This python script must contain function of <function_name> you called.
//                3. This python function must not contain input arguments.
//
// Parameter   :  char *function_name : function name in python script
//
// Return      :  YT_SUCCESS or YT_FAIL
//-------------------------------------------------------------------------------------------------------
int yt_inline( char *function_name ){
#ifdef SUPPORT_TIMER
    g_timer->record_time(function_name, 0);
#endif

// check if libyt has been initialized
   if ( !g_param_libyt.libyt_initialized ){
      YT_ABORT( "Please invoke yt_init() before calling %s()!\n", __FUNCTION__ );
   }

// Not sure if we need this MPI_Barrier
   MPI_Barrier(MPI_COMM_WORLD);

   log_info( "Performing YT inline analysis ...\n");

   int InlineFunctionWidth = strlen(function_name) + 4; // width = .<function_name>() + '\0'
   const int CallYT_CommandWidth = strlen( g_param_libyt.script ) + InlineFunctionWidth;
   char *CallYT = (char*) malloc( CallYT_CommandWidth*sizeof(char) );
   sprintf( CallYT, "%s.%s()", g_param_libyt.script, function_name );

   if ( PyRun_SimpleString( CallYT ) == 0 ){
      log_debug( "Invoking \"%s\" ... done\n", CallYT );
   }
   else{
      YT_ABORT(  "Invoking \"%s\" ... failed\n", CallYT );
   }

   log_info( "Performing YT inline analysis <%s> ... done.\n", CallYT);
   free( CallYT );

#ifdef SUPPORT_TIMER
    g_timer->record_time(function_name, 1);
#endif

   return YT_SUCCESS;
}