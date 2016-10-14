#ifndef __YT_MACRO_H__
#define __YT_MACRO_H__



#ifndef NULL
#define NULL               0
#endif

#define YT_SUCCESS         1
#define YT_FAIL            0

#define FLOAT_UNDEFINED    3.40282347e+38F
#define INT_UNDEFINED      2147483647


// convenient macro for outputting error messages and returning YT_FAIL
#define YT_ABORT( ... )                                             \
{                                                                   \
   log_error( __VA_ARGS__ );                                        \
   fprintf( stderr, "       file <%s>, line <%d>, function <%s>\n", \
            __FILE__, __LINE__, __FUNCTION__ );                     \
   return YT_FAIL;                                                  \
}



#endif // #ifndef __YT_MACRO_H__
