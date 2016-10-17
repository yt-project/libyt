#include "yt_combo.h"
#include <typeinfo>




//-------------------------------------------------------------------------------------------------------
// Function    :  add_dict_scalar
// Description :  Auxiliary function for adding a scalar item to a Python dictionary
//
// Note        :  1. Overloaded with various data types: float, double, int, long, uint, ulong
//                   ==> (float,double)        are converted to double internally
//                       (int,long,uint,ulong) are converted to long internally
//
// Parameter   :  dict  : Target Python dictionary
//                key   : Dictionary key
//                value : Value to be inserted
//
// Return      :  YT_SUCCESS or YT_FAIL
//-------------------------------------------------------------------------------------------------------
template <typename T>
int add_dict_scalar( PyObject *dict, const char *key, const T value )
{

// check if "dict" is indeeed a dict object
   if ( !PyDict_Check(dict) )
      YT_ABORT( "This is not a dict object (key = \"%s\", value = \"%.5g\")!\n",
                key, (double)value );


// convert "value" to a Python object
   PyObject *py_obj;

   if (  typeid(T) == typeid(float)  ||  typeid(T) == typeid(double)  )
      py_obj = PyFloat_FromDouble( (double)value );

   else if (  typeid(T) == typeid( int)  ||  typeid(T) == typeid( long)  ||
              typeid(T) == typeid(uint)  ||  typeid(T) == typeid(ulong)    )
      py_obj = PyLong_FromLong( (long)value );

   else
      YT_ABORT( "Unsupported data type (only support float, double, int, long, unit, ulong)!\n" );


// insert "value" into "dict" with "key"
   if ( PyDict_SetItemString( dict, key, py_obj ) != 0 )
      YT_ABORT( "Inserting a dictionary item with value \"%.5g\" and key \"%s\" ... failed!\n", (double)value, key );


// decrease the reference count
   Py_XDECREF( py_obj );

   return YT_SUCCESS;

} // FUNCTION : add_dict_scalar



//-------------------------------------------------------------------------------------------------------
// Function    :  add_dict_vector3
// Description :  Auxiliary function for adding a 3-element vector item to a Python dictionary
//
// Note        :  1. Overloaded with various data types: float, double, int, long, uint, ulong
//                   ==> (float,double)        are converted to double internally
//                       (int,long,uint,ulong) are converted to long internally
//                2. Currently the size of vector must be 3
//
// Parameter   :  dict   : Target Python dictionary
//                key    : Dictionary key
//                vector : Vector to be inserted
//
// Return      :  YT_SUCCESS or YT_FAIL
//-------------------------------------------------------------------------------------------------------
template <typename T>
int add_dict_vector3( PyObject *dict, const char *key, const T *vector )
{

// check if "dict" is indeeed a dict object
   if ( !PyDict_Check(dict) )
      YT_ABORT( "This is not a dict object (key = \"%s\")!\n", key );


// convert "vector" to a Python object (currently the size of vector is fixed to 3)
   const int VecSize = 3;
   PyObject *tuple, *element[VecSize];

   if (  typeid(T) == typeid(float)  ||  typeid(T) == typeid(double)  )
   {
      for (int v=0; v<VecSize; v++)
         element[v] = PyFloat_FromDouble( (double)vector[v] );
   }

   else if (  typeid(T) == typeid( int)  ||  typeid(T) == typeid( long)  ||
              typeid(T) == typeid(uint)  ||  typeid(T) == typeid(ulong)    )
   {
      for (int v=0; v<VecSize; v++)
         element[v] = PyLong_FromLong( (long)vector[v] );
   }

   else
      YT_ABORT( "Unsupported data type (only support float, double, int, long, unit, ulong)!\n" );


// create a tuple object
   if (  ( tuple = PyTuple_Pack( VecSize, element[0], element[1], element[2] ) ) == NULL  )
      YT_ABORT( "Creating a tuple object (key = \"%s\") ... failed!\n", key );


// insert "vector" into "dict" with "key"
   if ( PyDict_SetItemString( dict, key, tuple ) != 0 )
      YT_ABORT( "Inserting a dictionary item with the key \"%s\" ... failed!\n", key );


// decrease the reference count
   Py_XDECREF( tuple );
   for (int v=0; v<VecSize; v++)   Py_XDECREF( element[v] );

   return YT_SUCCESS;

} // FUNCTION : add_dict_vector3



// explicit template instantiation
template int add_dict_scalar <float > ( PyObject *dict, const char *key, const float  value );
template int add_dict_scalar <double> ( PyObject *dict, const char *key, const double value );
template int add_dict_scalar <int   > ( PyObject *dict, const char *key, const int    value );
template int add_dict_scalar <long  > ( PyObject *dict, const char *key, const long   value );
template int add_dict_scalar <uint  > ( PyObject *dict, const char *key, const uint   value );
template int add_dict_scalar <ulong > ( PyObject *dict, const char *key, const ulong  value );

template int add_dict_vector3 <float > ( PyObject *dict, const char *key, const float  *vector );
template int add_dict_vector3 <double> ( PyObject *dict, const char *key, const double *vector );
template int add_dict_vector3 <int   > ( PyObject *dict, const char *key, const int    *vector );
template int add_dict_vector3 <long  > ( PyObject *dict, const char *key, const long   *vector );
template int add_dict_vector3 <uint  > ( PyObject *dict, const char *key, const uint   *vector );
template int add_dict_vector3 <ulong > ( PyObject *dict, const char *key, const ulong  *vector );

