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
   Py_DECREF( py_obj );

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
   Py_DECREF( tuple );
   for (int v=0; v<VecSize; v++)   Py_DECREF( element[v] );

   return YT_SUCCESS;

} // FUNCTION : add_dict_vector3



//-------------------------------------------------------------------------------------------------------
// Function    :  add_dict_string
// Description :  Auxiliary function for adding a string item to a Python dictionary
//
// Note        :  1. There is no function overloading here
//
// Parameter   :  dict   : Target Python dictionary
//                key    : Dictionary key
//                string : String to be inserted
//
// Return      :  YT_SUCCESS or YT_FAIL
//-------------------------------------------------------------------------------------------------------
int add_dict_string( PyObject *dict, const char *key, const char *string )
{

// check if "dict" is indeeed a dict object
   if ( !PyDict_Check(dict) )
      YT_ABORT( "This is not a dict object (key = \"%s\", string = \"%s\")!\n",
                key, string );


// convert "string" to a Python object
   PyObject *py_obj = PyUnicode_FromString( string );


// insert "string" into "dict" with "key"
   if ( PyDict_SetItemString( dict, key, py_obj ) != 0 )
      YT_ABORT( "Inserting a dictionary item with string \"%s\" and key \"%s\" ... failed!\n", string, key );


// decrease the reference count
   Py_DECREF( py_obj );

   return YT_SUCCESS;

} // FUNCTION : add_dict_string



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

//-------------------------------------------------------------------------------------------------------
// Function    :  add_dict_field_list
// Description :  Function for adding a dictionary item to a Python dictionary
//
// Note        :  1. Add a series of key-value pair to libyt.dict, with key and value both as string.
//                2. Used in yt_set_parameter() on setting field_list = { field_name: field_define_type}
//                3. PyUnicode_FromString is Python-API >= 3.5
//
// Parameter   :  dict        : Target Python dictionary
//                field_num   : Number of field
//                field_list  : yt_field array to be added to dict
//
// Return      :  YT_SUCCESS or YT_FAIL
//-------------------------------------------------------------------------------------------------------
int add_dict_field_list(){

   PyObject *dict = PyDict_New();
   PyObject *key, *val;

   for (int i = 0; i < g_param_yt.num_fields; i++){
      key = PyUnicode_FromString((g_param_yt.field_list)[i].field_name);
      val = PyUnicode_FromString((g_param_yt.field_list)[i].field_define_type);
      if ( PyDict_SetItem(dict, key, val) != 0 ){
         YT_ABORT("On setting dictionary field_list in libyt, key-value pair [%s]-[%s] failed!\n", 
                   (g_param_yt.field_list)[i].field_name, (g_param_yt.field_list)[i].field_define_type);
      }
   }

   if ( PyDict_SetItemString( g_py_param_yt, "field_list", dict) != 0 ){
      YT_ABORT( "Inserting a dictionary field_list item ... failed!\n");
   }

   Py_DECREF( key );
   Py_DECREF( val );
   Py_DECREF( dict );

   return YT_SUCCESS;
}