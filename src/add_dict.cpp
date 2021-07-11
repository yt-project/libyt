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
// Note        :  1. Add a series of key-value pair to libyt.param_yt['field_list'] dictionary.
//                2. Used in yt_commit_grids() on loading field_list structure to python.
//                   This function will only be called when num_fields > 0.
//                3. PyUnicode_FromString is Python-API >= 3.5, and it returns a new reference.
//                4. We assume that we have all the field name unique.
//                5. If field_display_name is NULL, set it to Py_None.
//                6. Dictionary structure loaded in python:
//                   { <field_name>: {"field_define_type" :  <field_define_type>, 
//                                    "field_unit"        :  <field_unit>,
//                                    "field_name_alias"  : [<field_name_alias>, ],
//                                    "field_display_name":  <field_display_name> , 
//                                    "swap_axes"         :  true / false          } }
//
// Parameter   :  None
//
// Return      :  YT_SUCCESS or YT_FAIL
//-------------------------------------------------------------------------------------------------------
int add_dict_field_list(){

   PyObject  *field_list_dict = PyDict_New();
   PyObject  *key, *val;
   PyObject **field_info_dict = new PyObject* [g_param_yt.num_fields];
   PyObject **name_alias_list = new PyObject* [g_param_yt.num_fields];
   for (int i = 0; i < g_param_yt.num_fields; i++) { 
      field_info_dict[i] = PyDict_New( );
      name_alias_list[i] = PyList_New(0);
   }

   for (int i = 0; i < g_param_yt.num_fields; i++){

      // Load "field_define_type" to "field_info_dict".
      val = PyUnicode_FromString((g_param_yt.field_list)[i].field_define_type);
      if ( PyDict_SetItemString(field_info_dict[i], "field_define_type", val) != 0 ){
         YT_ABORT("On setting dictionary [field_list] in libyt, field_name [%s], key-value pair [%s]-[%s] failed!\n", 
                   (g_param_yt.field_list)[i].field_name, "field_define_type", (g_param_yt.field_list)[i].field_define_type);
      }
      Py_DECREF( val );

      // Load "field_unit" to "field_info_dict".
      val = PyUnicode_FromString((g_param_yt.field_list)[i].field_unit);
      if ( PyDict_SetItemString(field_info_dict[i], "field_unit", val) != 0 ){
         YT_ABORT("On setting dictionary [field_list] in libyt, field_name [%s], key-value pair [%s]-[%s] failed!\n", 
                   (g_param_yt.field_list)[i].field_name, "field_unit", (g_param_yt.field_list)[i].field_unit);
      }
      Py_DECREF( val );

      // Load "field_name_alias" to "field_info_dict".
      for (int j = 0; j < (g_param_yt.field_list)[i].num_field_name_alias; j++){
         val = PyUnicode_FromString( (g_param_yt.field_list)[i].field_name_alias[j] );
         if ( PyList_Append(name_alias_list[i], val) != 0 ){
            YT_ABORT("On setting dictionary [field_list] in libyt, field_name [%s], key-value pair [%s]-[ list item %s ] failed!\n",
                      (g_param_yt.field_list)[i].field_name, "field_name_alias", (g_param_yt.field_list)[i].field_name_alias[j]);
         }
         Py_DECREF( val );
      }
      if( PyDict_SetItemString( field_info_dict[i], "field_name_alias", name_alias_list[i]) != 0 ){
         YT_ABORT("On setting dictionary [field_list] in libyt, field_name [%s], key-value pair [%s]-[%s] failed!\n",
                   (g_param_yt.field_list)[i].field_name, "field_name_alias", "list of names");
      }

      // Load "field_display_name" to "field_info_dict".
      // If field_display_name == NULL, load Py_None.
      if ( (g_param_yt.field_list)[i].field_display_name == NULL ){
         if ( PyDict_SetItemString( field_info_dict[i], "field_display_name", Py_None) != 0 ){
            YT_ABORT("On setting dictionary [field_list] in libyt, field_name [%s], key-value pair [%s]-[ None ] failed!\n", 
                      (g_param_yt.field_list)[i].field_name, "field_display_name");
         }
      }
      else {
         val = PyUnicode_FromString( (g_param_yt.field_list)[i].field_display_name );
         if ( PyDict_SetItemString( field_info_dict[i], "field_display_name", val) != 0 ){
            YT_ABORT("On setting dictionary [field_list] in libyt, field_name [%s], key-value pair [%s]-[%s] failed!\n", 
                      (g_param_yt.field_list)[i].field_name, "field_display_name", (g_param_yt.field_list)[i].field_display_name);
         }
         Py_DECREF( val );
      }

      // Load "swap_axes" to "field_info_dict".
      if ( (g_param_yt.field_list)[i].swap_axes == true ){
         if ( PyDict_SetItemString( field_info_dict[i], "swap_axes", Py_True) != 0 ){
            YT_ABORT("On setting dictionary [field_list] in libyt, field_name [%s], key-value pair [%s]-[ true ] failed!\n", 
                      (g_param_yt.field_list)[i].field_name, "swap_axes");
         }
      } 
      else {
         if ( PyDict_SetItemString( field_info_dict[i], "swap_axes", Py_False) != 0 ){
            YT_ABORT("On setting dictionary [field_list] in libyt, field_name [%s], key-value pair [%s]-[ false ] failed!\n", 
                      (g_param_yt.field_list)[i].field_name, "swap_axes");
         }
      }


      // Load "field_info_dict" to "field_list_dict", with key-value pair {field_name : field_info_dict}
      key = PyUnicode_FromString( (g_param_yt.field_list)[i].field_name );
      if ( PyDict_SetItem( field_list_dict, key, field_info_dict[i]) != 0 ){
         YT_ABORT("On setting dictionary [field_list] in libyt, field_name [%s] failed to add dictionary!\n", (g_param_yt.field_list)[i].field_name);
      }
      Py_DECREF( key );

   }

   if ( PyDict_SetItemString( g_py_param_yt, "field_list", field_list_dict ) != 0 ){
      YT_ABORT( "Inserting dictionary [field_list] item to libyt ... failed!\n");
   }

   Py_DECREF( field_list_dict );
   for (int i = 0; i < g_param_yt.num_fields; i++) {
      Py_DECREF( field_info_dict[i] );
      Py_DECREF( name_alias_list[i] );
   }

   return YT_SUCCESS;
}


//-------------------------------------------------------------------------------------------------------
// Function    :  add_dict_particle_list
// Description :  Function for adding a dictionary item to a Python dictionary
//
// Note        :  1. Add a series of key-value pair to libyt.param_yt['particle_list'] dictionary.
//                2. Used in yt_commit_grids() on loading particle_list structure to python. 
//                   This function will only be called when g_param_yt.num_particles > 0.
//                3. PyUnicode_FromString is Python-API >= 3.5, and it returns a new reference.
//                4. We assume that we have all the particle name "species_name" unique. And in each 
//                   species, they have unique "attr_name".
//                5. If attr_display_name is NULL, set it to Py_None.
//                6. Dictionary structure loaded in python:
//           particle_list_dict   species_dict     attr_dict        attr_list  name_alias_list
//                   |                 |               |                |              |
//                   { <species_name>: { "attribute" : { <attr_name1> : [ <attr_unit>, [<attr_name_alias>], <attr_display_name>],
//                                                       <attr_name2> : [ <attr_unit>, [<attr_name_alias>], <attr_display_name>]},
//                                       "particle_coor_label" : [ <coor_x>, <coor_y>, <coor_z>]},
//                   }                                           |
//                                                               |
//                                                            coor_list
// Parameter   :  None
//
// Return      :  YT_SUCCESS or YT_FAIL
//-------------------------------------------------------------------------------------------------------
int add_dict_particle_list(){

   PyObject  *particle_list_dict = PyDict_New();
   PyObject  *key, *val;

   for ( int s = 0; s < g_param_yt.num_species; s++ ){
      
      PyObject  *species_dict = PyDict_New();
      
      // Insert a series of attr_list to attr_dict with key <attr_name>
      PyObject  *attr_dict    = PyDict_New();
      for ( int a = 0; a < g_param_yt.particle_list[s].num_attr; a++ ){

         PyObject    *attr_list = PyList_New(0);
         yt_attribute attr      = g_param_yt.particle_list[s].attr_list[a];

         // Append attr_unit to attr_list
         val = PyUnicode_FromString( attr.attr_unit );
         if ( PyList_Append(attr_list, val) != 0 ){
            YT_ABORT("In species_name == %s, attr_unit == %s, failed to append %s to list.\n",
                      g_param_yt.particle_list[s].species_name, attr.attr_unit, "attr_unit");
         }
         Py_DECREF( val );
         
         // Create name_alias_list and append to attr_list
         PyObject *name_alias_list = PyList_New(0);
         for ( int i = 0; i < attr.num_attr_name_alias; i++ ){
            val = PyUnicode_FromString( attr.attr_name_alias[i] );
            if ( PyList_Append(name_alias_list, val) != 0 ){
               YT_ABORT("In species_name == %s, attr_name == %s, attr_name_alias == %s, failed to append %s to list.\n",
                         g_param_yt.particle_list[s].species_name, attr.attr_name, attr.attr_name_alias[i], "attr_name_alias");
            }
            Py_DECREF( val );
         }
         if ( PyList_Append(attr_list, name_alias_list) != 0 ){
            YT_ABORT("In species_name == %s, attr_name == %s, failed to append %s to list.\n",
                      g_param_yt.particle_list[s].species_name, attr.attr_name, "name_alias_list");
         }
         Py_DECREF( name_alias_list );

         // Append attr_display_name to attr_list if != NULL, otherwise append None.
         if ( attr.attr_display_name == NULL ){
            if ( PyList_Append(attr_list, Py_None) != 0 ){
               YT_ABORT("In species_name == %s, attr_name == %s, attr_display_name == NULL, failed to append %s to list.\n",
                         g_param_yt.particle_list[s].species_name, attr.attr_name, "Py_None");
            }
         }
         else {
            val = PyUnicode_FromString( attr.attr_display_name );
            if ( PyList_Append(attr_list, val) != 0 ){
               YT_ABORT("In species_name == %s, attr_name == %s, attr_display_name == %s, failed to append %s to list.\n",
                         g_param_yt.particle_list[s].species_name, attr.attr_name, attr.attr_display_name, "attr_display_name");
            }
            Py_DECREF( val );
         }

         // Isert attr_list to attr_dict with key = <attr_name>
         key = PyUnicode_FromString( attr.attr_name );
         if ( PyDict_SetItem(attr_dict, key, attr_list) != 0 ){
            YT_ABORT("In species_name == %s, attr_name == %s, failed to append %s to %s.\n",
                      g_param_yt.particle_list[s].species_name, attr.attr_name, "attr_list", "attr_dict");
         }
         Py_DECREF( key );

         Py_DECREF( attr_list );
      }

      // Insert attr_dict to species_dict with key = "attribute"
      if ( PyDict_SetItemString(species_dict, "attribute", attr_dict) != 0 ){
         YT_ABORT("In species_name == %s, failed to insert key-value pair attribute:attr_dict to species_dict.\n", 
                   g_param_yt.particle_list[s].species_name);
      }
      Py_DECREF( attr_dict );


      // Create coor_list and insert it to species_dict with key = "particle_coor_label"
      PyObject *coor_list = PyList_New(0);

      if ( g_param_yt.particle_list[s].coor_x == NULL ){
         if ( PyList_Append(coor_list, Py_None) != 0 ){
            YT_ABORT("In species_name == %s, coor_x == NULL, failed to append %s to coor_list.\n",
                      g_param_yt.particle_list[s].species_name, "Py_None");
         }
      }
      else{
         val = PyUnicode_FromString( g_param_yt.particle_list[s].coor_x );
         if ( PyList_Append(coor_list, val) != 0 ){
            YT_ABORT("In species_name == %s, coor_x == %s, failed to append %s to list.\n",
                      g_param_yt.particle_list[s].species_name, g_param_yt.particle_list[s].coor_x, "coor_x");
         }
         Py_DECREF( val );
      }

      if ( g_param_yt.particle_list[s].coor_y == NULL ){
         if ( PyList_Append(coor_list, Py_None) != 0 ){
            YT_ABORT("In species_name == %s, coor_y == NULL, failed to append %s to coor_list.\n",
                      g_param_yt.particle_list[s].species_name, "Py_None");
         }
      }
      else{
         val = PyUnicode_FromString( g_param_yt.particle_list[s].coor_y );
         if ( PyList_Append(coor_list, val) != 0 ){
            YT_ABORT("In species_name == %s, coor_y == %s, failed to append %s to list.\n",
                      g_param_yt.particle_list[s].species_name, g_param_yt.particle_list[s].coor_y, "coor_y");
         }
         Py_DECREF( val );
      }

      if ( g_param_yt.particle_list[s].coor_z == NULL ){
         if ( PyList_Append(coor_list, Py_None) != 0 ){
            YT_ABORT("In species_name == %s, coor_z == NULL, failed to append %s to coor_list.\n",
                      g_param_yt.particle_list[s].species_name, "Py_None");
         }
      }
      else{
         val = PyUnicode_FromString( g_param_yt.particle_list[s].coor_z );
         if ( PyList_Append(coor_list, val) != 0 ){
            YT_ABORT("In species_name == %s, coor_z == %s, failed to append %s to list.\n",
                      g_param_yt.particle_list[s].species_name, g_param_yt.particle_list[s].coor_z, "coor_z");
         }
         Py_DECREF( val );
      }

      // Insert coor_list to species_dict with key = "particle_coor_label"
      if ( PyDict_SetItemString(species_dict, "particle_coor_label", coor_list) != 0 ){
         YT_ABORT("In species_name == %s, failed to insert key-value pair particle_coor_label:coor_list to species_dict.\n", 
                   g_param_yt.particle_list[s].species_name);
      }
      Py_DECREF( coor_list );


      // Insert species_dict to particle_list_dict with key = <species_name>
      key = PyUnicode_FromString( g_param_yt.particle_list[s].species_name );
      if ( PyDict_SetItem(particle_list_dict, key, species_dict) != 0 ){
         YT_ABORT("In species_name == %s, failed to insert key-value pair %s:species_dict to particle_list_dict.\n", 
                   g_param_yt.particle_list[s].species_name, g_param_yt.particle_list[s].species_name);
      }
      Py_DECREF( key );

      Py_DECREF( species_dict );
   }


   // Insert particle_list_dict to libyt.param_yt["particle_list"]
   if ( PyDict_SetItemString( g_py_param_yt, "particle_list", particle_list_dict ) != 0 ){
      YT_ABORT( "Inserting dictionary [particle_list] item to libyt ... failed!\n");
   }
   Py_DECREF( particle_list_dict );

   return YT_SUCCESS;
}