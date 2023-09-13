#include "yt_combo.h"
#include "LibytProcessControl.h"
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

   else if (  typeid(T) == typeid(long long)  )
       py_obj = PyLong_FromLongLong( (long long)value);

   else
      YT_ABORT( "Unsupported data type (only support float, double, int, long, long long, uint, ulong)!\n" );


// insert "value" into "dict" with "key"
   if ( PyDict_SetItemString( dict, key, py_obj ) != 0 )
      YT_ABORT( "Inserting a dictionary item with value \"%.5g\" and key \"%s\" ... failed!\n", (double)value, key );


// decrease the reference count
   Py_DECREF( py_obj );

   return YT_SUCCESS;

} // FUNCTION : add_dict_scalar



//-------------------------------------------------------------------------------------------------------
// Function    :  add_dict_vector_n
// Description :  Auxiliary function for adding an n-element vector item to a Python dictionary
//
// Note        :  1. Overloaded with various data types: float, double, int, long, uint, ulong
//                   ==> (float,double)        are converted to double internally
//                       (int,long,uint,ulong) are converted to long internally
//                       (long long)           are converted to long long internally
//
// Parameter   :  dict   : Target Python dictionary
//                key    : Dictionary key
//                len    : Length of the vector size
//                vector : Vector to be inserted
//
// Return      :  YT_SUCCESS or YT_FAIL
//-------------------------------------------------------------------------------------------------------
template <typename T>
int add_dict_vector_n( PyObject *dict, const char *key, const int len, const T *vector )
{

// check if "dict" is indeeed a dict object
   if ( !PyDict_Check(dict) )
      YT_ABORT( "This is not a dict object (key = \"%s\")!\n", key );


// convert "vector" to a Python object (currently the size of vector is fixed to 3)
   Py_ssize_t VecSize = len;
   PyObject *tuple = PyTuple_New(VecSize);

   if (tuple != NULL)
   {
       if (  typeid(T) == typeid(float)  ||  typeid(T) == typeid(double)  )
       {
           for (Py_ssize_t v=0; v<VecSize; v++) { PyTuple_SET_ITEM(tuple, v, PyFloat_FromDouble((double)vector[v])); }
       }
       else if (  typeid(T) == typeid( int)  ||  typeid(T) == typeid( long)  ||
                  typeid(T) == typeid(uint)  ||  typeid(T) == typeid(ulong)    )
       {
           for (Py_ssize_t v=0; v<VecSize; v++) { PyTuple_SET_ITEM(tuple, v, PyLong_FromLong((long)vector[v])); }
       }
       else if (  typeid(T) == typeid(long long)  )
       {
           for (Py_ssize_t v=0; v<VecSize; v++) { PyTuple_SET_ITEM(tuple, v, PyLong_FromLongLong((long long)vector[v])); }
       }
       else
       {
           YT_ABORT( "Unsupported data type (only support float, double, int, long, long long, uint, ulong)!\n" );
       }
   }
   else
   {
       YT_ABORT( "Creating a tuple object (key = \"%s\") ... failed!\n", key );
   }

// insert "vector" into "dict" with "key"
   if ( PyDict_SetItemString( dict, key, tuple ) != 0 )
      YT_ABORT( "Inserting a dictionary item with the key \"%s\" ... failed!\n", key );


// decrease the reference count
   Py_DECREF( tuple );

   return YT_SUCCESS;

} // FUNCTION : add_dict_vector_n



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
template int add_dict_scalar <float     > ( PyObject *dict, const char *key, const float     value );
template int add_dict_scalar <double    > ( PyObject *dict, const char *key, const double    value );
template int add_dict_scalar <int       > ( PyObject *dict, const char *key, const int       value );
template int add_dict_scalar <long      > ( PyObject *dict, const char *key, const long      value );
template int add_dict_scalar <long long > ( PyObject *dict, const char *key, const long long value );
template int add_dict_scalar <uint      > ( PyObject *dict, const char *key, const uint      value );
template int add_dict_scalar <ulong     > ( PyObject *dict, const char *key, const ulong     value );

template int add_dict_vector_n <float    > ( PyObject *dict, const char *key, const int len, const float     *vector );
template int add_dict_vector_n <double   > ( PyObject *dict, const char *key, const int len, const double    *vector );
template int add_dict_vector_n <int      > ( PyObject *dict, const char *key, const int len, const int       *vector );
template int add_dict_vector_n <long     > ( PyObject *dict, const char *key, const int len, const long      *vector );
template int add_dict_vector_n <long long> ( PyObject *dict, const char *key, const int len, const long long *vector );
template int add_dict_vector_n <uint     > ( PyObject *dict, const char *key, const int len, const uint      *vector );
template int add_dict_vector_n <ulong    > ( PyObject *dict, const char *key, const int len, const ulong     *vector );

//-------------------------------------------------------------------------------------------------------
// Function    :  add_dict_field_list
// Description :  Function for adding a dictionary item to a Python dictionary
//
// Note        :  1. Add a series of key-value pair to libyt.param_yt['field_list'] dictionary.
//                2. Used in yt_commit() on loading field_list structure to python.
//                   This function will only be called when num_fields > 0.
//                3. PyUnicode_FromString is Python-API >= 3.5, and it returns a new reference.
//                4. We assume that we have all the field name unique.
//                5. If field_display_name is NULL, set it to Py_None.
//                6. Dictionary structure loaded in python:
//            field_list_dict    field_info_dict        info_list     name_alias_list
//                   |               |                      |               |
//                   { <field_name>: {"attribute"         : [ <field_unit>, [<field_name_alias>, ], <field_display_name> ]
//                                    "field_type"        :  <field_type>,
//                                    "contiguous_in_x"   :  true / false
//                                    "ghost_cell"        : [ beginning of 0-dim, ending of 0-dim,
//                                                            beginning of 1-dim, ending of 1-dim,
//                                                            beginning of 2-dim, ending of 2-dim  ]                      },
//                   }
//
// Parameter   :  None
//
// Return      :  YT_SUCCESS or YT_FAIL
//-------------------------------------------------------------------------------------------------------
int add_dict_field_list(){
   SET_TIMER(__PRETTY_FUNCTION__);

   PyObject  *field_list_dict = PyDict_New();
   PyObject  *key, *val;

   yt_field *field_list = LibytProcessControl::Get().field_list;

   for (int i = 0; i < g_param_yt.num_fields; i++){

      PyObject *field_info_dict = PyDict_New( );
      PyObject *info_list       = PyList_New(0);

      // Append "field_unit" to "info_list"
      val = PyUnicode_FromString((field_list)[i].field_unit);
      if ( PyList_Append(info_list, val) != 0 ){
         YT_ABORT("In field_name == %s, field_unit == %s, failed to append %s to list!\n", 
                   (field_list)[i].field_name, (field_list)[i].field_unit, "field_unit");
      }
      Py_DECREF( val );

      // Create "name_alias_list" and append to "info_list"
      PyObject *name_alias_list = PyList_New(0);
      for (int j = 0; j < (field_list)[i].num_field_name_alias; j++){
         val = PyUnicode_FromString( (field_list)[i].field_name_alias[j] );
         if ( PyList_Append(name_alias_list, val) != 0 ){
            YT_ABORT("In field_name == %s, field_name_alias == %s, failed to append %s to list!\n",
                      (field_list)[i].field_name, (field_list)[i].field_name_alias[j], "field_name_alias");
         }
         Py_DECREF( val );
      }
      if ( PyList_Append(info_list, name_alias_list) != 0 ){
         YT_ABORT("In field_name == %s, failed to append name_alias_list to list!\n", (field_list)[i].field_name);
      }
      Py_DECREF( name_alias_list );

      // Load "field_display_name" to "info_list"
      // If field_display_name == NULL, load Py_None.
      if ( (field_list)[i].field_display_name == NULL ){
         if ( PyList_Append( info_list, Py_None ) != 0 ){
            YT_ABORT("In field_name == %s, field_display_name == NULL, failed to append Py_None to list!\n", 
                      (field_list)[i].field_name);
         }
      }
      else {
         val = PyUnicode_FromString( (field_list)[i].field_display_name );
         if ( PyList_Append( info_list, val ) != 0 ){
            YT_ABORT("In field_name == %s, field_display_name == %s, failed to append %s to list!\n", 
                      (field_list)[i].field_name, (field_list)[i].field_display_name, "field_display_name");
         }
         Py_DECREF( val );
      }

      // Insert "info_list" to "field_info_dict" with key "attribute"
      if ( PyDict_SetItemString(field_info_dict, "attribute", info_list) != 0 ){
         YT_ABORT("On setting dictionary [field_list] in libyt, field_name [%s], key-value pair ['attribute']-[info_list] failed!\n", 
                   (field_list)[i].field_name);
      }
      Py_DECREF(info_list);

      // Load "field_type" to "field_info_dict".
      val = PyUnicode_FromString((field_list)[i].field_type);
      if ( PyDict_SetItemString(field_info_dict, "field_type", val) != 0 ){
         YT_ABORT("On setting dictionary [field_list] in libyt, field_name [%s], key-value pair [%s]-[%s] failed!\n", 
                   (field_list)[i].field_name, "field_type", (field_list)[i].field_type);
      }
      Py_DECREF( val );

      // Load "contiguous_in_x" to "field_info_dict".
      if ( (field_list)[i].contiguous_in_x == true ){
         if ( PyDict_SetItemString( field_info_dict, "contiguous_in_x", Py_True) != 0 ){
            YT_ABORT("On setting dictionary [field_list] in libyt, field_name [%s], key-value pair [%s]-[ true ] failed!\n", 
                      (field_list)[i].field_name, "contiguous_in_x");
         }
      } 
      else {
         if ( PyDict_SetItemString( field_info_dict, "contiguous_in_x", Py_False) != 0 ){
            YT_ABORT("On setting dictionary [field_list] in libyt, field_name [%s], key-value pair [%s]-[ false ] failed!\n",
                      (field_list)[i].field_name, "contiguous_in_x");
         }
      }

      // Load "ghost_cell" to "field_info_dict"
      PyObject *ghost_cell_list = PyList_New(0);
      for(int d = 0; d < 6; d++){
          val = PyLong_FromLong((long) (field_list)[i].field_ghost_cell[d] );
          if( PyList_Append( ghost_cell_list, val ) != 0 ){
              YT_ABORT("On setting dictionary [field_list] in libyt, field_name [%s], failed to append ghost cell to list!\n",
                       (field_list)[i].field_name);
          }
          Py_DECREF(val);
      }
      if( PyDict_SetItemString( field_info_dict, "ghost_cell", ghost_cell_list ) != 0 ){
          YT_ABORT("On setting dictionary [field_list] in libyt, field_name [%s], key-value pair [%s]-[ list obj ] failed!\n",
                   (field_list)[i].field_name, "ghost_cell");
      }
      Py_DECREF( ghost_cell_list );

      // Load "field_info_dict" to "field_list_dict", with key = field_name
      key = PyUnicode_FromString( (field_list)[i].field_name );
      if ( PyDict_SetItem( field_list_dict, key, field_info_dict) != 0 ){
         YT_ABORT("On setting dictionary [field_list] in libyt, field_name [%s] failed to add dictionary!\n", (field_list)[i].field_name);
      }
      Py_DECREF( key );

      Py_DECREF( field_info_dict );

   }

   if ( PyDict_SetItemString( g_py_param_yt, "field_list", field_list_dict ) != 0 ){
      YT_ABORT( "Inserting dictionary [field_list] to libyt.param_yt ... failed!\n");
   }

   Py_DECREF( field_list_dict );

   return YT_SUCCESS;
}


//-------------------------------------------------------------------------------------------------------
// Function    :  add_dict_particle_list
// Description :  Function for adding a dictionary item to a Python dictionary
//
// Note        :  1. Add a series of key-value pair to libyt.param_yt['particle_list'] dictionary.
//                2. Used in yt_commit() on loading particle_list structure to python.
//                   This function will only be called when g_param_yt.num_particles > 0.
//                3. PyUnicode_FromString is Python-API >= 3.5, and it returns a new reference.
//                4. We assume that we have all the particle name "par_type" unique. And in each
//                   species, they have unique "attr_name".
//                5. If attr_display_name is NULL, set it to Py_None.
//                6. Dictionary structure loaded in python:
//           particle_list_dict   species_dict     attr_dict        attr_list  name_alias_list
//                   |                 |               |                |              |
//                   { <par_type>: { "attribute" : { <attr_name1> : [ <attr_unit>, [<attr_name_alias>], <attr_display_name>],
//                                                   <attr_name2> : [ <attr_unit>, [<attr_name_alias>], <attr_display_name>]},
//                                   "particle_coor_label" : [ <coor_x>, <coor_y>, <coor_z>]},
//                                                           |
//                                                           |
//                                                        coor_list
//                                   "label": <index in particle_list>}
//                    }
// Parameter   :  None
//
// Return      :  YT_SUCCESS or YT_FAIL
//-------------------------------------------------------------------------------------------------------
int add_dict_particle_list(){
   SET_TIMER(__PRETTY_FUNCTION__);

   PyObject  *particle_list_dict = PyDict_New();
   PyObject  *key, *val;

   yt_particle *particle_list = LibytProcessControl::Get().particle_list;

   for ( int s = 0; s < g_param_yt.num_par_types; s++ ){
      
      PyObject  *species_dict = PyDict_New();
      
      // Insert a series of attr_list to attr_dict with key <attr_name>
      PyObject  *attr_dict    = PyDict_New();
      for ( int a = 0; a < particle_list[s].num_attr; a++ ){

         PyObject    *attr_list = PyList_New(0);
         yt_attribute attr      = particle_list[s].attr_list[a];

         // Append attr_unit to attr_list
         val = PyUnicode_FromString( attr.attr_unit );
         if ( PyList_Append(attr_list, val) != 0 ){
            YT_ABORT("In par_type == %s, attr_unit == %s, failed to append %s to list.\n",
                      particle_list[s].par_type, attr.attr_unit, "attr_unit");
         }
         Py_DECREF( val );

         // Create name_alias_list and append to attr_list
         PyObject *name_alias_list = PyList_New(0);
         for ( int i = 0; i < attr.num_attr_name_alias; i++ ){
            val = PyUnicode_FromString( attr.attr_name_alias[i] );
            if ( PyList_Append(name_alias_list, val) != 0 ){
               YT_ABORT("In par_type == %s, attr_name == %s, attr_name_alias == %s, failed to append %s to list.\n",
                         particle_list[s].par_type, attr.attr_name, attr.attr_name_alias[i], "attr_name_alias");
            }
            Py_DECREF( val );
         }
         if ( PyList_Append(attr_list, name_alias_list) != 0 ){
            YT_ABORT("In par_type == %s, attr_name == %s, failed to append %s to list.\n",
                      particle_list[s].par_type, attr.attr_name, "name_alias_list");
         }
         Py_DECREF( name_alias_list );

         // Append attr_display_name to attr_list if != NULL, otherwise append None.
         if ( attr.attr_display_name == NULL ){
            if ( PyList_Append(attr_list, Py_None) != 0 ){
               YT_ABORT("In par_type == %s, attr_name == %s, attr_display_name == NULL, failed to append %s to list.\n",
                         particle_list[s].par_type, attr.attr_name, "Py_None");
            }
         }
         else {
            val = PyUnicode_FromString( attr.attr_display_name );
            if ( PyList_Append(attr_list, val) != 0 ){
               YT_ABORT("In par_type == %s, attr_name == %s, attr_display_name == %s, failed to append %s to list.\n",
                         particle_list[s].par_type, attr.attr_name, attr.attr_display_name, "attr_display_name");
            }
            Py_DECREF( val );
         }

         // Isert attr_list to attr_dict with key = <attr_name>
         key = PyUnicode_FromString( attr.attr_name );
         if ( PyDict_SetItem(attr_dict, key, attr_list) != 0 ){
            YT_ABORT("In par_type == %s, attr_name == %s, failed to append %s to %s.\n",
                      particle_list[s].par_type, attr.attr_name, "attr_list", "attr_dict");
         }
         Py_DECREF( key );

         Py_DECREF( attr_list );
      }

      // Insert attr_dict to species_dict with key = "attribute"
      if ( PyDict_SetItemString(species_dict, "attribute", attr_dict) != 0 ){
         YT_ABORT("In par_type == %s, failed to insert key-value pair attribute:attr_dict to species_dict.\n",
                   particle_list[s].par_type);
      }
      Py_DECREF( attr_dict );


      // Create coor_list and insert it to species_dict with key = "particle_coor_label"
      PyObject *coor_list = PyList_New(0);

      if ( particle_list[s].coor_x == NULL ){
         if ( PyList_Append(coor_list, Py_None) != 0 ){
            YT_ABORT("In par_type == %s, coor_x == NULL, failed to append %s to coor_list.\n",
                      particle_list[s].par_type, "Py_None");
         }
      }
      else{
         val = PyUnicode_FromString( particle_list[s].coor_x );
         if ( PyList_Append(coor_list, val) != 0 ){
            YT_ABORT("In par_type == %s, coor_x == %s, failed to append %s to list.\n",
                      particle_list[s].par_type, particle_list[s].coor_x, "coor_x");
         }
         Py_DECREF( val );
      }

      if ( particle_list[s].coor_y == NULL ){
         if ( PyList_Append(coor_list, Py_None) != 0 ){
            YT_ABORT("In par_type == %s, coor_y == NULL, failed to append %s to coor_list.\n",
                      particle_list[s].par_type, "Py_None");
         }
      }
      else{
         val = PyUnicode_FromString( particle_list[s].coor_y );
         if ( PyList_Append(coor_list, val) != 0 ){
            YT_ABORT("In par_type == %s, coor_y == %s, failed to append %s to list.\n",
                      particle_list[s].par_type, particle_list[s].coor_y, "coor_y");
         }
         Py_DECREF( val );
      }

      if ( particle_list[s].coor_z == NULL ){
         if ( PyList_Append(coor_list, Py_None) != 0 ){
            YT_ABORT("In par_type == %s, coor_z == NULL, failed to append %s to coor_list.\n",
                      particle_list[s].par_type, "Py_None");
         }
      }
      else{
         val = PyUnicode_FromString( particle_list[s].coor_z );
         if ( PyList_Append(coor_list, val) != 0 ){
            YT_ABORT("In par_type == %s, coor_z == %s, failed to append %s to list.\n",
                      particle_list[s].par_type, particle_list[s].coor_z, "coor_z");
         }
         Py_DECREF( val );
      }

      // Insert coor_list to species_dict with key = "particle_coor_label"
      if ( PyDict_SetItemString(species_dict, "particle_coor_label", coor_list) != 0 ){
         YT_ABORT("In par_type == %s, failed to insert key-value pair particle_coor_label:coor_list to species_dict.\n",
                   particle_list[s].par_type);
      }
      Py_DECREF( coor_list );

      // Insert label s to species_dict, with key = "label"
      key = PyLong_FromLong( (long) s );
      if ( PyDict_SetItemString(species_dict, "label", key) != 0 ){
          YT_ABORT("In par_type == %s, failed to insert key-value pair label:%d to species_dict.\n",
                   particle_list[s].par_type, s);
      }
      Py_DECREF( key );


      // Insert species_dict to particle_list_dict with key = <par_type>
      key = PyUnicode_FromString( particle_list[s].par_type );
      if ( PyDict_SetItem(particle_list_dict, key, species_dict) != 0 ){
         YT_ABORT("In par_type == %s, failed to insert key-value pair %s:species_dict to particle_list_dict.\n",
                   particle_list[s].par_type, particle_list[s].par_type);
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