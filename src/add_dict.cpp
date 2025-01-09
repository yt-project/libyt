#include <typeinfo>

#include "libyt_process_control.h"
#include "yt_combo.h"

#ifdef USE_PYBIND11
#include "pybind11/embed.h"
#endif

#ifndef USE_PYBIND11
//-------------------------------------------------------------------------------------------------------
// Function    :  add_dict_scalar
// Description :  Auxiliary function for adding a scalar item to a Python dictionary
//
// Note        :  1. Overloaded with various data types: float, double, int, long, unsigned int, unsigned long
//                   ==> (float,double)                        are converted to double internally
//                       (int,long,unsigned int,unsigned long) are converted to long internally
//                       (long long)                           are converted to long long internally
//
// Parameter   :  dict  : Target Python dictionary
//                key   : Dictionary key
//                value : Value to be inserted
//
// Return      :  YT_SUCCESS or YT_FAIL
//-------------------------------------------------------------------------------------------------------
template<typename T>
int add_dict_scalar(PyObject* dict, const char* key, const T value) {
    SET_TIMER(__PRETTY_FUNCTION__);

    // check if "dict" is indeeed a dict object
    if (!PyDict_Check(dict))
        YT_ABORT("This is not a dict object (key = \"%s\", value = \"%.5g\")!\n", key, (double)value);

    // convert "value" to a Python object
    PyObject* py_obj;

    if (typeid(T) == typeid(float) || typeid(T) == typeid(double))
        py_obj = PyFloat_FromDouble((double)value);

    else if (typeid(T) == typeid(int) || typeid(T) == typeid(long) || typeid(T) == typeid(unsigned int) ||
             typeid(T) == typeid(unsigned long))
        py_obj = PyLong_FromLong((long)value);

    else if (typeid(T) == typeid(long long))
        py_obj = PyLong_FromLongLong((long long)value);

    else
        YT_ABORT(
            "Unsupported data type (only support float, double, int, long, long long, unsigned int, unsigned long)!\n");

    // insert "value" into "dict" with "key"
    if (PyDict_SetItemString(dict, key, py_obj) != 0)
        YT_ABORT("Inserting a dictionary item with value \"%.5g\" and key \"%s\" ... failed!\n", (double)value, key);

    // decrease the reference count
    Py_DECREF(py_obj);

    return YT_SUCCESS;

}  // FUNCTION : add_dict_scalar

//-------------------------------------------------------------------------------------------------------
// Function    :  add_dict_vector_n
// Description :  Auxiliary function for adding an n-element vector item to a Python dictionary
//
// Note        :  1. Overloaded with various data types: float, double, int, long, unsigned int, unsigned long
//                   ==> (float,double)                        are converted to double internally
//                       (int,long,unsigned int,unsigned long) are converted to long internally
//                       (long long)                           are converted to long long internally
//
// Parameter   :  dict   : Target Python dictionary
//                key    : Dictionary key
//                len    : Length of the vector size
//                vector : Vector to be inserted
//
// Return      :  YT_SUCCESS or YT_FAIL
//-------------------------------------------------------------------------------------------------------
template<typename T>
int add_dict_vector_n(PyObject* dict, const char* key, const int len, const T* vector) {
    SET_TIMER(__PRETTY_FUNCTION__);

    // check if "dict" is indeeed a dict object
    if (!PyDict_Check(dict)) YT_ABORT("This is not a dict object (key = \"%s\")!\n", key);

    // convert "vector" to a Python object (currently the size of vector is fixed to 3)
    Py_ssize_t VecSize = len;
    PyObject* tuple = PyTuple_New(VecSize);

    if (tuple != NULL) {
        if (typeid(T) == typeid(float) || typeid(T) == typeid(double)) {
            for (Py_ssize_t v = 0; v < VecSize; v++) {
                PyTuple_SET_ITEM(tuple, v, PyFloat_FromDouble((double)vector[v]));
            }
        } else if (typeid(T) == typeid(int) || typeid(T) == typeid(long) || typeid(T) == typeid(unsigned int) ||
                   typeid(T) == typeid(unsigned long)) {
            for (Py_ssize_t v = 0; v < VecSize; v++) {
                PyTuple_SET_ITEM(tuple, v, PyLong_FromLong((long)vector[v]));
            }
        } else if (typeid(T) == typeid(long long)) {
            for (Py_ssize_t v = 0; v < VecSize; v++) {
                PyTuple_SET_ITEM(tuple, v, PyLong_FromLongLong((long long)vector[v]));
            }
        } else {
            YT_ABORT("Unsupported data type (only support float, double, int, long, long long, unsigned int, unsigned "
                     "long)!\n");
        }
    } else {
        YT_ABORT("Creating a tuple object (key = \"%s\") ... failed!\n", key);
    }

    // insert "vector" into "dict" with "key"
    if (PyDict_SetItemString(dict, key, tuple) != 0)
        YT_ABORT("Inserting a dictionary item with the key \"%s\" ... failed!\n", key);

    // decrease the reference count
    Py_DECREF(tuple);

    return YT_SUCCESS;

}  // FUNCTION : add_dict_vector_n

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
int add_dict_string(PyObject* dict, const char* key, const char* string) {
    SET_TIMER(__PRETTY_FUNCTION__);
    // check if "dict" is indeeed a dict object
    if (!PyDict_Check(dict)) YT_ABORT("This is not a dict object (key = \"%s\", string = \"%s\")!\n", key, string);

    // convert "string" to a Python object
    PyObject* py_obj = PyUnicode_FromString(string);

    // insert "string" into "dict" with "key"
    if (PyDict_SetItemString(dict, key, py_obj) != 0)
        YT_ABORT("Inserting a dictionary item with string \"%s\" and key \"%s\" ... failed!\n", string, key);

    // decrease the reference count
    Py_DECREF(py_obj);

    return YT_SUCCESS;

}  // FUNCTION : add_dict_string

// explicit template instantiation
template int add_dict_scalar<float>(PyObject* dict, const char* key, const float value);
template int add_dict_scalar<double>(PyObject* dict, const char* key, const double value);
template int add_dict_scalar<int>(PyObject* dict, const char* key, const int value);
template int add_dict_scalar<long>(PyObject* dict, const char* key, const long value);
template int add_dict_scalar<long long>(PyObject* dict, const char* key, const long long value);
template int add_dict_scalar<unsigned int>(PyObject* dict, const char* key, const unsigned int value);
template int add_dict_scalar<unsigned long>(PyObject* dict, const char* key, const unsigned long value);

template int add_dict_vector_n<float>(PyObject* dict, const char* key, const int len, const float* vector);
template int add_dict_vector_n<double>(PyObject* dict, const char* key, const int len, const double* vector);
template int add_dict_vector_n<int>(PyObject* dict, const char* key, const int len, const int* vector);
template int add_dict_vector_n<long>(PyObject* dict, const char* key, const int len, const long* vector);
template int add_dict_vector_n<long long>(PyObject* dict, const char* key, const int len, const long long* vector);
template int add_dict_vector_n<unsigned int>(PyObject* dict, const char* key, const int len,
                                             const unsigned int* vector);
template int add_dict_vector_n<unsigned long>(PyObject* dict, const char* key, const int len,
                                              const unsigned long* vector);
#endif  // #ifndef USE_PYBIND11

//-------------------------------------------------------------------------------------------------------
// Function    :  add_dict_particle_list
// Description :  Function for adding a dictionary item to a Python dictionary
//
// Note        :  1. Add a series of key-value pair to libyt.param_yt['particle_list'] dictionary.
//                2. Used in yt_commit() on loading particle_list structure to python.
//                   This function will only be called when LibytProcessControl::Get().param_yt_.num_particles > 0.
//                3. PyUnicode_FromString is Python-API >= 3.5, and it returns a new reference.
//                4. We assume that we have all the particle name "par_type" unique. And in each
//                   species, they have unique "attr_name".
//                5. If attr_display_name is NULL, set it to Py_None.
//                6. Dictionary structure loaded in python:
//      particle_list_dict   species_dict     attr_dict        attr_list  name_alias_list
//              |                 |               |                |              |
//              { <par_type>: { "attribute" : { <attr_name1> : ( <attr_unit>, (<attr_name_alias>), <attr_display_name>),
//                                              <attr_name2> : ( <attr_unit>, (<attr_name_alias>),
//                                              <attr_display_name>)},
//                              "particle_coor_label" : (<coor_x>, <coor_y>, <coor_z>),
//                                                      |
//                                                      |
//                                                   coor_list
//                              "label": <index in particle_list>}
//               }
// Parameter   :  None
//
// Return      :  YT_SUCCESS or YT_FAIL
//-------------------------------------------------------------------------------------------------------
int add_dict_particle_list() {
    SET_TIMER(__PRETTY_FUNCTION__);

#ifndef USE_PYBIND11
    PyObject* particle_list_dict = PyDict_New();
    PyObject *key, *val;

    yt_particle* particle_list = LibytProcessControl::Get().data_structure_amr_.particle_list_;

    for (int s = 0; s < LibytProcessControl::Get().param_yt_.num_par_types; s++) {
        PyObject* species_dict = PyDict_New();

        // Insert a series of attr_list to attr_dict with key <attr_name>
        PyObject* attr_dict = PyDict_New();
        for (int a = 0; a < particle_list[s].num_attr; a++) {
            PyObject* attr_list = PyList_New(0);
            yt_attribute attr = particle_list[s].attr_list[a];

            // Append attr_unit to attr_list
            val = PyUnicode_FromString(attr.attr_unit);
            if (PyList_Append(attr_list, val) != 0) {
                Py_DECREF(particle_list_dict);
                Py_DECREF(species_dict);
                Py_DECREF(attr_dict);
                Py_DECREF(attr_list);
                Py_XDECREF(val);
                YT_ABORT("In par_type == %s, attr_unit == %s, failed to append %s to list.\n",
                         particle_list[s].par_type, attr.attr_unit, "attr_unit");
            }
            Py_DECREF(val);

            // Create name_alias_list and append to attr_list
            PyObject* name_alias_list = PyList_New(0);
            for (int i = 0; i < attr.num_attr_name_alias; i++) {
                val = PyUnicode_FromString(attr.attr_name_alias[i]);
                if (PyList_Append(name_alias_list, val) != 0) {
                    Py_DECREF(particle_list_dict);
                    Py_DECREF(species_dict);
                    Py_DECREF(attr_dict);
                    Py_DECREF(attr_list);
                    Py_DECREF(name_alias_list);
                    Py_XDECREF(val);
                    YT_ABORT(
                        "In par_type == %s, attr_name == %s, attr_name_alias == %s, failed to append %s to list.\n",
                        particle_list[s].par_type, attr.attr_name, attr.attr_name_alias[i], "attr_name_alias");
                }
                Py_DECREF(val);
            }
            if (PyList_Append(attr_list, name_alias_list) != 0) {
                Py_DECREF(particle_list_dict);
                Py_DECREF(species_dict);
                Py_DECREF(attr_dict);
                Py_DECREF(attr_list);
                Py_DECREF(name_alias_list);
                YT_ABORT("In par_type == %s, attr_name == %s, failed to append %s to list.\n",
                         particle_list[s].par_type, attr.attr_name, "name_alias_list");
            }
            Py_DECREF(name_alias_list);

            // Append attr_display_name to attr_list if != NULL, otherwise append None.
            if (attr.attr_display_name == nullptr) {
                if (PyList_Append(attr_list, Py_None) != 0) {
                    Py_DECREF(particle_list_dict);
                    Py_DECREF(species_dict);
                    Py_DECREF(attr_dict);
                    Py_DECREF(attr_list);
                    YT_ABORT(
                        "In par_type == %s, attr_name == %s, attr_display_name == NULL, failed to append %s to list.\n",
                        particle_list[s].par_type, attr.attr_name, "Py_None");
                }
            } else {
                val = PyUnicode_FromString(attr.attr_display_name);
                if (PyList_Append(attr_list, val) != 0) {
                    Py_DECREF(particle_list_dict);
                    Py_DECREF(species_dict);
                    Py_DECREF(attr_dict);
                    Py_DECREF(attr_list);
                    Py_XDECREF(val);
                    YT_ABORT(
                        "In par_type == %s, attr_name == %s, attr_display_name == %s, failed to append %s to list.\n",
                        particle_list[s].par_type, attr.attr_name, attr.attr_display_name, "attr_display_name");
                }
                Py_DECREF(val);
            }

            // Isert attr_list to attr_dict with key = <attr_name>
            key = PyUnicode_FromString(attr.attr_name);
            if (PyDict_SetItem(attr_dict, key, attr_list) != 0) {
                Py_DECREF(particle_list_dict);
                Py_DECREF(species_dict);
                Py_DECREF(attr_dict);
                Py_DECREF(attr_list);
                Py_XDECREF(key);
                YT_ABORT("In par_type == %s, attr_name == %s, failed to append %s to %s.\n", particle_list[s].par_type,
                         attr.attr_name, "attr_list", "attr_dict");
            }
            Py_DECREF(key);

            Py_DECREF(attr_list);
        }

        // Insert attr_dict to species_dict with key = "attribute"
        if (PyDict_SetItemString(species_dict, "attribute", attr_dict) != 0) {
            Py_DECREF(particle_list_dict);
            Py_DECREF(species_dict);
            Py_DECREF(attr_dict);
            YT_ABORT("In par_type == %s, failed to insert key-value pair attribute:attr_dict to species_dict.\n",
                     particle_list[s].par_type);
        }
        Py_DECREF(attr_dict);

        // Create coor_list and insert it to species_dict with key = "particle_coor_label"
        PyObject* coor_list = PyList_New(0);

        if (particle_list[s].coor_x == nullptr) {
            if (PyList_Append(coor_list, Py_None) != 0) {
                Py_DECREF(particle_list_dict);
                Py_DECREF(species_dict);
                Py_DECREF(coor_list);
                YT_ABORT("In par_type == %s, coor_x == NULL, failed to append %s to coor_list.\n",
                         particle_list[s].par_type, "Py_None");
            }
        } else {
            val = PyUnicode_FromString(particle_list[s].coor_x);
            if (PyList_Append(coor_list, val) != 0) {
                Py_DECREF(particle_list_dict);
                Py_DECREF(species_dict);
                Py_DECREF(coor_list);
                Py_XDECREF(val);
                YT_ABORT("In par_type == %s, coor_x == %s, failed to append %s to list.\n", particle_list[s].par_type,
                         particle_list[s].coor_x, "coor_x");
            }
            Py_DECREF(val);
        }

        if (particle_list[s].coor_y == nullptr) {
            if (PyList_Append(coor_list, Py_None) != 0) {
                Py_DECREF(particle_list_dict);
                Py_DECREF(species_dict);
                Py_DECREF(coor_list);
                YT_ABORT("In par_type == %s, coor_y == NULL, failed to append %s to coor_list.\n",
                         particle_list[s].par_type, "Py_None");
            }
        } else {
            val = PyUnicode_FromString(particle_list[s].coor_y);
            if (PyList_Append(coor_list, val) != 0) {
                Py_DECREF(particle_list_dict);
                Py_DECREF(species_dict);
                Py_DECREF(coor_list);
                Py_XDECREF(val);
                YT_ABORT("In par_type == %s, coor_y == %s, failed to append %s to list.\n", particle_list[s].par_type,
                         particle_list[s].coor_y, "coor_y");
            }
            Py_DECREF(val);
        }

        if (particle_list[s].coor_z == nullptr) {
            if (PyList_Append(coor_list, Py_None) != 0) {
                Py_DECREF(particle_list_dict);
                Py_DECREF(species_dict);
                Py_DECREF(coor_list);
                YT_ABORT("In par_type == %s, coor_z == NULL, failed to append %s to coor_list.\n",
                         particle_list[s].par_type, "Py_None");
            }
        } else {
            val = PyUnicode_FromString(particle_list[s].coor_z);
            if (PyList_Append(coor_list, val) != 0) {
                Py_DECREF(particle_list_dict);
                Py_DECREF(species_dict);
                Py_DECREF(coor_list);
                Py_XDECREF(val);
                YT_ABORT("In par_type == %s, coor_z == %s, failed to append %s to list.\n", particle_list[s].par_type,
                         particle_list[s].coor_z, "coor_z");
            }
            Py_DECREF(val);
        }

        // Insert coor_list to species_dict with key = "particle_coor_label"
        if (PyDict_SetItemString(species_dict, "particle_coor_label", coor_list) != 0) {
            Py_DECREF(particle_list_dict);
            Py_DECREF(species_dict);
            Py_DECREF(coor_list);
            YT_ABORT(
                "In par_type == %s, failed to insert key-value pair particle_coor_label:coor_list to species_dict.\n",
                particle_list[s].par_type);
        }
        Py_DECREF(coor_list);

        // Insert label s to species_dict, with key = "label"
        key = PyLong_FromLong((long)s);
        if (PyDict_SetItemString(species_dict, "label", key) != 0) {
            Py_DECREF(particle_list_dict);
            Py_DECREF(species_dict);
            Py_XDECREF(key);
            YT_ABORT("In par_type == %s, failed to insert key-value pair label:%d to species_dict.\n",
                     particle_list[s].par_type, s);
        }
        Py_DECREF(key);

        // Insert species_dict to particle_list_dict with key = <par_type>
        key = PyUnicode_FromString(particle_list[s].par_type);
        if (PyDict_SetItem(particle_list_dict, key, species_dict) != 0) {
            Py_DECREF(particle_list_dict);
            Py_DECREF(species_dict);
            Py_XDECREF(key);
            YT_ABORT("In par_type == %s, failed to insert key-value pair %s:species_dict to particle_list_dict.\n",
                     particle_list[s].par_type, particle_list[s].par_type);
        }
        Py_DECREF(key);

        Py_DECREF(species_dict);
    }

    // Insert particle_list_dict to libyt.param_yt["particle_list"]
    if (PyDict_SetItemString(LibytProcessControl::Get().py_param_yt_, "particle_list", particle_list_dict) != 0) {
        Py_DECREF(particle_list_dict);
        YT_ABORT("Inserting dictionary [particle_list] item to libyt ... failed!\n");
    }
    Py_DECREF(particle_list_dict);
#else
    pybind11::module_ libyt = pybind11::module_::import("libyt");
    pybind11::dict py_param_yt = libyt.attr("param_yt");
    pybind11::dict py_particle_list = pybind11::dict();
    py_param_yt["particle_list"] = py_particle_list;

    yt_particle* particle_list = LibytProcessControl::Get().data_structure_amr_.particle_list_;

    for (int i = 0; i < LibytProcessControl::Get().param_yt_.num_par_types; i++) {
        py_particle_list[particle_list[i].par_type] = pybind11::dict();

        pybind11::dict py_attr_dict = pybind11::dict();
        py_particle_list[particle_list[i].par_type]["attribute"] = py_attr_dict;
        for (int v = 0; v < particle_list[i].num_attr; v++) {
            pybind11::tuple py_name_alias = pybind11::tuple(particle_list[i].attr_list[v].num_attr_name_alias);
            for (int a = 0; a < particle_list[i].attr_list[v].num_attr_name_alias; a++) {
                py_name_alias[a] = particle_list[i].attr_list[v].attr_name_alias[a];
            }

            py_attr_dict[particle_list[i].attr_list[v].attr_name] =
                pybind11::make_tuple(particle_list[i].attr_list[v].attr_unit, py_name_alias,
                                     particle_list[i].attr_list[v].attr_display_name);
        }

        py_particle_list[particle_list[i].par_type]["particle_coor_label"] =
            pybind11::make_tuple(particle_list[i].coor_x, particle_list[i].coor_y, particle_list[i].coor_z);
        py_particle_list[particle_list[i].par_type]["label"] = i;
    }

#endif

    return YT_SUCCESS;
}