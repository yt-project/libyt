#include "libyt.h"
#include "libyt_process_control.h"
#include "yt_combo.h"

//-------------------------------------------------------------------------------------------------------
// Function    :  yt_get_FieldsPtr
// Description :  Get pointer of the array of struct yt_field with length num_fields.
//
// Note        :  1. User should call this function after yt_set_Parameters(), because the Amr structure
//                   is initialized there.
//                   (This also shows a bad design of API, should fix it in libyt-v1.0.)
//                2. Probably should make setting of field info field-by-field, same for particles.
//
// Parameter   :  yt_field **field_list  : Initialize and store the field list array under this pointer
//                                         points to.
//
// Return      :  YT_SUCCESS or YT_FAIL
//-------------------------------------------------------------------------------------------------------
//
int yt_get_FieldsPtr(yt_field** field_list) {
    SET_TIMER(__PRETTY_FUNCTION__);

    // check if libyt has been initialized
    if (!LibytProcessControl::Get().libyt_initialized_) {
        YT_ABORT("Please invoke yt_initialize() before calling %s()!\n", __FUNCTION__);
    }

    // check if yt_set_Parameters() have been called
    if (!LibytProcessControl::Get().param_yt_set_) {
        YT_ABORT("Please invoke yt_set_Parameters() before calling %s()!\n", __FUNCTION__);
    }

    // check if num_fields > 0, if not, field_list won't be initialized
    if (LibytProcessControl::Get().param_yt_.num_fields <= 0) {
        YT_ABORT("num_fields == %d <= 0, you don't need to input field_list, and it is also not initialized!\n",
                 LibytProcessControl::Get().param_yt_.num_fields);
    }

    logging::LogInfo("Getting pointer to field list information ...\n");

    *field_list = LibytProcessControl::Get().data_structure_amr_.GetFieldList();

    LibytProcessControl::Get().get_fields_ptr_ = true;
    logging::LogInfo("Getting pointer to field list information  ... done.\n");

    return YT_SUCCESS;
}
