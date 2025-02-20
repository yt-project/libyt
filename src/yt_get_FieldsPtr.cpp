#include "libyt.h"
#include "libyt_process_control.h"
#include "logging.h"
#include "timer.h"

/**
 * \defgroup api_yt_get_FieldsPtr libyt API: yt_get_FieldsPtr
 * \fn int yt_get_FieldsPtr(yt_field** field_list)
 * \brief Get pointer of field information array
 * \details
 * 1. User should call this function after \ref yt_set_Parameters,
 *    because the AMR structure is initialized there.
 *
 * \todo
 * 1. The setting up of field info is just bad. Should fix it in libyt-v1.0.
 *    Probably should make setting of field info field-by-field
 *
 * @param field_list[out] Pointer to the array of struct yt_field is stored here
 * @return \ref YT_SUCCESS or \ref YT_FAIL
 */
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
    YT_ABORT("num_fields == %d <= 0, you don't need to input field_list, and it is also "
             "not initialized!\n",
             LibytProcessControl::Get().param_yt_.num_fields);
  }

  logging::LogInfo("Getting pointer to field list information ...\n");

  *field_list = LibytProcessControl::Get().data_structure_amr_.GetFieldList();

  LibytProcessControl::Get().get_fields_ptr_ = true;
  logging::LogInfo("Getting pointer to field list information  ... done.\n");

  return YT_SUCCESS;
}
