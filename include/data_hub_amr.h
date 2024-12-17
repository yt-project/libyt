#ifndef LIBYT_PROJECT_INCLUDE_DATA_HUB_AMR_H_
#define LIBYT_PROJECT_INCLUDE_DATA_HUB_AMR_H_

#include <string>
#include <vector>

#include "data_structure_amr.h"
#include "yt_type.h"

struct AmrDataArray3D {
    long id = -1;
    yt_dtype data_dtype = YT_DTYPE_UNKNOWN;
    int data_dim[3]{0, 0, 0};
    void* data_ptr = nullptr;
    bool contiguous_in_x = false;
};

struct AmrDataArray1D {
    long id = -1;
    yt_dtype data_dtype = YT_DTYPE_UNKNOWN;
    void* data_ptr = nullptr;
    long data_len = 0;
};

enum class DataHubStatus : int { kDataHubFailed = 0, kDataHubSuccess = 1 };

template<typename DataClass>
struct DataHubReturn {
    DataHubStatus status;
    const std::vector<DataClass>& data_list;
};

class DataHubAmr {
private:
    std::vector<AmrDataArray3D> amr_data_array_3d_list_;
    std::vector<AmrDataArray1D> amr_data_array_1d_list_;
    std::vector<bool> is_new_allocation_list_;
    std::string error_str_;

    // TODO: need to be able to set this flag based on user input,
    //       since this is only used in RMA currently, will apply it to get_particle/derived_func later.
    bool take_ownership_;

public:
    DataHubAmr() : take_ownership_(false) {}
    DataHubReturn<AmrDataArray3D> GetLocalFieldData(const DataStructureAmr& ds_amr, const std::string& field_name,
                                                    const std::vector<long>& grid_id_list);
    DataHubReturn<AmrDataArray1D> GetLocalParticleData(const DataStructureAmr& ds_amr, const std::string& ptype,
                                                       const std::string& pattr, const std::vector<long>& grid_id_list);
    void ClearCache();
    const std::string& GetErrorStr() const { return error_str_; }
    ~DataHubAmr() { ClearCache(); }
};

#endif  // LIBYT_PROJECT_INCLUDE_DATA_HUB_AMR_H_
