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

template<typename DataClass>
class DataHub {
private:
    bool take_ownership_;  // if true, caller takes the ownership of the data.

protected:
    std::vector<DataClass> data_array_list_;
    std::vector<bool> is_new_allocation_list_;
    std::string error_str_;

public:
    explicit DataHub(bool take_ownership) : take_ownership_(take_ownership) {}
    void ClearCache();
    const std::string& GetErrorStr() const { return error_str_; }
    ~DataHub() { ClearCache(); }
};

class DataHubAmrDataArray3D : public DataHub<AmrDataArray3D> {
public:
    explicit DataHubAmrDataArray3D(bool take_ownership) : DataHub<AmrDataArray3D>(take_ownership) {}
    DataHubReturn<AmrDataArray3D> GetLocalFieldData(const DataStructureAmr& ds_amr, const std::string& field_name,
                                                    const std::vector<long>& grid_id_list);
};

class DataHubAmrDataArray1D : public DataHub<AmrDataArray1D> {
public:
    explicit DataHubAmrDataArray1D(bool take_ownership) : DataHub<AmrDataArray1D>(take_ownership) {}
    DataHubReturn<AmrDataArray1D> GetLocalParticleData(const DataStructureAmr& ds_amr, const std::string& ptype,
                                                       const std::string& pattr, const std::vector<long>& grid_id_list);
};

#endif  // LIBYT_PROJECT_INCLUDE_DATA_HUB_AMR_H_
