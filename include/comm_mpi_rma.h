#ifndef LIBYT_PROJECT_INCLUDE_COMM_MPI_RMA_H_
#define LIBYT_PROJECT_INCLUDE_COMM_MPI_RMA_H_

#include <mpi.h>

#include <string>
#include <utility>
#include <vector>

#include "yt_type.h"

struct MpiRmaAddress {
    int mpi_rank;
    MPI_Aint mpi_address;
};

// Probably should define this in data structure header
// TODO: explore if I can std::move
struct AmrDataArray3DInfo {
    long id;
    yt_dtype data_type;
    int data_dim[3];
    bool swap_axes;
};

struct AmrDataArray3D {
    long id;
    yt_dtype data_type;
    int data_dim[3];
    bool swap_axes;
    void* data_ptr;
};

struct FetchedFromInfo {
    int mpi_rank;
    long id;
};

enum class CommMpiRmaStatus : int { kMpiFailed = 0, kMpiSuccess = 1 };

template<typename DataInfoClass, typename DataClass>
class CommMpiRma {
private:
    MPI_Win mpi_window_{};
    std::vector<DataInfoClass> mpi_prepared_data_info_list_;
    DataInfoClass* all_prepared_data_info_list_;
    std::vector<MpiRmaAddress> mpi_prepared_data_address_list_;
    MpiRmaAddress* all_prepared_data_address_list_;

    std::vector<long> search_range_;
    std::vector<DataClass> mpi_fetched_data_;

    std::string data_group_name_;
    std::string data_format_;
    std::string error_str_;

    CommMpiRmaStatus InitializeMpiWindow();
    CommMpiRmaStatus PrepareData(const std::vector<DataClass>& prepared_data_list);
    CommMpiRmaStatus GatherAllPreparedData();
    CommMpiRmaStatus FetchRemoteData(const std::vector<FetchedFromInfo>& fetch_id_list);
    CommMpiRmaStatus CleanUp();

public:
    CommMpiRma(const std::string& data_group_name, const std::string& data_format);
    std::pair<CommMpiRmaStatus, const std::vector<DataClass>&> GetRemoteData(
        const std::vector<DataClass>& prepared_data_list, const std::vector<FetchedFromInfo>& fetch_id_list);
    const std::string& GetErrorStr() const { return error_str_; }
};

#endif  // LIBYT_PROJECT_INCLUDE_COMM_MPI_RMA_H_
