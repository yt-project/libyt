#ifndef LIBYT_PROJECT_INCLUDE_COMM_MPI_RMA_H_
#define LIBYT_PROJECT_INCLUDE_COMM_MPI_RMA_H_

#include <mpi.h>

#include <string>
#include <utility>
#include <vector>

#include "yt_type.h"

struct MPIRmaAddress {
    int mpi_rank;
    MPI_Aint mpi_address;
};

// Probably should define this in data structure header
// TODO: explore if I can std::move
struct AMRDataArray3DInfo {
    long id;
    yt_dtype data_type;
    int data_dim[3];
    bool swap_axes;
};

struct AMRDataArray3D {
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

enum class CommMPIRmaStatus : int { kMPIFailed = 0, kMPISuccess = 1 };

template<typename DataInfoClass, typename DataClass>
class CommMPIRma {
private:
    MPI_Win mpi_window_{};
    std::vector<DataInfoClass> mpi_prepared_data_info_list_;
    DataInfoClass* all_prepared_data_info_list_;
    std::vector<MPIRmaAddress> mpi_prepared_data_address_list_;
    MPIRmaAddress* all_prepared_data_address_list_;

    std::vector<long> search_range_;
    std::vector<DataClass> mpi_fetched_data_;

    std::string data_group_name_;
    std::string data_format_;
    std::string error_str_;

    CommMPIRmaStatus InitializeMPIWindow();
    CommMPIRmaStatus PrepareData(const std::vector<DataClass>& prepared_data_list);
    CommMPIRmaStatus GatherAllPreparedData();
    CommMPIRmaStatus FetchRemoteData();
    CommMPIRmaStatus CleanUp();

public:
    CommMPIRma(const std::string& data_group_name, const std::string& data_format);
    std::pair<CommMPIRmaStatus, const std::vector<DataClass>&> GetRemoteData(
        const std::vector<DataClass>& prepared_data_list, const std::vector<FetchedFromInfo>& fetch_id_list);
    const std::string& GetErrorStr() const { return error_str_; }
};

#endif  // LIBYT_PROJECT_INCLUDE_COMM_MPI_RMA_H_
