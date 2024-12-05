#ifndef LIBYT_PROJECT_INCLUDE_COMM_MPI_RMA_H_
#define LIBYT_PROJECT_INCLUDE_COMM_MPI_RMA_H_
#ifndef SERIAL_MODE

#include <mpi.h>

#include <string>
#include <utility>
#include <vector>

#include "yt_type.h"

struct MpiRmaAddress {
    MPI_Aint mpi_address;
    int mpi_rank;
};

struct FetchedFromInfo {
    int mpi_rank;
    long id;
};

enum class CommMpiRmaStatus : int { kMpiFailed = 0, kMpiSuccess = 1 };

template<typename DataClass>
struct CommMpiRmaReturn {
    CommMpiRmaStatus status;
    const std::vector<DataClass>& data_list;
};

template<typename DataClass>
class CommMpiRma {
private:
    MPI_Win mpi_window_{};
    DataClass* all_prepared_data_list_;
    std::vector<MpiRmaAddress> mpi_prepared_data_address_list_;
    MpiRmaAddress* all_prepared_data_address_list_;

    std::vector<long> search_range_;
    std::vector<DataClass> mpi_fetched_data_;

    std::string data_group_name_;
    std::string data_format_;
    std::string error_str_;

    CommMpiRmaStatus InitializeMpiWindow();
    CommMpiRmaStatus PrepareData(const std::vector<DataClass>& prepared_data_list);
    CommMpiRmaStatus GatherAllPreparedData(const std::vector<DataClass>& prepared_data_list);
    CommMpiRmaStatus FetchRemoteData(const std::vector<FetchedFromInfo>& fetch_id_list);
    CommMpiRmaStatus CleanUp(const std::vector<DataClass>& prepared_data_list);

public:
    CommMpiRma(const std::string& data_group_name, const std::string& data_format);
    CommMpiRmaReturn<DataClass> GetRemoteData(const std::vector<DataClass>& prepared_data_list,
                                              const std::vector<FetchedFromInfo>& fetch_id_list);
    const std::string& GetErrorStr() const { return error_str_; }
};

#endif  // #ifndef SERIAL_MODE
#endif  // LIBYT_PROJECT_INCLUDE_COMM_MPI_RMA_H_
