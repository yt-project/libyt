#ifndef LIBYT_PROJECT_INCLUDE_COMM_MPI_RMA_H_
#define LIBYT_PROJECT_INCLUDE_COMM_MPI_RMA_H_

#include <mpi.h>

#include <string>
#include <utility>
#include <vector>

#include "yt_type.h"

template<typename T>
struct MPIRmaData {
    long id;
    int label;
    MPI_Aint mpi_address;
    int mpi_rank;
    T data;
};

// Probably should define this in data structure header
// TODO: explore if I can std::move
struct AMRFieldDataArray3DInfo {
    yt_dtype data_type;
    int data_dim[3];
    bool swap_axes_;
};

struct AMRFieldDataArray3D {
    AMRFieldDataArray3DInfo data_info;
    void* data_ptr;
};

struct FetchedFromInfo {
    long gid;
    int mpi_rank;
};

enum class CommMPIRmaStatus : int { kMPIFailed = 0, kMPISuccess = 1 };

template<typename DataInfoClass, typename DataClass>
class CommMPIRma {
private:
    MPI_Win mpi_window_;
    std::vector<MPIRmaData<DataInfoClass>> mpi_prepared_data_;
    std::vector<DataClass> mpi_fetched_data_;
    std::string data_group_name_;
    std::string error_str_;

    CommMPIRmaStatus InitializeMPIWindow();
    CommMPIRmaStatus PrepareData();
    CommMPIRmaStatus GatherAllPreparedData();
    CommMPIRmaStatus FetchRemoteData();
    CommMPIRmaStatus CleanUp();

public:
    CommMPIRma(const std::string& data_group_name);
    std::pair<CommMPIRmaStatus, const std::vector<DataClass>&> GetRemoteData(
        const std::vector<DataClass>& prepared_data, const std::vector<FetchedFromInfo>& fetch_id_list);
    const std::string& GetErrorStr() const { return error_str_; }
};

#endif  // LIBYT_PROJECT_INCLUDE_COMM_MPI_RMA_H_
