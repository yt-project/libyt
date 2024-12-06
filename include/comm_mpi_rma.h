#ifndef LIBYT_PROJECT_INCLUDE_COMM_MPI_RMA_H_
#define LIBYT_PROJECT_INCLUDE_COMM_MPI_RMA_H_
#ifndef SERIAL_MODE

#include <mpi.h>

#include <string>
#include <utility>
#include <vector>

#include "data_structure_amr.h"

struct MpiRmaAddress {
    MPI_Aint mpi_address;
    int mpi_rank;
};

struct CommMpiRmaQueryInfo {
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
    CommMpiRmaStatus FetchRemoteData(const std::vector<CommMpiRmaQueryInfo>& fetch_id_list);
    CommMpiRmaStatus CleanUp(const std::vector<DataClass>& prepared_data_list);

    virtual std::size_t GetDataSize(const DataClass& data) = 0;
    virtual std::size_t GetDataLen(const DataClass& data) = 0;
    // TODO: create virtual function to select the data (ex: id)

public:
    CommMpiRma(const std::string& data_group_name, const std::string& data_format);
    CommMpiRmaReturn<DataClass> GetRemoteData(const std::vector<DataClass>& prepared_data_list,
                                              const std::vector<CommMpiRmaQueryInfo>& fetch_id_list);
    const std::string& GetErrorStr() const { return error_str_; }
};

class CommMpiRmaAmrDataArray3D : public CommMpiRma<AmrDataArray3D> {
private:
    std::size_t GetDataSize(const AmrDataArray3D& data) override;
    std::size_t GetDataLen(const AmrDataArray3D& data) override;

public:
    CommMpiRmaAmrDataArray3D(const std::string& data_group_name, const std::string& data_format)
        : CommMpiRma<AmrDataArray3D>(data_group_name, data_format) {}
};

class CommMpiRmaAmrDataArray1D : public CommMpiRma<AmrDataArray1D> {
private:
    std::size_t GetDataSize(const AmrDataArray1D& data) override;
    std::size_t GetDataLen(const AmrDataArray1D& data) override;

public:
    CommMpiRmaAmrDataArray1D(const std::string& data_group_name, const std::string& data_format)
        : CommMpiRma<AmrDataArray1D>(data_group_name, data_format) {}
};

#endif  // #ifndef SERIAL_MODE
#endif  // LIBYT_PROJECT_INCLUDE_COMM_MPI_RMA_H_
