#ifndef LIBYT_PROJECT_INCLUDE_COMM_MPI_RMA_H_
#define LIBYT_PROJECT_INCLUDE_COMM_MPI_RMA_H_
#ifndef SERIAL_MODE

#include <mpi.h>

#include <string>
#include <utility>
#include <vector>

#include "data_hub_amr.h"

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
    CommMpiRmaStatus all_status;
    const std::vector<DataClass>& data_list;
};

template<typename DataClass>
class CommMpiRma {
private:
    static MPI_Datatype mpi_rma_data_type_;
    MPI_Win mpi_window_{};
    DataClass* all_prepared_data_list_;
    std::vector<MpiRmaAddress> mpi_prepared_data_address_list_;
    MpiRmaAddress* all_prepared_data_address_list_;

    std::vector<long> search_range_;
    std::vector<DataClass> mpi_fetched_data_;

    std::string data_group_name_;
    std::string data_format_;
    std::string error_str_;

    // Initializations
    static void InitializeMpiAddressDataType();

    // Rma operations
    CommMpiRmaStatus InitializeMpiWindow();
    CommMpiRmaStatus PrepareData(const std::vector<DataClass>& prepared_data_list);
    CommMpiRmaStatus GatherAllPreparedData(const std::vector<DataClass>& prepared_data_list);
    CommMpiRmaStatus FetchRemoteData(const std::vector<CommMpiRmaQueryInfo>& fetch_id_list);
    CommMpiRmaStatus FreeMpiWindow();
    CommMpiRmaStatus DetachBuffer(const std::vector<DataClass>& prepared_data_list);
    CommMpiRmaStatus CleanUp(const std::vector<DataClass>& prepared_data_list);

    // Custom implementations for derived classes
    virtual long GetDataSize(const DataClass& data) = 0;
    virtual long GetDataLen(const DataClass& data) = 0;
    // TODO: create virtual function to select the data (ex: id)

public:
    CommMpiRma(const std::string& data_group_name, const std::string& data_format);
    CommMpiRmaReturn<DataClass> GetRemoteData(const std::vector<DataClass>& prepared_data_list,
                                              const std::vector<CommMpiRmaQueryInfo>& fetch_id_list);
    const std::string& GetErrorStr() const { return error_str_; }
    MPI_Datatype& GetMpiAddressDataType() { return mpi_rma_data_type_; }

    // Custom implementations for derived classes
    virtual MPI_Datatype& GetMpiDataType() = 0;
};

class CommMpiRmaAmrDataArray3D : public CommMpiRma<AmrDataArray3D> {
private:
    static MPI_Datatype mpi_data_type_;
    long GetDataSize(const AmrDataArray3D& data) override;
    long GetDataLen(const AmrDataArray3D& data) override;
    static void InitializeMpiDataType();

public:
    CommMpiRmaAmrDataArray3D(const std::string& data_group_name, const std::string& data_format);
    MPI_Datatype& GetMpiDataType() override { return CommMpiRmaAmrDataArray3D::mpi_data_type_; }
};

class CommMpiRmaAmrDataArray1D : public CommMpiRma<AmrDataArray1D> {
private:
    static MPI_Datatype mpi_data_type_;
    long GetDataSize(const AmrDataArray1D& data) override;
    long GetDataLen(const AmrDataArray1D& data) override;
    static void InitializeMpiDataType();

public:
    CommMpiRmaAmrDataArray1D(const std::string& data_group_name, const std::string& data_format);
    MPI_Datatype& GetMpiDataType() override { return CommMpiRmaAmrDataArray1D::mpi_data_type_; }
};

#endif  // #ifndef SERIAL_MODE
#endif  // LIBYT_PROJECT_INCLUDE_COMM_MPI_RMA_H_
