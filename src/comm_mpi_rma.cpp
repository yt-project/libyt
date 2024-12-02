#ifndef SERIAL_MODE
#include "comm_mpi_rma.h"

#include <iostream>

#include "timer.h"

template<typename DataInfoClass, typename DataClass>
CommMPIRma<DataInfoClass, DataClass>::CommMPIRma(const std::string& data_group_name, int len_prepare, long len_to_get)
    : data_group_name_(data_group_name) {
    SET_TIMER(__PRETTY_FUNCTION__);

    mpi_prepared_data_.reserve(len_prepare);
    mpi_fetched_data_.reserve(len_to_get);
}

template<typename DataInfoClass, typename DataClass>
std::vector<DataClass>& CommMPIRma<DataInfoClass, DataClass>::GetRemoteData(
    const std::vector<DataClass>& prepared_data, const std::vector<FetchedFromInfo>& fetch_id_list) {
    std::cout << "GetRemoteData" << std::endl;
}

template class CommMPIRma<AMRFieldDataArray3DInfo, AMRFieldDataArray3D>;

#endif