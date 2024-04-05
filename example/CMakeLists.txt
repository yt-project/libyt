if (NOT SERIAL_MODE)
	find_package(MPI REQUIRED)
endif ()

add_executable(example example.cpp)

if (NOT SERIAL_MODE)
    target_link_libraries(example PUBLIC MPI::MPI_CXX)
endif ()
target_link_libraries(example PUBLIC yt)
target_include_directories(example PUBLIC "../include")
