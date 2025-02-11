if (NOT SERIAL_MODE)
	find_package(MPI REQUIRED)
endif ()

add_executable(example example.cpp)

if (NOT SERIAL_MODE)
    target_include_directories(example PUBLIC
            ${MPI_CXX_INCLUDE_DIRS}
            ${CMAKE_SOURCE_DIR}/include
    )
    target_link_libraries(example PUBLIC
            MPI::MPI_CXX
            yt
    )
else ()
    target_include_directories(example PUBLIC
            ${CMAKE_SOURCE_DIR}/include
    )
    target_link_libraries(example PUBLIC
            yt
    )
endif ()
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inline_script.py ${CMAKE_CURRENT_BINARY_DIR}/inline_script.py COPYONLY)
