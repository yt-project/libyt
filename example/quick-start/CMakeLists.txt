if (NOT SERIAL_MODE)
    find_package(MPI REQUIRED)
endif ()

add_executable(quick-start quick-start.cpp)

if (NOT SERIAL_MODE)
    target_link_libraries(quick-start PUBLIC MPI::MPI_CXX)
endif ()
target_link_libraries(quick-start PUBLIC yt)
target_include_directories(quick-start PUBLIC "../../include")
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inline_script.py ${CMAKE_CURRENT_BINARY_DIR}/inline_script.py COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/LIBYT_STOP ${CMAKE_CURRENT_BINARY_DIR}/LIBYT_STOP COPYONLY)
