if (LIBYT_RUN_TEST)
include(FetchContent)

# Getting dependencies
## GoogleTest
set(LIBYT_GOOGLETEST_FETCH_VERSION       1.15.2 )

FetchContent_Declare(
        googletest
        GIT_REPOSITORY "https://github.com/google/googletest.git"
        GIT_TAG        "v${LIBYT_GOOGLETEST_FETCH_VERSION}"
        GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(googletest)

## Libyt dependencies
find_package(Python 3.7 COMPONENTS Development NumPy REQUIRED)
if (NOT SERIAL_MODE)
    find_package(MPI REQUIRED)
endif ()

# Test scripts and discover tests
include(GoogleTest)

## Test Comm MPI
if (NOT SERIAL_MODE)
add_executable(TestCommMPI test_comm_mpi.cpp)
target_include_directories(TestCommMPI PUBLIC
    ${MPI_CXX_INCLUDE_DIRS}
    ${Python_INCLUDE_DIRS}
    ${CMAKE_SOURCE_DIR}/include
)
target_link_libraries(TestCommMPI PUBLIC
    gtest_main
    MPI::MPI_CXX
    ${Python_LIBRARIES}
    yt
)
endif ()

## Test NumPy
add_executable(TestNumPyController test_numpy_controller.cpp)
target_include_directories(TestNumPyController PUBLIC
    ${Python_INCLUDE_DIRS}
    ${CMAKE_SOURCE_DIR}/include
)
target_link_libraries(TestNumPyController PUBLIC
    gtest_main
    ${Python_LIBRARIES} Python::NumPy
    yt
)

## Test Python Shell
if (INTERACTIVE_MODE OR JUPYTER_KERNEL)
if (NOT SERIAL_MODE)
add_executable(TestPythonShellWithCommMPI test_python_shell.cpp)
target_include_directories(TestPythonShellWithCommMPI PUBLIC
    ${MPI_CXX_INCLUDE_DIRS}
    ${Python_INCLUDE_DIRS}
    ${CMAKE_SOURCE_DIR}/include
)
target_link_libraries(TestPythonShellWithCommMPI PUBLIC
    gtest_main
    MPI::MPI_CXX
    ${Python_LIBRARIES}
    yt
)
else ()
add_executable(TestPythonShell test_python_shell.cpp)
target_include_directories(TestPythonShell PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${Python_INCLUDE_DIRS}
)
target_link_libraries(TestPythonShell PUBLIC
    gtest_main
    ${Python_LIBRARIES}
    yt
)
endif ()
endif ()

## Test Data Structure Amr
if (NOT SERIAL_MODE)
add_executable(TestDataStructureAmrWithCommMpi test_data_structure_amr.cpp)
target_include_directories(TestDataStructureAmrWithCommMpi PUBLIC
    ${MPI_CXX_INCLUDE_DIRS}
    ${Python_INCLUDE_DIRS}
    ${CMAKE_SOURCE_DIR}/include
)
target_link_libraries(TestDataStructureAmrWithCommMpi PUBLIC
    gtest_main
    MPI::MPI_CXX
    ${Python_LIBRARIES} Python::NumPy
    yt
)
else()
add_executable(TestDataStructureAmr test_data_structure_amr.cpp)
target_include_directories(TestDataStructureAmr PUBLIC
        ${Python_INCLUDE_DIRS}
        ${CMAKE_SOURCE_DIR}/include
)
target_link_libraries(TestDataStructureAmr PUBLIC
        gtest_main
        ${Python_LIBRARIES} Python::NumPy
        yt
)
endif()

endif () # if (LIBYT_RUN_TEST)
