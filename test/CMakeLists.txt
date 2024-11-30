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
if (NOT SERIAL_MODE)
    find_package(MPI REQUIRED)
endif ()

# Test scripts and discover tests
include(GoogleTest)

## Test template
add_executable(TestTemplate test_template.cpp)
target_link_libraries(TestTemplate PUBLIC gtest_main)
gtest_discover_tests(TestTemplate)

## Test Comm MPI
if (NOT SERIAL_MODE)
add_executable(TestCommMPI test_comm_mpi.cpp)
target_include_directories(TestCommMPI PUBLIC
    ${MPI_INCLUDE_PATH}
    ${CMAKE_SOURCE_DIR}/include
)
target_link_libraries(TestCommMPI PUBLIC
    gtest_main
    MPI::MPI_CXX
    yt
)
gtest_discover_tests(TestCommMPI)
endif ()

endif () # if (LIBYT_RUN_TEST)
