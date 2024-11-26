if (LIBYT_RUN_TEST)
include(FetchContent)

set(LIBYT_GOOGLETEST_FETCH_VERSION       1.14.0 )

FetchContent_Declare(
        googletest
        GIT_REPOSITORY "https://github.com/google/googletest.git"
        GIT_TAG        "v${LIBYT_GOOGLETEST_FETCH_VERSION}"
        GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(googletest)

# Test scripts
add_executable(TestTemplate test_template.cpp)
target_link_libraries(TestTemplate PUBLIC gtest_main)
# TODO: Also need to make sure every components are tested based on the option set by users.
#       (so that maybe they can also test it on their machine, but the test I write should separate all the
#        components so that we user turn off some of the options, only some part of the tests is turned off.)

# Discover tests
include(GoogleTest)
gtest_discover_tests(TestTemplate)

endif () # if (LIBYT_RUN_TEST)
