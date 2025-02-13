if (LIBYT_RUN_TEST)
    add_subdirectory(unit_test)
endif ()

if (LIBYT_RUN_MEMORY_PROFILE)
    add_subdirectory(memory_profile)
endif ()