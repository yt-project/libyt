### libyt COMPILATION ###
add_library(yt SHARED
        add_dict.cpp allocate_hierarchy.cpp append_grid.cpp check_data.cpp
        define_command.cpp func_status.cpp func_status_list.cpp get_dtype_property.cpp
        init_libyt_module.cpp init_python.cpp logging.cpp print_data.cpp
        yt_commit.cpp yt_finalize.cpp yt_free.cpp yt_get_FieldsPtr.cpp yt_get_GridsPtr.cpp
        yt_get_ParticlesPtr.cpp yt_getGridInfo.cpp yt_initialize.cpp yt_rma_field.cpp
        yt_rma_particle.cpp yt_run.cpp yt_run_InteractiveMode.cpp yt_set_Parameters.cpp
        yt_set_UserParameter.cpp LibytProcessControl.cpp Timer.cpp TimerControl.cpp)

# print options
function(print_option option_name)
    if (${option_name})
        message(STATUS "${option_name} is ON")
    else ()
        message(STATUS "${option_name} is OFF")
    endif ()
endfunction()

print_option(SERIAL_MODE)
print_option(INTERACTIVE_MODE)
print_option(SUPPORT_TIMER)

# define
if (SUPPORT_TIMER)
    target_compile_definitions(yt PUBLIC SUPPORT_TIMER)
endif ()

if (INTERACTIVE_MODE)
    target_compile_definitions(yt PUBLIC INTERACTIVE_MODE)
endif ()

if (SERIAL_MODE)
    target_compile_definitions(yt PUBLIC SERIAL_MODE)
endif ()

# compile option
add_library(libyt_compiler_flags INTERFACE)
target_compile_features(libyt_compiler_flags INTERFACE cxx_std_11)

# include dir
target_include_directories(yt PRIVATE
        ${PROJECT_SOURCE_DIR}/include
        $<$<NOT:$<BOOL:${SERIAL_MODE}>>:${MPI_INCLUDE_PATH}>
        ${Python_INCLUDE_DIRS}
        $<$<AND:$<BOOL:${INTERACTIVE_MODE}>,$<NOT:$<STREQUAL:${READLINE_PATH},>>>:${READLINE_PATH}/include>)

# link lib
target_link_directories(yt PRIVATE
        $<$<AND:$<BOOL:${INTERACTIVE_MODE}>,$<NOT:$<STREQUAL:${READLINE_PATH},>>>:${READLINE_PATH}/lib>)
target_link_libraries(yt PRIVATE
        $<$<NOT:$<BOOL:${SERIAL_MODE}>>:MPI::MPI_CXX>
        ${Python_LIBRARIES}
        Python::NumPy
        $<$<BOOL:${INTERACTIVE_MODE}>:readline>)
target_link_libraries(yt PUBLIC
        libyt_compiler_flags)

# set properties
set(LIBYT_PUBLIC_HEADER
        "${PROJECT_SOURCE_DIR}/include/libyt.h;"
        "${PROJECT_SOURCE_DIR}/include/yt_type.h;"
        "${PROJECT_SOURCE_DIR}/include/yt_type_param_libyt.h;"
        "${PROJECT_SOURCE_DIR}/include/yt_type_param_yt.h;"
        "${PROJECT_SOURCE_DIR}/include/yt_type_grid.h;"
        "${PROJECT_SOURCE_DIR}/include/yt_type_field.h;"
        "${PROJECT_SOURCE_DIR}/include/yt_type_particle.h;"
        "${PROJECT_SOURCE_DIR}/include/yt_type_array.h;"
        "${PROJECT_SOURCE_DIR}/include/yt_macro.h;")
set_target_properties(yt PROPERTIES
        VERSION ${PROJECT_VERSION}                       # version
        SOVERSION ${PROJECT_VERSION_MAJOR}               # soversion (.so.x.y.z)
        PUBLIC_HEADER "${LIBYT_PUBLIC_HEADER}"           # APIs
        POSITION_INDEPENDENT_CODE ${BUILD_SHARED_LIBS})  # build PIC

### libyt INSTALLATION RULE ###
include(GNUInstallDirs)
install(TARGETS yt
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
