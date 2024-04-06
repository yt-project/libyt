include(FetchContent)

if (JUPYTER_KERNEL)

## requirements ########################################################################################################
set(LIBYT_NLOHMANN_JSON_REQUIRED_VERSION 3.2.0 )
set(LIBYT_CPPZMQ_REQUIRED_VERSION        4.8.1 )
set(LIBYT_XTL_REQUIRED_VERSION           0.7.0 )
set(LIBYT_XEUS_REQUIRED_VERSION          3.0.0 )
set(LIBYT_XEUS_ZMQ_REQUIRED_VERSION      1.0   )
set(LIBYT_ZEROMQ_REQUIRED_VERSION        4.2.5 )

## prepare and redefine ################################################################################################
# redefining install() and export() to make xeus and xeus-zmq work (ref: https://stackoverflow.com/a/35345933)
if (NOT DEFINED _DISABLE_INSTALLATION)
    set(_DISABLE_INSTALLATION FALSE)
    macro (install)
        if (NOT _DISABLE_INSTALLATION)
            _install(${ARGN})
        endif ()
    endmacro ()
endif ()

if (NOT DEFINED _DISABLE_EXPORT)
    set(_DISABLE_EXPORT FALSE)
    macro(export)
        if (NOT _DISABLE_EXPORT)
            _export(${ARGN})
        endif ()
    endmacro()
endif ()

## get dependencies and populate #######################################################################################
find_package(nlohmann_json ${LIBYT_NLOHMANN_JSON_REQUIRED_VERSION} QUIET)
if (NOT nlohmann_json_FOUND)
    message(STATUS "Unable to find nlohmann_json, getting dependency...")
    set(FETCHCONTENT_BASE_DIR "${PROJECT_SOURCE_DIR}/vendor/nlohmann_json")
    FetchContent_Declare(
            nlohmann_json
            GIT_REPOSITORY "https://github.com/nlohmann/json.git"
            GIT_TAG v3.11.3
    )
    FetchContent_GetProperties(nlohmann_json)
    if (NOT nlohmann_json_POPULATED)
        FetchContent_Populate(nlohmann_json)
        option(JSON_Install ON)
        add_subdirectory(${nlohmann_json_SOURCE_DIR} ${nlohmann_json_BINARY_DIR} EXCLUDE_FROM_ALL)
    endif ()
else ()
    message(STATUS "Found nlohmann_json: ${nlohmann_json_DIR} (found version \"${nlohmann_json_VERSION}\")")
endif ()

find_package(xtl ${LIBYT_XTL_REQUIRED_VERSION} QUIET)
if (NOT xtl_FOUND)
    message(STATUS "Unable to find xtl, getting dependency...")
    set(FETCHCONTENT_BASE_DIR "${PROJECT_SOURCE_DIR}/vendor/xtl")
    FetchContent_Declare(
            xtl
            GIT_REPOSITORY "https://github.com/xtensor-stack/xtl.git"
            GIT_TAG 0.7.7
    )
    FetchContent_GetProperties(xtl)
    if (NOT xtl_POPULATED)
        FetchContent_Populate(xtl)
        add_subdirectory(${xtl_SOURCE_DIR} ${xtl_BINARY_DIR} EXCLUDE_FROM_ALL)
    endif ()
else ()
    message(STATUS "Found xtl: ${xtl_DIR} (found version \"${xtl_VERSION}\")")
endif ()

find_package(cppzmq ${LIBYT_CPPZMQ_REQUIRED_VERSION} QUIET)
if (NOT cppzmq_FOUND)
    message(STATUS "Unable to find cppzmq, getting dependency...")
    set(FETCHCONTENT_BASE_DIR "${PROJECT_SOURCE_DIR}/vendor/cppzmq")
    FetchContent_Declare(
            cppzmq
            GIT_REPOSITORY "https://github.com/zeromq/cppzmq.git"
            GIT_TAG v4.10.0
    )
    FetchContent_GetProperties(cppzmq)
    if (NOT cppzmq_POPULATED)
        FetchContent_Populate(cppzmq)
        option(CPPZMQ_BUILD_TESTS OFF)
        add_subdirectory(${cppzmq_SOURCE_DIR} ${cppzmq_BINARY_DIR} EXCLUDE_FROM_ALL)
    endif ()
else ()
    message(STATUS "Found cppzmq: ${cppzmq_DIR} (found version \"${cppzmq_VERSION}\")")
endif ()

find_package(xeus ${LIBYT_XEUS_REQUIRED_VERSION} QUIET)
if (NOT xeus_FOUND)
    message(STATUS "Unable to find xeus, getting dependency...")
    set(FETCHCONTENT_BASE_DIR "${PROJECT_SOURCE_DIR}/vendor/xeus")
    FetchContent_Declare(
            xeus
            GIT_REPOSITORY "https://github.com/jupyter-xeus/xeus.git"
            GIT_TAG 3.2.0
    )
    FetchContent_GetProperties(xeus)
    if (NOT xeus_POPULATED)
        FetchContent_Populate(xeus)
        set(_DISABLE_INSTALLATION_OLD ${_DISABLE_INSTALLATION})
        set(_DISABLE_INSTALLATION TRUE)
        set(_DISABLE_EXPORT TRUE)
        add_subdirectory(${xeus_SOURCE_DIR} ${xeus_BINARY_DIR} EXCLUDE_FROM_ALL)
        set(_DISABLE_INSTALLATION ${_DISABLE_INSTALLATION_OLD})
    endif ()
else ()
    message(STATUS "Found xeus: ${xeus_DIR} (found version \"${xeus_VERSION}\")")
endif ()

find_package(xeus-zmq ${LIBYT_XEUS_ZMQ_REQUIRED_VERSION} QUIET)
if (NOT xeus-zmq_FOUND)
    message(STATUS "Unable to find xeus-zmq, getting dependency...")
    set(FETCHCONTENT_BASE_DIR "${PROJECT_SOURCE_DIR}/vendor/xeus-zmq")
    FetchContent_Declare(
            xeus-zmq
            GIT_REPOSITORY "https://github.com/jupyter-xeus/xeus-zmq.git"
            GIT_TAG 1.3.0
    )
    FetchContent_GetProperties(xeus-zmq)
    if (NOT xeus-zmq_POPULATED)
        FetchContent_Populate(xeus-zmq)
        set(_DISABLE_INSTALLATION_OLD ${_DISABLE_INSTALLATION})
        set(_DISABLE_INSTALLATION TRUE)
        set(_DISABLE_EXPORT TRUE)
        add_subdirectory(${xeus-zmq_SOURCE_DIR} ${xeus-zmq_BINARY_DIR} EXCLUDE_FROM_ALL)
        set(_DISABLE_INSTALLATION ${_DISABLE_INSTALLATION_OLD})
    endif ()
else ()
    message(STATUS "Found xeus-zmq: ${xeus-zmq_DIR} (found version \"${xeus-zmq_VERSION}\")")
endif ()

endif () # if (JUPYTER_KERNEL)
