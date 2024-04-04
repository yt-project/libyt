include(FetchContent)

if (JUPYTER_KERNEL)

## get dependency
set(FETCHCONTENT_BASE_DIR "${PROJECT_SOURCE_DIR}/vendor/nlohmann_json")
FetchContent_Declare(
        nlohmann_json
        GIT_REPOSITORY "https://github.com/nlohmann/json.git"
        GIT_TAG v3.11.3
)

set(FETCHCONTENT_BASE_DIR "${PROJECT_SOURCE_DIR}/vendor/xtl")
FetchContent_Declare(
        xtl
        GIT_REPOSITORY "https://github.com/xtensor-stack/xtl.git"
        GIT_TAG 0.7.7
)

set(FETCHCONTENT_BASE_DIR "${PROJECT_SOURCE_DIR}/vendor/cppzmq")
FetchContent_Declare(
        cppzmq
        GIT_REPOSITORY "https://github.com/zeromq/cppzmq.git"
        GIT_TAG v4.10.0
)

set(FETCHCONTENT_BASE_DIR "${PROJECT_SOURCE_DIR}/vendor/xeus")
FetchContent_Declare(
        xeus
        GIT_REPOSITORY "https://github.com/jupyter-xeus/xeus.git"
        GIT_TAG 3.2.0
)

set(FETCHCONTENT_BASE_DIR "${PROJECT_SOURCE_DIR}/vendor/xeus-zmq")
FetchContent_Declare(
        xeus-zmq
        GIT_REPOSITORY "https://github.com/jupyter-xeus/xeus-zmq.git"
        GIT_TAG 1.3.0
)

## populate
FetchContent_GetProperties(nlohmann_json)
if (NOT nlohmann_json_POPULATED)
    FetchContent_Populate(nlohmann_json)
    option(JSON_Install ON)
    add_subdirectory(${nlohmann_json_SOURCE_DIR} ${nlohmann_json_BINARY_DIR})
endif ()

FetchContent_GetProperties(cppzmq)
if (NOT cppzmq_POPULATED)
    FetchContent_Populate(cppzmq)
    option(CPPZMQ_BUILD_TESTS OFF)
    add_subdirectory(${cppzmq_SOURCE_DIR} ${cppzmq_BINARY_DIR})
endif ()

FetchContent_MakeAvailable(xtl)

# ref: https://stackoverflow.com/a/35345933
if (NOT DEFINED _DISABLE_INSTALLATION)
    # This variable is responsible for installation disabling.
    set(_DISABLE_INSTALLATION FALSE)

    # Replace install() with conditional installation.
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

FetchContent_GetProperties(xeus)
if (NOT xeus_POPULATED)
    FetchContent_Populate(xeus)
	set(_DISABLE_INSTALLATION TRUE)
    set(_DISABLE_EXPORT TRUE)
	add_subdirectory(${xeus_SOURCE_DIR} ${xeus_BINARY_DIR})
endif ()

FetchContent_GetProperties(xeus-zmq)
if (NOT xeus-zmq_POPULATED)
    FetchContent_Populate(xeus-zmq)
	set(_DISABLE_INSTALLATION TRUE)
    set(_DISABLE_EXPORT TRUE)
	add_subdirectory(${xeus-zmq_SOURCE_DIR} ${xeus-zmq_BINARY_DIR})
endif ()

endif () # if (JUPYTER_KERNEL)
