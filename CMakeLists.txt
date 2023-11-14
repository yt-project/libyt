cmake_minimum_required(VERSION 3.15)

###### PROJECT Info ####################################################################################################
project(LIBYT_PROJECT VERSION 0.1.0 DESCRIPTION "In situ Python analysis tool using yt and Python")

## set options ##
option(SERIAL_MODE       "Compile library for serial process" OFF)
option(INTERACTIVE_MODE  "Use interactive mode"               OFF)
option(SUPPORT_TIMER     "Support timer"                      OFF)

## set paths ##
set(PYTHON_PATH    "" CACHE PATH "Path to Python installation prefix")
set(MPI_PATH       "" CACHE PATH "Path to MPI installation prefix")
set(READLINE_PATH  "" CACHE PATH "Path to Readline installation prefix")

###### COMPILATION (DO NOT TOUCH) ######################################################################################
## build static/shared library ##
option(BUILD_SHARED_LIBS "Building using shared libraries"    ON )

## find dependencies ##
if (NOT SERIAL_MODE)
    set(MPI_HOME ${MPI_PATH})
    find_package(MPI REQUIRED)
endif ()

set(Python_ROOT_DIR ${PYTHON_PATH})
find_package(Python COMPONENTS Development NumPy REQUIRED)

## sub directory ##
add_subdirectory(src)                       # for library
#add_subdirectory(example)                  # for example exe

### libyt PACKAGING ###
#include(InstallRequiredSystemLibraries)
#set(MYPROJECT_SRC "${CMAKE_CURRENT_SOURCE_DIR}")
#set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE") # set license
#set(CPACK_PACKAGE_VERSION_MAJOR "${LIBYT_PROJECT_VERSION_MAJOR}")      # major version
#set(CPACK_PACKAGE_VERSION_MINOR "${LIBYT_PROJECT_VERSION_MINOR}")      # minor version
#set(CPACK_SOURCE_GENERATOR "TGZ")                                      # packed format
#
#include(CPack)
