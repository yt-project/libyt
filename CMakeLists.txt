cmake_minimum_required(VERSION 3.15)

### PROJECT Info ###
project(LIBYT_PROJECT VERSION 0.1 DESCRIPTION "In situ Python analysis tool using yt")

# set option
option(SERIAL            "Compile library for serial process" OFF)
option(INTERACTIVE_MODE  "Use interactive mode"               OFF)
option(SUPPORT_TIMER     "Support timer"                      OFF)
option(BUILD_SHARED_LIBS "Building using shared libraries"    ON )

# set path
set(MPI_PATH       "")
set(PYTHON_PATH    "")
set(PYTHON_VERSION "")
set(NUMPY_PATH     "")
set(READLINE_PATH  "")

### COMPILATION ###
add_subdirectory(src)                      # for library
#add_subdirectory(example)                  # for example exe
#configure_file(libyt.pc.in libyt.pc @ONLY) # for pkgconfig

### MyLib PACKAGING ###
# Only packaging library works
include(InstallRequiredSystemLibraries)
set(MYPROJECT_SRC "${CMAKE_CURRENT_SOURCE_DIR}")
set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE") # set license
set(CPACK_PACKAGE_VERSION_MAJOR "${LIBYT_PROJECT_VERSION_MAJOR}")      # major version
set(CPACK_PACKAGE_VERSION_MINOR "${LIBYT_PROJECT_VERSION_MINOR}")      # minor version
set(CPACK_SOURCE_GENERATOR "TGZ")                                      # packed format

include(CPack)
