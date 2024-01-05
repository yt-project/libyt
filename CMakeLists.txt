cmake_minimum_required(VERSION 3.15)

###### PROJECT Info ####################################################################################################
project(LIBYT_PROJECT VERSION 0.1.0 DESCRIPTION "In situ Python analysis tool using yt and Python")

## set options ##
include(CMakeDependentOption)
option(SERIAL_MODE       "Compile library for serial process"                                                 OFF)
option(INTERACTIVE_MODE  "Use interactive mode"                                                               OFF)
option(JUPYTER_KERNEL    "Use Jupyter notebook interface"                                                     OFF)
option(SUPPORT_TIMER     "Support time profiling"                                                             OFF)

## set paths ##
set(PYTHON_PATH    "" CACHE PATH "Path to Python installation prefix (Always)")
set(MPI_PATH       "" CACHE PATH "Path to MPI installation prefix (-DSERIAL_MODE=OFF)")
set(READLINE_PATH  "" CACHE PATH "Path to Readline installation prefix (-DINTERACTIVE_MODE=ON)")
set(nlohmann_json_DIR "" CACHE PATH "Path to nlohmann_jsonConfig.cmake (-DJUPYTER_KERNEL=ON)")
set(cppzmq_DIR        "" CACHE PATH "Path to cppzmqConfig.cmake (-DJUPYTER_KERNEL=ON)")
set(xtl_DIR           "" CACHE PATH "Path to xtlConfig.cmake (-DJUPYTER_KERNEL=ON)")
set(xeus_DIR          "" CACHE PATH "Path to xeusConfig.cmake (-DJUPYTER_KERNEL=ON)")
set(xeus-zmq_DIR      "" CACHE PATH "Path to xeus-zmqConfig.cmake (-DJUPYTER_KERNEL=ON)")
set(ZeroMQ_DIR        "" CACHE PATH "Path to ZeroMQConfig.cmake (-DJUPYTER_KERNEL=ON)")

###### COMPILATION (DO NOT TOUCH) ######################################################################################
## check options ##
if (JUPYTER_KERNEL)
    if (NOT INTERACTIVE_MODE)
        message(STATUS "JUPYTER_KERNEL requires INTERACTIVE_MODE to be ON, INTERACTIVE_MODE is ON")
        set(INTERACTIVE_MODE ON)
    endif ()
endif ()

## cpp version ##
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

## build static/shared library ##
option(BUILD_SHARED_LIBS "Building using shared libraries"    ON )

## find dependencies ##
if (NOT SERIAL_MODE)
    set(MPI_HOME ${MPI_PATH})
    find_package(MPI REQUIRED)
endif ()

if (JUPYTER_KERNEL)
    find_package(xeus-zmq 1.0 REQUIRED)
endif ()

set(Python_ROOT_DIR ${PYTHON_PATH})
find_package(Python COMPONENTS Development NumPy REQUIRED)

## config files ##
if (JUPYTER_KERNEL)
    set(LIBYT_KERNELSPEC_RELATIVE_PATH "share/jupyter/kernels/libyt_kernel")
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/${LIBYT_KERNELSPEC_RELATIVE_PATH}/kernel.json
                   ./${LIBYT_KERNELSPEC_RELATIVE_PATH}/kernel.json COPYONLY)
endif ()

## sub directory ##
add_subdirectory(src)                       # for library
add_subdirectory(example)                   # for example exe

###### INSTALLATION RULE ###############################################################################################
if (JUPYTER_KERNEL)
    include(GNUInstallDirs)
    install(FILES ${CMAKE_CURRENT_BINARY_DIR}/${LIBYT_KERNELSPEC_RELATIVE_PATH}/kernel.json
            DESTINATION "${LIBYT_KERNELSPEC_RELATIVE_PATH}")
endif ()
