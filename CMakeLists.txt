cmake_minimum_required(VERSION 3.15)

###### PROJECT Info ####################################################################################################
project(LIBYT_PROJECT VERSION 0.1.0 DESCRIPTION "In situ Python analysis tool using yt and Python")

## set options ##
option(SERIAL_MODE       "Compile library for serial process"                                                 OFF)
option(INTERACTIVE_MODE  "Use interactive mode"                                                               OFF)
option(JUPYTER_KERNEL    "Use Jupyter notebook interface"                                                     OFF)
option(SUPPORT_TIMER     "Support time profiling"                                                             OFF)
option(USE_PYBIND11      "Use pybind11"                                                                       ON )

## set paths ##
# It is recommended that we always provide PYTHON_PATH and MPI_PATH.
# libyt will also try to find the package if the paths weren't provided.
set(PYTHON_PATH    "" CACHE PATH "Path to Python installation prefix (Always)")
set(MPI_PATH       "" CACHE PATH "Path to MPI installation prefix (-DSERIAL_MODE=OFF)")
set(READLINE_PATH  "" CACHE PATH "Path to Readline installation prefix (-DINTERACTIVE_MODE=ON)")

## set paths (optional) ##
# libyt will get the dependencies if it cannot find one when needed.
set(PYBIND11_PATH     "" CACHE PATH "Path to pybind11Config.cmake (Always)")
set(nlohmann_json_DIR "" CACHE PATH "Path to nlohmann_jsonConfig.cmake (-DJUPYTER_KERNEL=ON)")
set(cppzmq_DIR        "" CACHE PATH "Path to cppzmqConfig.cmake (-DJUPYTER_KERNEL=ON)")
set(xtl_DIR           "" CACHE PATH "Path to xtlConfig.cmake (-DJUPYTER_KERNEL=ON)")
set(xeus_DIR          "" CACHE PATH "Path to xeusConfig.cmake (-DJUPYTER_KERNEL=ON)")
set(xeus-zmq_DIR      "" CACHE PATH "Path to xeus-zmqConfig.cmake (-DJUPYTER_KERNEL=ON)")
set(ZeroMQ_DIR        "" CACHE PATH "Path to ZeroMQConfig.cmake (-DJUPYTER_KERNEL=ON)")

###### COMPILATION (DO NOT TOUCH) ######################################################################################
## cpp version ##
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

## build static/shared library ##
option(BUILD_SHARED_LIBS "Building using shared libraries"                                                    ON )

# run test ##
option(LIBYT_RUN_TEST    "Run unit test"                                                                      OFF)

## set path for dependencies ##
set(Python_ROOT_DIR ${PYTHON_PATH})
if (USE_PYBIND11)
    set(pybind11_DIR ${PYBIND11_PATH})
endif ()
if (NOT SERIAL_MODE)
    set(MPI_HOME ${MPI_PATH})
endif ()
if (LIBYT_RUN_TEST)
    include(CTest)
endif ()

## sub directory ##
add_subdirectory(vendor)                    # for vendoring
add_subdirectory(src)                       # for library
add_subdirectory(share)                     # for share
add_subdirectory(example)                   # for example exe
add_subdirectory(test)                      # for testing
