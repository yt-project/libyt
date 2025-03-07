# Get Valgrind
find_path(
  Valgrind_ROOT_DIR
  NAMES include/valgrind/valgrind.h
  HINTS ${VALGRIND_PATH}
)

find_path(
  Valgrind_INCLUDE_DIR
  NAMES valgrind/valgrind.h
  HINTS ${Valgrind_ROOT_DIR}/include
)

find_path(
  Valgrind_LIBRARY
  NAMES valgrind
  HINTS ${Valgrind_ROOT_DIR}/lib
)

if (Valgrind_INCLUDE_DIR AND Valgrind_LIBRARY)
  if (Valgrind_ROOT_DIR STREQUAL VALGRIND_PATH)
    set(ENV{PKG_CONFIG_PATH} "${Valgrind_ROOT_DIR}/lib/pkgconfig:$ENV{PKG_CONFIG_PATH}")
  endif ()
  execute_process(
    COMMAND pkg-config --libs valgrind --static
    OUTPUT_VARIABLE Valgrind_LIBFLAGS
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )
  set(Valgrind_LIBFLAGS
      ${Valgrind_LIBFLAGS}
      PARENT_SCOPE
  )
  message(STATUS "Found valgrind: ${Valgrind_LIBRARY} (link flags: ${Valgrind_LIBFLAGS})")
else ()
  message(
    FATAL_ERROR
      "Unable to find valgrind, please visit http://www.valgrind.org/ for more information..."
  )
endif ()

# Get mpi
if (NOT SERIAL_MODE)
  find_package(MPI REQUIRED)
endif ()

# Compile memory profile test program
if (NOT SERIAL_MODE)
  add_executable(MpiMemoryProfile memory_profile.cpp)
  target_link_libraries(MpiMemoryProfile PRIVATE MPI::MPI_CXX yt)
  target_include_directories(
    MpiMemoryProfile PRIVATE ${MPI_CXX_INCLUDE_DIRS} ${Valgrind_INCLUDE_DIR}
                             ${CMAKE_SOURCE_DIR}/include
  )
else ()
  add_executable(MemoryProfile memory_profile.cpp)
  target_link_libraries(MemoryProfile PRIVATE yt)
  target_include_directories(
    MemoryProfile PRIVATE ${Valgrind_INCLUDE_DIR} ${CMAKE_SOURCE_DIR}/include
  )
endif ()

configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/test_memory_profile.py
  ${CMAKE_CURRENT_BINARY_DIR}/test_memory_profile.py COPYONLY
)
