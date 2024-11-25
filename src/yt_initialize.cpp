// define DEFINE_GLOBAL since this file **defines** all global variables
#define DEFINE_GLOBAL

#include "yt_combo.h"

#undef DEFINE_GLOBAL

#include "libyt.h"
#include "libyt_process_control.h"

static void print_libyt_info();
#ifndef SERIAL_MODE
static void init_yt_long_mpi_type();
static void init_yt_hierarchy_mpi_type();
static void init_yt_rma_grid_info_mpi_type();
static void init_yt_rma_particle_info_mpi_type();
#endif

//-------------------------------------------------------------------------------------------------------
// Function    :  yt_initialize
// Description :  Initialize libyt
//
// Note        :  1. Input "param_libyt" will be backed up to a libyt global variable
//                2. This function should not be called more than once (even if yt_finalize has been called)
//                   since some extensions (e.g., NumPy) may not work properly.
//                3. Initialize general info, user-defined MPI data type, and LibytProcessControl
//
// Parameter   :  argc        : Argument count
//                argv        : Argument vector
//                param_libyt : libyt runtime parameters
//
// Return      :  YT_SUCCESS or YT_FAIL
//-------------------------------------------------------------------------------------------------------
int yt_initialize(int argc, char* argv[], const yt_param_libyt* param_libyt) {
    LibytProcessControl::Get().Initialize();

    SET_TIMER(__PRETTY_FUNCTION__);

    // yt_initialize should only be called once
    static int init_count = 0;
    init_count++;

    // still need to check "init_count" since yt_finalize() will set check point libyt_initialized = false"
    if (LibytProcessControl::Get().libyt_initialized || init_count >= 2)
        YT_ABORT("yt_initialize() should not be called more than once!\n");

    // store user-provided parameters to a libyt internal variable
    // --> better do it **before** calling any log function since they will query param_libyt.verbose
    LibytProcessControl::Get().param_libyt_.verbose = param_libyt->verbose;
    LibytProcessControl::Get().param_libyt_.script = param_libyt->script;
    LibytProcessControl::Get().param_libyt_.counter =
        param_libyt->counter;  // useful during restart, where the initial counter can be non-zero
    LibytProcessControl::Get().param_libyt_.check_data = param_libyt->check_data;

    log_info("******libyt version******\n");
    log_info("         %d.%d.%d\n", LIBYT_MAJOR_VERSION, LIBYT_MINOR_VERSION, LIBYT_MICRO_VERSION);
    print_libyt_info();
    log_info("*************************\n");

    log_info("Initializing libyt ...\n");
    log_info("   verbose = %d\n", LibytProcessControl::Get().param_libyt_.verbose);
    log_info("    script = %s\n", LibytProcessControl::Get().param_libyt_.script);
    log_info("   counter = %ld\n", LibytProcessControl::Get().param_libyt_.counter);
    log_info("check_data = %s\n", (LibytProcessControl::Get().param_libyt_.check_data ? "true" : "false"));

#ifndef USE_PYBIND11
    // create libyt module, should be before init_python
    if (create_libyt_module() == YT_FAIL) return YT_FAIL;
#endif

    // initialize Python interpreter
    if (init_python(argc, argv) == YT_FAIL) return YT_FAIL;

    // import libyt and inline python script.
    if (init_libyt_module() == YT_FAIL) return YT_FAIL;

#ifndef SERIAL_MODE
    // Initialize user-defined MPI data type
    init_yt_long_mpi_type();
    init_yt_hierarchy_mpi_type();
    init_yt_rma_grid_info_mpi_type();
    init_yt_rma_particle_info_mpi_type();
#endif

    // set python exception hook and set not-yet-done error msg
#if defined(INTERACTIVE_MODE) || defined(JUPYTER_KERNEL)
    if (LibytPythonShell::set_exception_hook() != YT_SUCCESS) return YT_FAIL;
    if (LibytPythonShell::init_not_done_err_msg() != YT_SUCCESS) return YT_FAIL;
    if (LibytPythonShell::init_script_namespace() != YT_SUCCESS) return YT_FAIL;
#endif

    LibytProcessControl::Get().libyt_initialized = true;

    return YT_SUCCESS;

}  // FUNCTION : yt_initialize

static void print_libyt_info() {
#ifdef SERIAL_MODE
    log_info("  SERIAL_MODE: ON\n");
#else
    log_info("  SERIAL_MODE: OFF\n");
#endif

#ifdef INTERACTIVE_MODE
    log_info("  INTERACTIVE_MODE: ON\n");
#else
    log_info("  INTERACTIVE_MODE: OFF\n");
#endif

#ifdef JUPYTER_KERNEL
    log_info("  JUPYTER_KERNEL: ON\n");
#else
    log_info("  JUPYTER_KERNEL: OFF\n");
#endif

#ifdef SUPPORT_TIMER
    log_info("  SUPPORT_TIMER: ON\n");
#else
    log_info("  SUPPORT_TIMER: OFF\n");
#endif
}

#ifndef SERIAL_MODE
static void init_yt_long_mpi_type() {
    SET_TIMER(__PRETTY_FUNCTION__);

    int length[1] = {1};
    const MPI_Aint displacements[1] = {0};
    MPI_Datatype types[1] = {MPI_LONG};
    MPI_Type_create_struct(1, length, displacements, types, &LibytProcessControl::Get().yt_long_mpi_type_);
    MPI_Type_commit(&LibytProcessControl::Get().yt_long_mpi_type_);
}

static void init_yt_hierarchy_mpi_type() {
    SET_TIMER(__PRETTY_FUNCTION__);

    int lengths[7] = {3, 3, 1, 1, 3, 1, 1};
    const MPI_Aint displacements[7] = {0,
                                       3 * sizeof(double),
                                       6 * sizeof(double),
                                       6 * sizeof(double) + sizeof(long),
                                       6 * sizeof(double) + 2 * sizeof(long),
                                       6 * sizeof(double) + 2 * sizeof(long) + 3 * sizeof(int),
                                       6 * sizeof(double) + 2 * sizeof(long) + 4 * sizeof(int)};
    MPI_Datatype types[7] = {MPI_DOUBLE, MPI_DOUBLE, MPI_LONG, MPI_LONG, MPI_INT, MPI_INT, MPI_INT};
    MPI_Type_create_struct(7, lengths, displacements, types, &LibytProcessControl::Get().yt_hierarchy_mpi_type_);
    MPI_Type_commit(&LibytProcessControl::Get().yt_hierarchy_mpi_type_);
}

static void init_yt_rma_grid_info_mpi_type() {
    SET_TIMER(__PRETTY_FUNCTION__);

    int lengths[5] = {1, 1, 1, 1, 3};
    const MPI_Aint displacements[5] = {0, 1 * sizeof(long), 1 * sizeof(long) + 1 * sizeof(MPI_Aint),
                                       1 * sizeof(long) + 1 * sizeof(MPI_Aint) + 1 * sizeof(int),
                                       1 * sizeof(long) + 1 * sizeof(MPI_Aint) + 2 * sizeof(int)};
    MPI_Datatype types[5] = {MPI_LONG, MPI_AINT, MPI_INT, MPI_INT, MPI_INT};
    MPI_Type_create_struct(5, lengths, displacements, types, &LibytProcessControl::Get().yt_rma_grid_info_mpi_type_);
    MPI_Type_commit(&LibytProcessControl::Get().yt_rma_grid_info_mpi_type_);
}

static void init_yt_rma_particle_info_mpi_type() {
    SET_TIMER(__PRETTY_FUNCTION__);

    int lengths[4] = {1, 1, 1, 1};
    const MPI_Aint displacements[4] = {0, 1 * sizeof(long), 1 * sizeof(long) + 1 * sizeof(MPI_Aint),
                                       2 * sizeof(long) + 1 * sizeof(MPI_Aint)};
    MPI_Datatype types[4] = {MPI_LONG, MPI_AINT, MPI_LONG, MPI_INT};
    MPI_Type_create_struct(4, lengths, displacements, types,
                           &LibytProcessControl::Get().yt_rma_particle_info_mpi_type_);
    MPI_Type_commit(&LibytProcessControl::Get().yt_rma_particle_info_mpi_type_);
}
#endif
