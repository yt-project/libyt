// define DEFINE_GLOBAL since this file **defines** all global variables
#define DEFINE_GLOBAL

#include "yt_combo.h"

#undef DEFINE_GLOBAL

#include "libyt.h"

static void init_yt_long_mpi_type();

static void init_yt_hierarchy_mpi_type();

static void init_yt_rma_grid_info_mpi_type();

static void init_yt_rma_particle_info_mpi_type();

static void init_general_info();

static void init_func_status_list();

//-------------------------------------------------------------------------------------------------------
// Function    :  yt_init
// Description :  Initialize libyt
//
// Note        :  1. Input "param_libyt" will be backed up to a libyt global variable
//                2. This function should not be called more than once (even if yt_finalize has been called)
//                   since some extensions (e.g., NumPy) may not work properly.
//                3. Initialize general info and user-defined MPI data type.
//
// Parameter   :  argc        : Argument count
//                argv        : Argument vector
//                param_libyt : libyt runtime parameters
//
// Return      :  YT_SUCCESS or YT_FAIL
//-------------------------------------------------------------------------------------------------------
int yt_init(int argc, char *argv[], const yt_param_libyt *param_libyt) {
#ifdef SUPPORT_TIMER
    // initialize timer
    char filename[50];
    sprintf(filename, "RecordTime_%d", g_myrank);
    g_timer = new Timer(filename);
    // start timer.
    g_timer->record_time("yt_init", 0);
#endif

    // yt_init should only be called once
    static int init_count = 0;
    init_count++;

    // still need to check "init_count" since yt_finalize() will set "g_param_libyt.libyt_initialized = false"
    if (g_param_libyt.libyt_initialized || init_count >= 2)
        YT_ABORT("yt_init() should not be called more than once!\n");


    // store user-provided parameters to a libyt internal variable
    // --> better do it **before** calling any log function since they will query g_param_libyt.verbose
    g_param_libyt.verbose = param_libyt->verbose;
    g_param_libyt.script = param_libyt->script;
    g_param_libyt.counter = param_libyt->counter;   // useful during restart, where the initial counter can be non-zero
    g_param_libyt.check_data = param_libyt->check_data;

    log_info("Initializing libyt ...\n");
    log_info("   verbose = %d\n", g_param_libyt.verbose);
    log_info("    script = %s\n", g_param_libyt.script);
    log_info("   counter = %ld\n", g_param_libyt.counter);
    log_info("check_data = %s\n", (g_param_libyt.check_data ? "true" : "false"));

    // create libyt module, should be before init_python
    if (create_libyt_module() == YT_FAIL) return YT_FAIL;

    // initialize Python interpreter
    if (init_python(argc, argv) == YT_FAIL) return YT_FAIL;

    // import libyt and inline python script.
    if (init_libyt_module() == YT_FAIL) return YT_FAIL;

    // Initialize general info: mpi size and rank ...
    init_general_info();

#ifdef INTERACTIVE_MODE
    init_func_status_list();
#endif

    // Initialize user-defined MPI data type
    init_yt_long_mpi_type();
    init_yt_hierarchy_mpi_type();
    init_yt_rma_grid_info_mpi_type();
    init_yt_rma_particle_info_mpi_type();

    g_param_libyt.libyt_initialized = true;

#ifdef SUPPORT_TIMER
    // end timer.
    g_timer->record_time("yt_init", 1);
#endif

    return YT_SUCCESS;

} // FUNCTION : yt_init


//-------------------------------------------------------------------------------------------------------
// Function    :  init_func_status_list
// Description :  Initialize func_status_list
//
// Note        :  1. Use only inside yt_init
//                2. Find functions and their bodies in inline script, add them to libyt.interactive_mode["func_body"]
//                3. Add function name to g_func_status_list.
//
// Parameter   :  None
//
// Return      :  None
//-------------------------------------------------------------------------------------------------------
#ifdef INTERACTIVE_MODE
static void init_func_status_list() {
    // get function and its body in inline script, store in libyt.interactive_mode dict
    // libyt.interactive_mode["func_body"] = {func_name: [func_body line 1, line 2, ...] }
    int command_width = 400 + strlen(g_param_libyt.script);
    char *command = (char*) malloc(command_width * sizeof(char));
    sprintf(command, "libyt.interactive_mode[\"func_body\"] = {}\n"
                     "libyt.interactive_mode[\"temp\"] = []\n"
                     "for key in libyt.interactive_mode[\"script_globals\"].keys():\n"
                     "    if inspect.isfunction(libyt.interactive_mode[\"script_globals\"][key]):\n"
                     "        libyt.interactive_mode[\"temp\"].append(key.encode(\"utf-8\"))\n"
                     "        libyt.interactive_mode[\"func_body\"][key] = "
                     "        eval(\"inspect.getsource(%s.\" + str(key) +\").split('\\\\n')\")\n",
                     g_param_libyt.script);

    if ( PyRun_SimpleString( command ) == 0 ) log_debug("Load inline function body ... done\n");
    else                                      log_debug("Load inline function body ... failed\n");

    free(command);

    // get all function name, and build up g_func_status_list
    PyObject *py_func_name_list = PyDict_GetItemString(g_py_interactive_mode, "temp");

    for (Py_ssize_t py_index=0; py_index<PyList_Size(py_func_name_list); py_index++) {
        char *func_name = PyBytes_AsString(PyList_GetItem(py_func_name_list, py_index));
        g_func_status_list.add_new_func(func_name);
    }

    // clean up
    PyRun_SimpleString("del libyt.interactive_mode[\"temp\"]");
}
#endif

static void init_general_info() {
    MPI_Comm_size(MPI_COMM_WORLD, &g_mysize);
    MPI_Comm_rank(MPI_COMM_WORLD, &g_myrank);
}

static void init_yt_long_mpi_type() {
    int length[1] = {1};
    const MPI_Aint displacements[1] = {0};
    MPI_Datatype types[1] = {MPI_LONG};
    MPI_Type_create_struct(1, length, displacements, types, &yt_long_mpi_type);
    MPI_Type_commit(&yt_long_mpi_type);
}

static void init_yt_hierarchy_mpi_type() {
    int lengths[7] = {3, 3, 1, 1, 3, 1, 1};
    const MPI_Aint displacements[7] = {0,
                                       3 * sizeof(double),
                                       6 * sizeof(double),
                                       6 * sizeof(double) + sizeof(long),
                                       6 * sizeof(double) + 2 * sizeof(long),
                                       6 * sizeof(double) + 2 * sizeof(long) + 3 * sizeof(int),
                                       6 * sizeof(double) + 2 * sizeof(long) + 4 * sizeof(int)};
    MPI_Datatype types[7] = {MPI_DOUBLE, MPI_DOUBLE, MPI_LONG, MPI_LONG, MPI_INT, MPI_INT, MPI_INT};
    MPI_Type_create_struct(7, lengths, displacements, types, &yt_hierarchy_mpi_type);
    MPI_Type_commit(&yt_hierarchy_mpi_type);
}

static void init_yt_rma_grid_info_mpi_type() {
    int lengths[5] = {1, 1, 1, 1, 3};
    const MPI_Aint displacements[5] = {0,
                                       1 * sizeof(long),
                                       1 * sizeof(long) + 1 * sizeof(MPI_Aint),
                                       1 * sizeof(long) + 1 * sizeof(MPI_Aint) + 1 * sizeof(int),
                                       1 * sizeof(long) + 1 * sizeof(MPI_Aint) + 2 * sizeof(int)};
    MPI_Datatype types[5] = {MPI_LONG, MPI_AINT, MPI_INT, MPI_INT, MPI_INT};
    MPI_Type_create_struct(5, lengths, displacements, types, &yt_rma_grid_info_mpi_type);
    MPI_Type_commit(&yt_rma_grid_info_mpi_type);
}

static void init_yt_rma_particle_info_mpi_type() {
    int lengths[4] = {1, 1, 1, 1};
    const MPI_Aint displacements[4] = {0,
                                       1 * sizeof(long),
                                       1 * sizeof(long) + 1 * sizeof(MPI_Aint),
                                       2 * sizeof(long) + 1 * sizeof(MPI_Aint)};
    MPI_Datatype types[4] = {MPI_LONG, MPI_AINT, MPI_LONG, MPI_INT};
    MPI_Type_create_struct(4, lengths, displacements, types, &yt_rma_particle_info_mpi_type);
    MPI_Type_commit(&yt_rma_particle_info_mpi_type);
}