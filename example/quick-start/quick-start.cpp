#include <iostream>

#ifndef SERIAL_MODE
#include <mpi.h>
#endif

#include "libyt.h"

int main(int argc, char* argv[]) {
    int myrank;
    int nrank;
#ifndef SERIAL_MODE
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    MPI_Comm_size(MPI_COMM_WORLD, &nrank);
#else
    myrank = 0;
    nrank = 1;
#endif

    // =======================================
    // Initialize libyt
    // =======================================
    yt_param_libyt param_libyt;
    param_libyt.verbose = YT_VERBOSE_INFO;  // libyt log level
    param_libyt.script = "inline_script";   // inline python script, excluding ".py"
    param_libyt.check_data = false;         // check passed in data or not
    if (yt_initialize(argc, argv, &param_libyt) != YT_SUCCESS) {
        fprintf(stderr, "ERROR: yt_initialize() failed!\n");
        exit(EXIT_FAILURE);
    }

    // =======================================
    // Execute Python Function
    // =======================================
    if (yt_run_Function("print_hello_world") != YT_SUCCESS) {
        exit(EXIT_FAILURE);
    }

    if (yt_run_FunctionArguments("print_args", 3, "\'1\'", "2", "3.0") != YT_SUCCESS) {
        exit(EXIT_FAILURE);
    }

    if (yt_run_InteractiveMode("LIBYT_STOP") != YT_SUCCESS) {
        exit(EXIT_FAILURE);
    }

    // =======================================
    // Finalize libyt
    // =======================================
    yt_finalize();

#ifndef SERIAL_MODE
    MPI_Finalize();
#endif

    return EXIT_SUCCESS;
}