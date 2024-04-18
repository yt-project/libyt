/*
 * [Description]
 *   The quick start initializes `libyt`, runs Python functions defined in inline Python script,
 *   activates Python prompt, and then finalizes it.
 *   It does not load any data from C/C++ application to Python yet.
 *   The Python prompt is just raw Python interpreter activate by libyt API.
 *
 *   After reading quick start, we will have an overview of how to use `libyt` for in situ analysis.
 */
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

    // ==========================================================
    // Execute Python functions and activate Python entry points
    // ==========================================================
    if (yt_run_Function("print_hello_world") != YT_SUCCESS) {
        exit(EXIT_FAILURE);
    }

    if (yt_run_FunctionArguments("print_args", 3, "\'1\'", "2", "3.0") != YT_SUCCESS) {
        exit(EXIT_FAILURE);
    }

    // Activate Python prompt
    if (yt_run_InteractiveMode("LIBYT_STOP") != YT_SUCCESS) {
        exit(EXIT_FAILURE);
    }

    // Activate file-based Python prompt where inputs and outputs are all present through files
    if (yt_run_ReloadScript("LIBYT_RELOAD", "RELOAD", "test_reload.py") != YT_SUCCESS) {
        fprintf(stderr, "ERROR: yt_run_ReloadScript failed!\n");
        exit(EXIT_FAILURE);
    }

    // Activate libyt Jupyter kernel and enable access through Jupyter Notebook / JupyterLab
    if (yt_run_JupyterKernel("LIBYT_JUPYTER", false) != YT_SUCCESS) {
        fprintf(stderr, "ERROR: yt_run_JupyterKernel failed!\n");
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