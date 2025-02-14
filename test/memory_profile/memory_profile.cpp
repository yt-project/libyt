#include <valgrind/valgrind.h>

#include <iostream>

#ifndef SERIAL_MODE
#include <mpi.h>
#endif

int main(int argc, char* argv[]) {
    int my_rank;
    int my_size;
#ifndef SERIAL_MODE
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &my_size);
#else
    my_rank = 0;
    my_size = 1;
#endif

    const int kLenValgrind = 100;
    char valgrind[kLenValgrind];
    const char* suffix = ".mem_prof";

    int* ptr_leak = new int[100];

    snprintf(valgrind, kLenValgrind, "detailed_snapshot MPI%dAfterFree%d%s\0", 0, 0, suffix);
    VALGRIND_MONITOR_COMMAND(valgrind);

#ifndef SERIAL_MODE
    MPI_Finalize();
#endif

    return 0;
}