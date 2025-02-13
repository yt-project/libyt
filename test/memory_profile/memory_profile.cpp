#include <valgrind/valgrind.h>

#include <iostream>

int main() {
    const int kLenValgrind = 100;
    char valgrind[kLenValgrind];

    int* ptr_leak = new int[100];

    snprintf(valgrind, kLenValgrind, "detailed_snapshot MPI%dAfterFree%d\0", 0, 0);
    VALGRIND_MONITOR_COMMAND(valgrind);

    return 0;
}