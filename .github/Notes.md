# GitHub Action

## File Structure
- **tests**: Store test related files. 
  - Test file will be run as inline python script in `libyt/example.cpp`. File name ends with `Test`.
- **tools**: Miscellaneous.
- **workflows**: Store GitHub Action yaml script.

## Workflows

- `cmake-build-test.yml`: Build different options of libyt on different os.
- `example-test-run.yml`: Make sure the AMR example can run in parallel and in serial modes.
- `unit-test.yml`: Run unit tests. Since some of the units have MPI coupled in them, we need to run the tests in parallel mode and in serial mode. For parallel mode, `openmpi` and `mpich` are tested. For serial mode, `Python3.7` ~ `Python3.14` are tested. All of them have a combination with `-DUSE_PYBIND11=ON` and `-DUSE_PYBIND11=OFF` options.
- `memory-profile.yml`: Run memory profiling tests using valgrind.