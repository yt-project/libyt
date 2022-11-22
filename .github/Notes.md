# GitHub Action

## Usage of Each Folder
- **tests**: Store test related files. 
  - Test file will be run as inline python script in `libyt/example.cpp`. File name ends with `Test`.
- **tools**: Miscellaneous.
- **workflows**: Store GitHub Action yaml script.

## Tests

### DataIO
- **Workflows**: `.github/workflows/build-test.yml`
- **Test File**: `.github/tests/DataIOTest.py`
- **Description**:
  - Check if `libyt/example` can successfully run in serial and in parallel using `mpirun`.
  - Check if data read by `libyt` is the same as data original in C++ array.