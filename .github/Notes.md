# GitHub Action

## Usage of Each Folder
- **data**: Store data for libyt IO to compare to.
- **tests**: Store test related files. 
  - Test file will be run as inline python script in `libyt/example.cpp`. File name ends with `Test`.
  - Files to check if a test passes or not will run after `example` finishes. File name starts with `Check`.
- **tools**: Miscellaneous.
- **workflows**: Store GitHub Action yaml script.

## Tests

### DataIO
- **Workflows**: `.github/workflows/build-test.yml`
- **Test File**: `.github/tests/DataIOTest.py`
- **Check File**: `.github/tests/CheckNumericalError.py`
- **Description**:
  - Check if `libyt/example` can successfully run in serial and in parallel.
  - Check if data read by `libyt` is the same in data original in C++ array the same.