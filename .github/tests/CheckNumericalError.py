# [Name] Check Numerical Error
# [Description]
# This script is use to check the test in .github/tests/DataIOTest.py def test_function function.
# It will get the numerical difference in data read by libyt data IO and data originally in C++
# array. If the difference in a cell of a grid in one step is greater than 1e-6, then it will raise
# DataIOTestFailed error.
# [Usage]
# python CheckNumericalError.py MPI_SIZE

import pandas as pd
import sys

criteria = 1e-6

class DataIOTestFailed(Exception):
    """
    Raised when the grid data read by libyt DataIO and
    grid data in C++ array in a cell differ more than
    1e-6.
    """
    pass

for rank in range(len(int(sys.argv[1]))):
    filepath = "MPI" + str(rank) + "_result.txt"
    df = pd.read_csv(filepath, delimiter="\n", header=None)
    df = df.to_numpy().flatten()
    for i in range(len(df)):
        if df[i] > criteria:
            raise DataIOTestFailed

