import os
import argparse

class Bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def extract_value_from_file(filename: str, attribute: str) -> dict:
    """extract_value_from_file
    Assume attribute in a file is in a format 'attribute=value'.
    """

    # make sure file exist and read the file
    if not os.path.exists(filename):
        print(f"{Bcolors.FAIL}File {filename} does not exist {Bcolors.ENDC}")
        raise "No such file"
    with open(filename, "r") as f:
        raw = f.read()
        print(f"Reading file {Bcolors.WARNING} {filename} {Bcolors.ENDC} ... done")

    # find all the matching attribute
    found = -1
    extract_value = []
    while True:
        # find matching attributes
        found = raw.find(attribute, found + 1)
        if found < 0:
            break

        found_newline = raw.find("\n", found)
        extract_value.append(int(raw[found + len(attribute) + 1: found_newline]))

    results = dict()
    results[attribute] = extract_value
    results[attribute + "_diff"] = [extract_value[i] - extract_value[i - 1] for i in range(1, len(extract_value))]
    print(f"Extracting attribute {Bcolors.OKGREEN} {attribute} {Bcolors.ENDC} in {filename}  ... done")

    return results

# class ReadmeTable:
#     def __init__(self, title: str, columns: list, rows: list):
#         self.title = title
#         self.column = columns
#         self.row = rows
#
#     def readme_format(self) -> str:
#         output = ""
#         # TODO: print the table
#         return output



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Combine detailed snapshots dumped by valgrind")
    parser.add_argument('--tags', metavar='tags', type=str, nargs='*',
                        help='A list of tags, e.g., "BeforeFree_rank0.mem_prof" has tag "BeforeFree"')
    parser.add_argument('--attr', metavar='attr', type=str, nargs=1,
                        help='Attribute to extract in valgrind massif dump.')
    parser.add_argument('--mpi_size', metavar='mpi_size', type=int, nargs=1,
                        help='MPI size')
    args = parser.parse_args()

    # Extract value
    for tag in args.tags:
        for r in range(args.mpi_size[0]):
            filename = tag + "_rank{}.mem_prof".format(r)
            attr_value = extract_value_from_file(filename, args.attr[0])

            for key in attr_value:
                print("**{}({}_rank{})**:".format(key, tag, r) , attr_value[key], end=" <br>")
