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


def extract_value_from_file(folder: str or None, filename: str, attribute: str) -> dict:
    """extract_value_from_file
    Assume attribute in a file is in a format 'attribute=value'.
    """

    file_full_name = filename if folder is None else os.path.join(folder, filename)

    # make sure file exist and read the file
    if not os.path.exists(file_full_name):
        print(f"{Bcolors.FAIL}File {file_full_name} does not exist {Bcolors.ENDC}")
        raise "No such file"
    with open(file_full_name, "r") as f:
        raw = f.read()
        print(f"Reading file {Bcolors.WARNING} {file_full_name} {Bcolors.ENDC} ... done")

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
    print(f"Extracting attribute {Bcolors.OKGREEN} {attribute} {Bcolors.ENDC} in {file_full_name}  ... done")

    return results

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Combine detailed snapshots dumped by valgrind")
    parser.add_argument('--title', metavar='title', type=str, nargs=1,
                        help='Title')
    parser.add_argument('--folder', metavar='folder', type=str, nargs=1,
                        help='Folder where the files are stored. (optional)')
    parser.add_argument('--tags', metavar='tags', type=str, nargs='*',
                        help='A list of tags, e.g., "BeforeFree_rank0.mem_prof" has tag "BeforeFree"')
    parser.add_argument('--attr', metavar='attr', type=str, nargs=1,
                        help='Attribute to extract in valgrind massif dump.')
    parser.add_argument('--mpi_size', metavar='mpi_size', type=int, nargs=1,
                        help='MPI size')
    parser.add_argument('--output_filename', metavar='output_filename', type=str, nargs=1,
                        help='Output to file.')
    args = parser.parse_args()

    # Extract value and write to file
    with open(args.output_filename[0], "a") as f:
        f.write("#### " + args.title[0] + "\n\n")
    for tag in args.tags:
        for r in range(args.mpi_size[0]):
            filename = tag + "_rank{}.mem_prof".format(r)
            if args.folder is not None:
                attr_value = extract_value_from_file(args.folder[0], filename, args.attr[0])
            else:
                attr_value = extract_value_from_file(None, filename, args.attr[0])

            for key in attr_value:
                with open(args.output_filename[0], "a") as f:
                    f.write("**{}({}_rank{})**: ".format(key, tag, r) + str(attr_value[key]) + "\n")
    with open(args.output_filename[0], "a") as f:
        f.write("\n---\n\n")