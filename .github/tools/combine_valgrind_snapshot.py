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

def combine_detailed_snapshot(filebase: str, combined_filename: str, mpi_rank: int, iter_total: int):
    """combined_detailed_snapshot
    Combined individual detailed snapshots dumped by valgrind,
    so that it can be visualized in massif-visualizer in a time series.
    The time series starts at t = 0.
    """

    snapshot_flag = """#-----------\nsnapshot={}\n#-----------\n"""
    combined_filename_suffix = ".mem_prof"

    # make sure the combined_filename ends with .mem_prof
    if not combined_filename.endswith(combined_filename_suffix):
        combined_filename += combined_filename_suffix

    # make sure the file does not exist
    if os.path.exists(combined_filename):
        print(f"{Bcolors.FAIL}File {combined_filename} already exists {Bcolors.ENDC}")
        raise "File already exists"

    # write the combined file
    with open(combined_filename, "a") as f_combined:

        for t in range(iter_total):

            # read raw file
            filename = filebase.format(mpi_rank, t)
            with open(filename, "r") as f:
                raw = f.read()
                print(f"Reading file {Bcolors.WARNING} {filename} {Bcolors.ENDC} ... done")

            # ignore the heading in the file except the first one,
            # and we also need to append our own flag
            if t != 0:
                # write the snapshot flag
                f_combined.write(snapshot_flag.format(t))

                # ignore the heading and the snapshot flag
                pos = raw.find(snapshot_flag.format(0))
                pos = pos + len(snapshot_flag.format(0))
                f_combined.write(raw[pos:-1])
                f_combined.write("\n")

            else:
                f_combined.write(raw)

    print(f"Writing file to {Bcolors.OKGREEN} {combined_filename} {Bcolors.ENDC} ... done")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Combine detailed snapshots dumped by valgrind")
    parser.add_argument('--tag', metavar='tag', type=str, nargs=1,
                        help='Tag, e.g., "BeforeFree_rank0_time0.mem_prof" has tag "BeforeFree"')
    parser.add_argument('--mpi_size', metavar='mpi_size', type=int, nargs=1,
                        help='MPI size')
    parser.add_argument('--total_time_steps', metavar='time_steps', type=int, nargs=1,
                        help='Total number of time steps')
    args = parser.parse_args()

    # Call combine_detailed_snapshot
    file_base_name = args.tag[0] + "_rank{}_time{}.mem_prof"
    combined_file_name = args.tag[0] + "_rank{}.mem_prof"

    for r in range(args.mpi_size[0]):
        combine_detailed_snapshot(file_base_name, combined_file_name.format(r), r, args.total_time_steps[0])