import os

def extract_value_from_file(filename: str, attribute: str) -> list:
    """extract_value_from_file
    Assume attribute in a file is in a format 'attribute=value'.
    """

    # make sure file exist and read the file
    if not os.path.exists(filename):
        raise "No such file"
    with open(filename, "r") as f:
        raw = f.read()

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

    return extract_value

if __name__ == "__main__":

    values = extract_value_from_file("AfterFree_rank0.mem_prof", "mem_heap_B")

    print(values, "<br>", values)