def ftos(fd):
    with open(fd, "r") as f:
        return f.read()


def write_file(fname: str, content: str):
    with open(fname, "w") as f:
        f.write(content)
