import logging
import os
import subprocess
import sys
from typing import Tuple

from autogen.code_utils import extract_code

logging.basicConfig(
    level=logging.INFO,
    filename="logs/func.log",
    format="%(asctime)s  👁‍🗨  %(levelname)s  👁‍🗨  :\n%(message)s",
)
logger = logging.getLogger("functions")


# see autogen/code_utils.py
def exec_py(code: str, infile: str = None, outfile: str = None) -> Tuple[int, str, str]:
    """Execute generated python code or a python file"""
    try:
        if infile:
            cmd = [sys.executable, infile]
        else:
            cmd = [sys.executable, "-c", code]

        stdout = subprocess.PIPE if not outfile else open(outfile, "w")
        result = subprocess.run(
            cmd,
            stdout=stdout,
            stderr=subprocess.PIPE,
            text=True,
        )

        if outfile:
            stdout.close()
        if result.stderr:
            print(f"An error occurred while executing the code: {result.stderr}")
        return result.returncode, result.stdout, result.stderr
    except Exception as e:
        print(f"An error occurred while executing the code: {e}")


def write_file(fname: str, content: str):
    with open(fname, "w") as f:
        f.write(content)


def get_code_blocks(recipient, messages, sender, config) -> Tuple[bool, object]:
    # cb = []
    i = 0
    logger.info(f"Getting code blocks from {sender} to {recipient}")
    for msg in reversed(messages):
        if not msg["content"]:
            continue
        code_block = extract_code(msg["content"])
        for l, code in code_block:
            if l == "python":
                # cb.append((l, code))
                write_file(f"./sandbox/temp{i}.py", code)
                i += 1
    return False, None
