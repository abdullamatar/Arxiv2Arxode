import logging
import os
import subprocess
import sys
from typing import Tuple
from uuid import uuid4

from autogen.code_utils import execute_code, extract_code

from utils.misc import ftos, write_file

############NOTE##################
# THIS FILE IS NOT USED ANYWHERE #
############NOTE##################

if not os.path.exists("./logs"):
    os.makedirs("./logs")

funchand = logging.FileHandler("./logs/functions.log")
logger = logging.getLogger("functions")
logger.addHandler(funchand)
logger.info("functions.py loaded")


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


# Params def'd according to requirements for autogen agent reply function
def get_code_blocks(recipient, messages, sender, config) -> Tuple[bool, object]:
    # cb = []
    i = 0
    logger.info(f"Getting code blocks from {sender} to {recipient}")
    logger.info(f"config: {config}")
    for msg in reversed(messages):
        if not msg["content"]:
            continue
        code_block = extract_code(msg["content"])
        for l, code in code_block:
            if l == "python":
                # cb.append((l, code))
                fd = f"./sandbox/temp-{uuid4()}.py"
                write_file(fd, code)
                i += 1
    return False, None
