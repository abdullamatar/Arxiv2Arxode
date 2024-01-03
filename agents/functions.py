import os
import subprocess
import sys
from typing import Tuple


class Functions:
    # see autogen/code_utils.py
    def exec_py(
        self, code: str, infile: str = None, outfile: str = None
    ) -> Tuple[int, str, str]:
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

    def write_file(self, fname: str, content: str):
        with open(fname, "w") as f:
            f.write(content)
