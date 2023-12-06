import os
import shutil
import subprocess


def clone_and_clean_repo(git_url, target_dir):
    """
    Clones a Git repository from the given URL, then removes all non-Python files.
    -----------
    git_url: The URL of the Git repository to clone.
    target_dir: The directory to clone the repository into.
    """

    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    subprocess.run(["git", "clone", git_url, target_dir], check=True)

    is_python_file = lambda filename: filename.endswith(".py")

    for root, dirs, files in os.walk(target_dir, topdown=False):
        for file in files:
            if not is_python_file(file):
                os.remove(os.path.join(root, file))
        for dir in dirs:
            dir_path = os.path.join(root, dir)
            if not os.listdir(dir_path):
                shutil.rmtree(dir_path)


# https://github.com/openai/openai-cookbook/blob/main/examples/Code_search_using_embeddings.ipynb
def get_function_name(code):
    """
    Extract function name from a line beginning with 'def' or 'async def'.
    """
    for prefix in ["def ", "async def "]:
        if code.startswith(prefix):
            return code[len(prefix) : code.index("(")]


def get_until_no_space(all_lines, i):
    """
    Get all lines until a line outside the function definition is found.
    """
    ret = [all_lines[i]]
    for j in range(i + 1, len(all_lines)):
        if len(all_lines[j]) == 0 or all_lines[j][0] in [" ", "\t", ")"]:
            ret.append(all_lines[j])
        else:
            break
    return "\n".join(ret)


def get_functions(filepath):
    """
    Get all functions in a Python file.
    """
    with open(filepath, "r") as file:
        all_lines = file.read().replace("\r", "\n").split("\n")
        for i, l in enumerate(all_lines):
            for prefix in ["def ", "async def "]:
                if l.startswith(prefix):
                    code = get_until_no_space(all_lines, i)
                    function_name = get_function_name(code)
                    yield {
                        "code": code,
                        "function_name": function_name,
                        "filepath": filepath,
                    }
                    break


def extract_functions_from_repo(code_root):
    """
    Extract all .py functions from the repository.
    """
    code_files = list(code_root.glob("**/*.py"))

    num_files = len(code_files)
    print(f"Total number of .py files: {num_files}")

    assert num_files > 0, "No .py files found in the repository."

    all_funcs = [
        func for code_file in code_files for func in get_functions(str(code_file))
    ]

    num_funcs = len(all_funcs)
    print(f"Total number of functions extracted: {num_funcs}")

    return all_funcs
