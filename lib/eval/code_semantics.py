import json

# import logging

import matplotlib.pyplot as plt
import numpy as np
from autogen import OpenAIWrapper


def get_code_attempts(file_path):
    with open(file_path, "r") as file:
        data = [json.loads(line) for line in file]
    tds = [t["task_description"] for t in data]
    exe_feedback = [t["exe_feedback"] for t in data]
    codes = [x["code"] for x in exe_feedback]
    return (tds, codes)


if __name__ == "__main__":
    file_paths = [
        "./lib/eval/stats/tasks_and_code_exe_results/combined_stats_a2a.jsonl",
        "./lib/eval/stats/tasks_and_code_exe_results/combined_stats_a2a_turbo.jsonl",
        "./lib/eval/stats/tasks_and_code_exe_results/combined_stats_baseGPT.jsonl",
        "./lib/eval/stats/tasks_and_code_exe_results/combined_stats_base_turbo.jsonl",
    ]

    x = [get_code_attempts(path) for path in file_paths]
    print(x)
