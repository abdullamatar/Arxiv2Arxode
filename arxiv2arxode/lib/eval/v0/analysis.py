import json

import matplotlib.pyplot as plt
import numpy as np
# from autogen import OpenAIWrapper
from autogen.code_utils import execute_code, extract_code

# from agents.agent_conf import base_cfg
# from lib.embeddings import get_db_connection

# client = OpenAIWrapper(config_list=base_cfg["config_list"])
# logger = logging.getLogger("analysis")


def load_tasks(file_path: str):
    """
    Load tasks from a JSON Lines (.jsonl) file.
    """
    with open(file_path, "r") as file:
        x = [json.loads(line) for line in file]

    tasks = [task for task in x]
    # vec_dbs = [task["vec_db"] for task in x]

    return tasks


def calculate_success_rates_from_combined_stats(combined_stats):
    """
    Calculate success rates over attempts
    """
    success_rates = {}
    for i, stats in enumerate(combined_stats):
        # task_desc = f"Task_{i}"
        # print(stats)
        # exit()
        # total_attempts = len(stats["exe_feedback"])
        # print(stats["exit_codes"])
        # print(f"stats['exit_codes']==0 : {stats['exit_codes'] == 0}")

        # print(stats)
        success_rate = (
            sum(attempt == 0 for attempt in stats["exit_codes"])
            / len(stats["exit_codes"])
            * 100
            if stats["exit_codes"]
            else 0
        )
        # success_rate = (successes / total_attempts) * 100 if total_attempts else 0
        stats["passrate"] = success_rate
    return success_rates


def load_success_rates(file_paths):
    success_rates = []
    for file_path in file_paths:
        with open(file_path, "r") as file:
            combined_stats = [json.loads(line) for line in file]
            rates = calculate_success_rates_from_combined_stats(combined_stats)
            success_rates.append(list(rates.values()))
    return success_rates


def calculate_average_success_rate(success_rates):
    total_success_rate = sum(success_rates.values())
    average_success_rate = (
        total_success_rate / len(success_rates) if success_rates else 1
    )
    return average_success_rate


def plot_grouped_success_rates(
    file_paths, labels, title="Grouped Success Rates by Task", save_figurename=None
):
    success_rates = load_success_rates(file_paths)
    n_groups = len(success_rates[0])

    average_success_rates = [
        calculate_average_success_rate(
            dict(zip([f"Task_{i+1}" for i in range(n_groups)], rates))
        )
        for rates in success_rates
    ]

    fig, ax = plt.subplots(figsize=(12, 8))
    index = np.arange(n_groups + 1)
    bar_width = 0.2
    opacity = 0.8

    colors = ["#001a3b", "#3266a8", "#ff1500", "#fc9b92"]
    for i, (rates, label) in enumerate(zip(success_rates, labels)):
        avg_rate = average_success_rates[i]
        avg_label = f"{label} ({avg_rate:.2f}% Success)"
        plt.bar(
            index[:-1] + i * bar_width,
            rates,
            bar_width,
            alpha=opacity,
            label=avg_label,
            color=colors[i],
        )

    for i, (avg_rate, color) in enumerate(zip(average_success_rates, colors)):
        plt.bar(
            index[-1] + i * bar_width,
            avg_rate,
            bar_width,
            alpha=opacity,
            # label=("Average" if i == 0 else "_nolegend_"),
            color=color,
        )

    plt.xlabel("Task")
    plt.ylabel("Success Rate (%)")
    plt.title(title)
    plt.xticks(
        index + bar_width, [f"Task {i+1}" for i in range(n_groups)] + ["Average"]
    )

    plt.legend(loc="upper left", bbox_to_anchor=(1, 1))

    plt.tight_layout()

    if save_figurename:
        plt.savefig(save_figurename)
    plt.show()


if __name__ == "__main__":
    # file_path = "combined_stats_baseGPT.jsonl"
    # tasks, vec_dbs = load_tasks("./temp.jsonl")

    # execution_results = [
    #     execute_with_gpt(task, vec_db) for task, vec_db in zip(tasks, vec_dbs)
    # ]

    file_paths = [
        "./mlquart7-gc.jsonl",
        # "./lib/eval/stats/tasks_and_code_exe_results/combined_stats_a2a_turbo.jsonl",
        # "./lib/eval/stats/tasks_and_code_exe_results/combined_stats_baseGPT.jsonl",
        # "./lib/eval/stats/tasks_and_code_exe_results/combined_stats_base_turbo.jsonl",
    ]

    labels = ["A2A GPT-4"]  # , "A2A GPT-3.5", "Base GPT-4", "Base GPT-3.5"]
    plot_grouped_success_rates(
        file_paths=file_paths,
        labels=labels,
        save_figurename="grouped_success_rates.png",
    )

    # plot_success_rates_from_combined_stats(
    #     "./lib/eval/stats/combined_stats_a2a.jsonl",
    #     "./lib/eval/stats/figs/success_rates_a2a.png",
    # )

    # plot_success_rates_from_combined_stats(
    #     "./lib/eval/stats/combined_stats_a2a_turbo.jsonl",
    #     "./lib/eval/stats/figs/success_a2a_turbo.png",
    # )

    # plot_success_rates_from_combined_stats(
    #     "./lib/eval/stats/combined_stats_baseGPT.jsonl",
    #     "./lib/eval/stats/figs/success_rates_baseGPT.png",
    # )

    # plot_success_rates_from_combined_stats(
    #     "./lib/eval/stats/combined_stats_base_turbo.jsonl",
    #     "./lib/eval/stats/figs/succ_rates_base_turbo.png",
    # )

# AL 7amdulillah
