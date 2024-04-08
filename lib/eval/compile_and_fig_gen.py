import json
import matplotlib.pyplot as plt

# import numpy as np


def read_metrics(file_path):
    with open(file_path, "r") as file:
        data = json.load(file)
    return data


def aggregate_metrics(data_list):
    cc_metrics_summary = {
        "avg_cc_over_functions_within_attempt": [],
        "num_functions": [],
    }
    halstead_metrics_keys_of_interest = [
        "vocabulary",
        "length",
        "calculated_length",
        "volume",
        "difficulty",
        "effort",
        "time",
        "bugs",
    ]
    halstead_metrics_summary = {key: [] for key in halstead_metrics_keys_of_interest}

    for entry in data_list:
        cc_metrics = entry.get("cc_metrics")
        if cc_metrics:
            avg_cc = cc_metrics.get("avg_cc_over_functions_within_attempt")
            if isinstance(avg_cc, (float, int)):
                cc_metrics_summary["avg_cc_over_functions_within_attempt"].append(
                    avg_cc
                )
            else:
                cc_metrics_summary["avg_cc_over_functions_within_attempt"].append(0)

            num_functions = cc_metrics.get("num_functions", 0)
            cc_metrics_summary["num_functions"].append(num_functions)
        else:
            cc_metrics_summary["avg_cc_over_functions_within_attempt"].append(0)
            cc_metrics_summary["num_functions"].append(0)

        halstead_metrics = entry.get("halstead_metrics", {})
        for key in halstead_metrics_summary.keys():
            halstead_metrics_summary[key].append(halstead_metrics.get(key, 0))

    return cc_metrics_summary, halstead_metrics_summary


def plot_combined_metrics(cc_metrics_summary, halstead_metrics_summary):
    plt.figure(figsize=(14, 7))

    # Subplot 1: Distribution of Average Cyclomatic Complexity
    plt.subplot(1, 2, 1)  # 1 row, 2 columns, 1st subplot
    plt.hist(
        cc_metrics_summary["avg_cc_over_functions_within_attempt"],
        bins=20,
        color="skyblue",
        alpha=0.7,
    )
    plt.xlabel("Average Cyclomatic Complexity")
    plt.ylabel("Frequency")
    plt.title("Average Cyclomatic Complexity Distribution")
    plt.grid(True)

    # Subplot 2: Distribution of Halstead Volume
    plt.subplot(1, 2, 2)  # 1 row, 2 columns, 2nd subplot
    plt.hist(halstead_metrics_summary["volume"], bins=20, color="lightgreen", alpha=0.7)
    plt.xlabel("Halstead Volume")
    plt.ylabel("Frequency")
    plt.title("Halstead Volume Distribution")
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(7, 7))
    plt.scatter(
        cc_metrics_summary["avg_cc_over_functions_within_attempt"],
        halstead_metrics_summary["effort"],
        alpha=0.7,
        c="orange",
    )
    plt.xlabel("Average Cyclomatic Complexity")
    plt.ylabel("Halstead Effort")
    plt.title("Cyclomatic Complexity vs. Halstead Effort")
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    file_paths = [
        "/home/sonofman/Research/Arxiv2Arxode/lib/eval/stats/code_complexity/cc_a2a_gpt4.json",
        "/home/sonofman/Research/Arxiv2Arxode/lib/eval/stats/code_complexity/cc_a2a_turbo.json",
        "/home/sonofman/Research/Arxiv2Arxode/lib/eval/stats/code_complexity/cc_baseGPT.json",
        "/home/sonofman/Research/Arxiv2Arxode/lib/eval/stats/code_complexity/cc_base_turbo.json",
    ]
    all_data = []
    for path in file_paths:
        data = read_metrics(path)
        all_data.extend(data)

    cc_metrics_summary, halstead_metrics_summary = aggregate_metrics(all_data)
    plot_combined_metrics(cc_metrics_summary, halstead_metrics_summary)
