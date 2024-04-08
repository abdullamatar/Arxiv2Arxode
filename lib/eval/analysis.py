import json


import matplotlib.pyplot as plt
import numpy as np
from autogen import OpenAIWrapper
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

    tasks = [task["task_description"] for task in x]
    vec_dbs = [task["vec_db"] for task in x]

    return (tasks, vec_dbs)


def execute_with_gpt(task_desc: str, vec_db):
    print(f"vec_db: {vec_db}")
    dbconn = get_db_connection(vec_db)
    vec_db_query = client.create(
        messages=[
            {
                "role": "user",
                "content": f"Create a query for the following task:\n{task_desc}\nThe query will be used to search a vector database with relevant research papers in it, keep the query short, useful, and relevant. DO NOT RESPOND WITH ANYTHING BUT A QUERY",
            }
        ],
        model="gpt-3.5-turbo-0125",
        cache_seed=33,
    )
    vec_db_query = vec_db_query.choices[0].message.content
    assert vec_db_query, "No query generated for vector database search."

    retrieved_excerpts = dbconn.similarity_search(vec_db_query)
    context = [d.page_content for d in retrieved_excerpts]

    formatted_context = " ".join(
        ["Context {}: {}".format(i + 1, doc) for i, doc in enumerate(context)]
    )

    # exit_codes = []
    # execution_logs_list = []
    exe_feedback = []
    # success = 0
    attempts = 0
    while attempts < 5:
        messages = [
            {
                "role": "system",
                "content": f"You are tasked with generating code based off of information from research papers, think through the process in a step by step manner and always generate complete working code. Do not fill in functions with pass or ellipses, DO NOT make any assumptions without justification. Here is the code generated from your previous attempt:\n{exe_feedback[-1]['code'] if exe_feedback else 'This is your first attempt.'}\n Here is the previous execution log:\n{exe_feedback[-1]['logs'] if exe_feedback else 'This is your first attempt.'}",
            },
            {
                "role": "user",
                "content": task_desc,
            },
            {
                "role": "assistant",
                "content": f"Here is retrieved information related to the task at hand:\n{formatted_context}",
            },
        ]
        response = client.create(messages=messages, cache_seed=33)
        generated_content = response.choices[0].message.content
        assert generated_content
        extracted_codes = extract_code(generated_content)
        with open("generated_code.txt", "a") as f:
            f.write(
                f"generated_content for summarization task attempt #{attempts+1}:\n{generated_content}\n"
            )

        if extracted_codes:
            extracted_code = extracted_codes[0][1]
            exit_code, execution_logs, _ = execute_code(
                extracted_code, use_docker=False
            )
            exe_feedback.append(
                {
                    "code": extracted_code,
                    "exit_code": exit_code if execution_logs else 1,
                    "logs": execution_logs,
                }
            )
        # logger.info(f"Attempt {attempts + 1} completed.")
        attempts += 1

    print(f"client.total_usage_summary() = {client.total_usage_summary}")

    exit_codes = [feedback["exit_code"] for feedback in exe_feedback]
    combined_stats = {
        "task_description": task_desc,
        "usage_stats": {
            "total_cost": response.cost,
            "gpt-3.5-turbo": {
                "cost": response.cost,
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            },
        },
        "exe_feedback": exe_feedback,
    }
    combined_stats["exit_codes"] = exit_codes

    # usage_stats = {
    #     "task_description": task_desc,
    #     "attempted_generations": attempts,
    #     "successful_generation": success,
    #     "cost": response.cost,
    #     "completion_tokens": response.usage.completion_tokens,
    #     "prompt_tokens": response.usage.prompt_tokens,
    #     "total_tokens": response.usage.total_tokens,
    #     "generated_content": (
    #         generated_content if success else "Failed to generate successful code."
    #     ),
    #     "extracted_code": extracted_code if success else "",
    #     "exit_codes": exit_codes,
    #     "execution_logs": execution_logs_list,
    # }
    with open("combined_stats_base_turbo.jsonl", "a") as f:
        f.write(json.dumps(combined_stats) + "\n")
    return combined_stats


# def calculate_success_rates(execution_results):
#     """
#     Calculate and return success rates for all executed tasks.
#     """
#     success_rates = {}
#     for i, result in enumerate(execution_results, 1):
#         exit_codes = result["exit_code"]
#         task_desc = f"Task_{i}"
#         success_rate = (
#             (exit_codes.count(0) / len(exit_codes)) * 100 if exit_codes else 0
#         )
#         success_rates[task_desc] = success_rate
#     return success_rates


# def plot_results(exe_results):
#     """
#     Plot the calculated success rates.
#     """
#     tasks = list(exe_results.keys())
#     rates = list(exe_results.values())

#     plt.figure(figsize=(10, 6))
#     y_pos = np.arange(len(tasks))
#     plt.barh(y_pos, rates, color="skyblue", edgecolor="black")
#     plt.yticks(y_pos, tasks)
#     plt.xlabel("Success Rate (%)")
#     plt.title("Task Success Rates")
#     plt.xlim(0, 100)

#     for index, value in enumerate(rates):
#         plt.text(value, index, f"{value:.2f}%")

#     plt.tight_layout()
#     plt.show()
#     plt.savefig("success_rates_base_gpt.png", format="png")


def calculate_success_rates_from_combined_stats(combined_stats):
    """
    Calculate success rates over attempts
    """
    success_rates = {}
    for i, stats in enumerate(combined_stats, 1):
        task_desc = f"Task_{i}"
        # print(stats)
        # exit()
        # total_attempts = len(stats["exe_feedback"])
        # print(stats["exit_codes"])
        # print(f"stats['exit_codes']==0 : {stats['exit_codes'] == 0}")

        success_rate = (
            sum(attempt == 0 for attempt in stats["exit_codes"])
            / len(stats["exit_codes"])
            * 100
        )
        # success_rate = (successes / total_attempts) * 100 if total_attempts else 0
        success_rates[task_desc] = success_rate
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
        total_success_rate / len(success_rates) if success_rates else 0
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
        "./lib/eval/stats/tasks_and_code_exe_results/combined_stats_a2a.jsonl",
        "./lib/eval/stats/tasks_and_code_exe_results/combined_stats_a2a_turbo.jsonl",
        "./lib/eval/stats/tasks_and_code_exe_results/combined_stats_baseGPT.jsonl",
        "./lib/eval/stats/tasks_and_code_exe_results/combined_stats_base_turbo.jsonl",
    ]

    labels = ["A2A GPT-4", "A2A GPT-3.5", "Base GPT-4", "Base GPT-3.5"]
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
