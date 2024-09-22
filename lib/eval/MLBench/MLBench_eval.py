# stdlib
# import sys
import json
import multiprocessing as mp
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn.functional as F
from datasets import load_from_disk  # , Dataset
from transformers import AutoModel, AutoTokenizer  # , AutoModelForCausalLM

# a2a
# import lib.embeddings as emb

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

td = lambda m: m.to(device)  # model to device
inp2d = lambda inp: {k: v.to(device) for k, v in inp.items()}  # input arg 2 device


def load_tasks(file_path: str):
    with open(file_path, "r") as file:
        xs = [json.loads(line) for line in file]

    simplified_structure = []

    for task in xs:
        task_description = task["task_description"]
        # Flatten exe_feedback for each task which is a list of dicts for each attempt that contains keys ["code","exit_code", "logs"]
        codes = [feedback["code"] for feedback in task.get("exe_feedback", None)]
        # len exit_codes == len codes == attempts
        exit_codes = task["exit_codes"]
        task_idx = task["task_idx"]
        simplified_structure.append(
            {
                "task_description": task_description,
                "code_generations": codes,
                "exit_codes": exit_codes,
                "task_idx": task_idx,
            }
        )

    return simplified_structure


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    )
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9
    )


def compute_sentence_embeddings(
    sentences: list[str], tokenizer: AutoTokenizer, model: AutoModel
) -> torch.Tensor:
    """
    Compute sentence embeddings for a list of sentences using a specified tokenizer and model.
    Args:
        sentences (list of str): A list of sentences to be encoded.
        tokenizer (PreTrainedTokenizer): A tokenizer to convert sentences into token IDs.
        model (PreTrainedModel): A pre-trained model to generate embeddings.
    Returns:
        torch.Tensor: A tensor containing the normalized sentence embeddings.
    """
    # model = model.to(device)
    encoded_input = tokenizer(
        sentences, padding=True, truncation=True, return_tensors="pt"
    )  # type: ignore
    encoded_input = inp2d(encoded_input)
    # multiplication with the expanded attention mask that occurs in the mean pooling above discards vital information that perhaps cannot be afforded in the case of programming language.
    with torch.no_grad():
        model_output = model(**encoded_input)  # type: ignore
    sentence_embeddings = model_output.last_hidden_state[:, 0, :]
    # sentence_embeddings = mean_pooling(model_output, encoded_input["attention_mask"])
    sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
    return sentence_embeddings


def compute_cosine_similarity(
    embedding1: torch.Tensor, embedding2: torch.Tensor
) -> float:
    # Ensure embeddings are 2D tensors with a batch dimension
    if embedding1.dim() == 1:
        embedding1 = embedding1.unsqueeze(0)
    if embedding2.dim() == 1:
        embedding2 = embedding2.unsqueeze(0)
    return F.cosine_similarity(embedding1, embedding2).mean().item()


def evaluate_generated_code(
    runs: list[dict], test_set: list[dict], t: AutoTokenizer, m: AutoModel
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Evaluates the generated code against a test set and computes performance metrics.

    Args:
        runs (list): A list of dictionaries containing the generated code and execution results.
        test_set (list): A list of dictionaries containing the test set tasks (ground truths) with their ids, instructions, and outputs.
        t: Model or tokenizer used for computing sentence embeddings.
        m: Model or metric used for computing sentence embeddings.

    Returns:
        tuple: A tuple containing:
            - pd.DataFrame: A DataFrame with detailed attempt-level information including task index, description, attempt number, generated code, exit code, and similarity score.
            - pd.Series: A Series with general performance information including evaluation score and weighted success rate.
    """
    similarities = []
    table_data = []
    weighted_success_sum = 0

    # default dict because the MLBench dataset has duplicated task ids with different instructions and outputs...
    test_set_taskids_output_map = defaultdict(list)
    test_set_tids_instruction_map = defaultdict(list)
    for entry in test_set:
        # map task id to ground truth output
        test_set_taskids_output_map[entry["id"]].append(entry["output"])

        # map task id to task instruction
        test_set_tids_instruction_map[entry["id"]].append(entry["instruction"])

    for run in runs:
        try:
            succesful_execution_count = 0
            # see function load_tasks above for how this part of the dict is constructed, it is a list of code gens taken from exe_feedback of 'raw' data.
            generated_codes = run["code_generations"]

            # Index via the task ids present in the tested runs, maybe some tasks in the dataset failed to be present in the tested runs
            task_idx = run[
                "task_idx"
            ]  # Used to match task_idx from the generated code to the "id" field in the test set

            try:
                task_desc = test_set_tids_instruction_map.get(int(task_idx)).pop(0)

                # gt is now (potentially) a list, after processing the first instance of an id pop the first element in the list
                ground_truth_list = test_set_taskids_output_map.get(int(task_idx))

                ground_truth = ground_truth_list.pop(
                    0
                )  # get first element (potential first instance of repeated task["id"]), which should correspond to the first instance of the given candidate task_idx as it appeared in the candidate outputs. However, given the updated  multiprocessed nature of the code (for claude attempts and in distribution python), this may now not be the case. i,e,. the first instance of the task id from the test set may be matched against any other instance of the task id in the candidate outputs, depending on what order it was executed... This is only an issue for dup'd ids and there are only a few.
            except Exception as e:
                raise ValueError(
                    f"Task {task_idx} not found in test set or data is malformed: {e}"
                )

            assert (
                ground_truth
            ), f"Task {task_idx} not found in test set, ensure the correct subset is being used."

            # There is a single GroundTruth output per task instruction
            ground_truth_embedding = compute_sentence_embeddings(ground_truth, t, m)

            # each task will have multiple candidate generations and all those should be compared to the gt output
            for i, generated_code in enumerate(generated_codes):
                try:
                    candidate_embedding = compute_sentence_embeddings(
                        generated_code, t, m
                    )
                    similarity = compute_cosine_similarity(
                        candidate_embedding, ground_truth_embedding
                    )  # Due to norm of embeddings, this is equivalent to std matmul -> X_hat @ X.T
                except Exception as e:
                    raise RuntimeError(
                        f"Error computing embeddings or similarity for task {task_idx}, attempt {i + 1}: {e}"
                    )

                succesful_execution_count += (
                    1 if (exit_code := run["exit_codes"][i]) == 0 else 0
                )

                table_data.append(
                    {
                        "task_idx": int(task_idx),
                        "task_description": task_desc,
                        "attempt": i + 1,
                        "generated_code": generated_code,
                        "exit_code": abs(exit_code),
                        "similarity": similarity,
                    }
                )
                similarities.append(similarity)
                weighted_success_sum += succesful_execution_count / (
                    len(generated_codes) if generated_codes else 1
                )

        except Exception as e:
            print(f"Error processing task {task_idx}: {e}")
            raise

    weighted_success_rate = weighted_success_sum / len(table_data)
    evaluation = (
        0.7 * np.mean(similarities) + weighted_success_rate
    )  # total performance under entire dataset

    return pd.DataFrame(table_data), pd.Series(
        {
            "valuation": evaluation,
            "weighted_success_rate": weighted_success_rate,
        }
    )


def plot_progress(similarity_progress, pass_rate_progress):
    plt.figure(figsize=(10, 6))
    plt.plot(similarity_progress, label="Cosine Similarity Progress", color="blue")
    plt.plot(pass_rate_progress, label="Pass Rate Progress", color="green")
    plt.xlabel("Attempt Number")
    plt.ylabel("Value")
    plt.title("Code Evaluation Progress Over Attempts")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_exit_code_stacked_bars(exit_code_progress):
    success, failure = zip(
        *[
            (
                # count succesful exes...
                x := sum(1 for code in codes if code == 0),
                # that leaves any attempts with a nonzero exit code as the failures
                len(codes) - x,
            )
            for codes in exit_code_progress  # list of attempt exit codes per task i,e [[1,0,0,1],[0,0]...]
            ##############################                                             task1^   task2^ ...
        ]
    )
    plt.figure(figsize=(10, 6))
    attempts = range(1, len(success) + 1)
    plt.bar(attempts, success, label="Success (exit code 0)", color="green")
    plt.bar(
        attempts,
        failure,
        bottom=success,
        label="Failure (non-zero exit code)",
        color="red",
    )

    plt.title("Success vs Failure of Exit Codes per Attempt")
    plt.xlabel("Attempt Number")
    plt.ylabel("Number of Attempts")
    plt.legend()
    plt.grid(True)
    plt.show()


def g(runs, subset):
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    model = td(AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2"))

    mlbench = load_from_disk(os.path.join(root_dir, "MLBench/datasets"))
    # mlbench = load_from_disk("./datasets")
    # root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    runs = load_tasks(runs)
    tested_runs = mlbench[subset]

    evaluation_results = evaluate_generated_code(runs, tested_runs, tokenizer, model)

    # plot_progress(
    #     evaluation_results["similarity_progress"],
    #     evaluation_results["pass_rate_progress"],
    # )
    # plot_exit_code_distribution(evaluation_results["exit_code_progress"])

    return evaluation_results
    # return value, similarities


def process_file(args):
    file_path, subset = args
    return g(file_path, subset)


if __name__ == "__main__":
    # Cuda mp requires spawn and not fork
    # NOTE: This code does not function correctly, it is under development.
    mp.set_start_method("spawn", force=True)

    input_files = [
        ("./mlfull.jsonl", "full"),
        ("./mlquart7-gc.jsonl", "quarter"),
        ("./lib/eval/baseGPT4full.jsonl", "full"),
        ("./lib/eval/baseGPT4quart.jsonl", "quarter"),
    ]

    with mp.Pool(processes=len(input_files)) as pool:
        results = pool.map(process_file, input_files)

    (
        (a2a_attempts_full, a2a_sumstats_full),
        (a2a_attempts_quart, a2a_sumstats_quart),
        (basegpt_attempts_full, basegpt_sumstats_full),
        (basegpt_attempts_quart, basegpt_sumstats_quart),
    ) = results

    print("Results processed successfully.")
