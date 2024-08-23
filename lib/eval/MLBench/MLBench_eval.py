# stdlib
# import sys
import json

# pytorch
import torch
import torch.nn.functional as F
from datasets import load_from_disk  # , Dataset
# hf
from transformers import AutoModel, AutoTokenizer  # , AutoModelForCausalLM

# import os



# from huggingface_hub import login


# a2a
# import lib.embeddings as emb

# root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_tasks(file_path: str):
    with open(file_path, "r") as file:
        x = [json.loads(line) for line in file]

    simplified_structure = []

    for task in x:
        task_description = task["task_description"]
        codes = [feedback["code"] for feedback in task.get("exe_feedback", [])]
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


def compute_sentence_embeddings(sentences, tokenizer, model):
    encoded_input = tokenizer(
        sentences, padding=True, truncation=True, return_tensors="pt"
    )
    with torch.no_grad():
        model_output = model(**encoded_input)
    sentence_embeddings = mean_pooling(model_output, encoded_input["attention_mask"])
    sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
    return sentence_embeddings


def compute_cosine_similarity(embedding1, embedding2):
    # Ensure embeddings are 2D tensors with a batch dimension
    if embedding1.dim() == 1:
        embedding1 = embedding1.unsqueeze(0)
    if embedding2.dim() == 1:
        embedding2 = embedding2.unsqueeze(0)
    return F.cosine_similarity(embedding1, embedding2).mean().item()


def evaluate_generated_code(runs, tested_runs_output, t, m):
    similarities = []

    test_set_taskids = {entry["id"]: entry["output"] for entry in tested_runs_output}
    succesful_execution_count = 0
    for run in runs:
        generated_codes = run["code_generations"]

        ground_truth = test_set_taskids.get(int(run["task_idx"]))
        if not ground_truth:
            continue
        assert ground_truth, f"Task {run['task_idx']} not found in test set"

        if any(exit_code == 0 for exit_code in run["exit_codes"]):
            succesful_execution_count += 1

        for generated_code in generated_codes:
            generated_embedding = compute_sentence_embeddings(generated_code, t, m)[0]
            ground_truth_embedding = compute_sentence_embeddings(ground_truth, t, m)[0]

            similarity = compute_cosine_similarity(
                generated_embedding, ground_truth_embedding
            )
            similarities.append(similarity)

    pass_rate = succesful_execution_count / len(runs)

    average_similarity = sum(similarities) / len(similarities)
    evaluation = 0.5 * average_similarity + 0.5 * pass_rate

    return {
        "evaluation": evaluation,
        "average_similarity": average_similarity,
        "pass_rate": pass_rate,
    }


def g(dset, subset):
    # print(os.getcwd())
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

    # print(root_dir)
    # mlbench = load_from_disk("./lib/eval/MLBench/datasets")
    mlbench = load_from_disk("./datasets")
    # root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # print(f"Root dir: {root_dir}")

    runs = load_tasks(dset)

    tested_runs = mlbench[subset]

    # print(f"Average similarity: {average_similarity}, Similarities: {similarities}")
    return evaluate_generated_code(runs, tested_runs, tokenizer, model)
    # return value, similarities


if __name__ == "__main__":
    g("😎", "😶‍🌫️")
