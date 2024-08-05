# stdlib
# import sys
import json
import os

# pytorch
import torch
import torch.nn.functional as F
from datasets import load_from_disk  # , Dataset
# hf
from transformers import AutoModel, AutoTokenizer  # , AutoModelForCausalLM

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

        simplified_structure.append(
            {"task_description": task_description, "code_generations": codes}
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


# Assuming `runs` contains the generated code and `tested_runs['output']` contains the ground truth
def evaluate_generated_code(runs, tested_runs_output, t, m):
    similarities = []

    for i, run in enumerate(runs):
        generated_codes = run["code_generations"]
        ground_truth = tested_runs_output[i]

        for generated_code in generated_codes:
            generated_embedding = compute_sentence_embeddings([generated_code], t, m)[0]
            ground_truth_embedding = compute_sentence_embeddings([ground_truth], t, m)[
                0
            ]

            similarity = compute_cosine_similarity(
                generated_embedding, ground_truth_embedding
            )
            similarities.append(similarity)

    average_similarity = sum(similarities) / len(similarities)
    return average_similarity, similarities


def g():
    print(os.getcwd())
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

    # print(root_dir)
    mlbench = load_from_disk("./datasets")
    runs = load_tasks("../../../agent_runs.jsonl")

    tested_runs = mlbench["quarter"]

    average_similarity, similarities = evaluate_generated_code(
        runs, tested_runs["output"], tokenizer, model
    )
    print(f"Average similarity: {average_similarity}, Similarities: {similarities}")
    return average_similarity, similarities
