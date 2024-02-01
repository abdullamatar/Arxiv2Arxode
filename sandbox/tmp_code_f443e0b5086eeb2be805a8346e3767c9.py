import numpy as np
from datasets import load_dataset
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          Trainer, TrainingArguments)

# Load model and tokenizer
model_name = "distilbert-base-uncased-finetuned-sst-2-english"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load a dataset (using the SST-2 validation split for simplicity)
dataset = load_dataset(
    "glue", "sst2", split="validation[:10%]"
)  # Only taking 10% for quicker evaluation


# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples["sentence"], padding="max_length", truncation=True)


tokenized_dataset = dataset.map(tokenize_function, batched=True)


# Define compute_metrics function to calculate accuracy
def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=-1)
    accuracy = np.mean(preds == labels)
    return {"accuracy": accuracy}


# Define Trainer arguments
training_args = TrainingArguments(
    output_dir="./results",
    per_device_eval_batch_size=64,
    use_cpu=True,
    report_to="none",
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    compute_metrics=compute_metrics,
    tokenizer=tokenizer,
    eval_dataset=tokenized_dataset,
)

# Evaluate the model
eval_results = trainer.evaluate()

print(f"Evaluation Accuracy: {eval_results['eval_accuracy']}")
