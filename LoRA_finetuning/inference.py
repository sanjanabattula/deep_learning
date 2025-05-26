# Importing required libraries
import os, math, torch
import numpy as np
import pandas as pd
import pickle

from datasets import concatenate_datasets, load_dataset, Dataset
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, TrainerCallback,
    DataCollatorWithPadding
)
from torch.nn import functional as F
from sklearn.metrics import accuracy_score
from peft import get_peft_model, AdaLoraConfig

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load tokenizer & collator from best_model/
tokenizer = AutoTokenizer.from_pretrained("best_model")
collator  = DataCollatorWithPadding(tokenizer, return_tensors="pt")

# Load unlabelled test
with open("/kaggle/input/deep-learning-spring-2025-project-2/test_unlabelled.pkl","rb") as f:
    test_data = pickle.load(f)
test_ds = Dataset.from_dict({"text": test_data["text"]})

# Tokenize
def tokenize_fn(batch):
    return tokenizer(batch["text"], truncation=True, padding=True, max_length=256)

test_tok = test_ds.map(
    tokenize_fn,
    batched=True,
    remove_columns=["text"]
)

# Create DataLoader
test_loader = DataLoader(
    test_tok,
    batch_size=64,
    shuffle=False,
    collate_fn=collator
)

# Load AdaLoRA model from best_model/
# This merges the adapters into the base model.
base_model = AutoModelForSequenceClassification.from_pretrained(
    "roberta-base", num_labels=4
)
model = PeftModel.from_pretrained(base_model, "best_model") \
                 .merge_and_unload() \
                 .to(device)
model.eval()

# Run inference
all_ids, all_preds = [], []
bs = test_loader.batch_size

for i, batch in enumerate(test_loader):
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        logits = model(**batch).logits
    preds = logits.argmax(dim=-1).cpu().tolist()

    start = i * bs
    all_ids.extend(range(start, start + len(preds)))
    all_preds.extend(preds)

# Save submission
submission = pd.DataFrame({"ID": all_ids, "label": all_preds})
submission.to_csv("best_model-submission.csv", index=False)
print("Saved submission.csv")
