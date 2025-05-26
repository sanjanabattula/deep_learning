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

# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_MODEL    = "roberta-base"
TEACHER_MODEL = None  # no distillation here

# Load & tokenize AG News
tok = AutoTokenizer.from_pretrained(BASE_MODEL)
raw = load_dataset("ag_news")
def prep(batch):
    return tok(batch["text"], truncation=True, padding=False, max_length=256)

toked = raw.map(prep, batched=True, remove_columns=["text"])
toked = toked.rename_column("label", "labels")
toked["train"] = toked["train"].map(lambda x: {"labels": int(x["labels"])})
toked["test"]  = toked["test"].map(lambda x: {"labels": int(x["labels"])})

split = toked["train"].train_test_split(test_size=640, seed=42, stratify_by_column="labels")
train_ds, val_ds = split["train"], split["test"]
collator = DataCollatorWithPadding(tok)

# Build model with AdaLoRA
student = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL, num_labels=4)

# compute total steps
NUM_EPOCHS = 3
BATCH_SIZE = 32
steps_per_epoch = math.ceil(len(train_ds) / BATCH_SIZE)
TOTAL_STEPS = NUM_EPOCHS * steps_per_epoch

ada_cfg = AdaLoraConfig(
    init_r=11,
    target_r=8,
    lora_alpha=16,
    lora_dropout=0.15,
    target_modules=["query","value"],
    bias="none",
    modules_to_save=["classifier"],
    total_step=TOTAL_STEPS,
    tinit=0,
    tfinal=TOTAL_STEPS,
    deltaT=TOTAL_STEPS+1,
    beta1=0.9,
    beta2=0.999,
    task_type="SEQ_CLS"
)
peft_model = get_peft_model(student, ada_cfg)

# print trainable params
total = sum(p.numel() for p in peft_model.parameters())
trainable = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
print(f"🔢 Total params: {total:,} — Trainable: {trainable:,}")

# Callback to Excel‑log per‑epoch
class EpochLogger(TrainerCallback):
    def __init__(self, path="results/epoch_metrics.xlsx"):
        self.path    = path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # epoch → {"train_loss": float or None, "eval_accuracy": float or None}
        self.metrics = {}

    def on_log(self, args, state, control, logs=None, **kwargs):
        logs = logs or {}
        # Use integer epoch
        epoch = int(state.epoch or 0)

        # Initialize if needed
        if epoch not in self.metrics:
            self.metrics[epoch] = {"train_loss": None, "eval_accuracy": None}

        # Update whichever values are present
        if "loss" in logs:
            self.metrics[epoch]["train_loss"] = logs["loss"]
        if "eval_accuracy" in logs:
            self.metrics[epoch]["eval_accuracy"] = logs["eval_accuracy"]

        # Write out whole table to Excel
        df = pd.DataFrame([
            {"epoch": e, **vals}
            for e, vals in sorted(self.metrics.items())
        ])
        df.to_excel(self.path, index=False)

        # If both metrics are now available for this epoch, print once
        row = self.metrics[epoch]
        if row["train_loss"] is not None and row["eval_accuracy"] is not None:
            print(
                f"[epoch {epoch}] "
                f"loss={row['train_loss']:.4f}  "
                f"acc={row['eval_accuracy']:.4f}"
            )
            
# TrainingArguments
training_args = TrainingArguments(
    output_dir="results",
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=64,
    learning_rate=2e-4,
    warmup_ratio=0.1,
    weight_decay=0.01,
    fp16=True,

    do_train=True,
    do_eval=True,
    eval_strategy="epoch",   # run eval at end of each epoch
    logging_strategy="epoch",# log loss at end of each epoch
    save_strategy="epoch",   # save checkpoint each epoch
    save_total_limit=2,

    load_best_model_at_end=True,
    metric_for_best_model="eval_accuracy",
    greater_is_better=True,

    logging_steps=50,
    report_to="none",
    label_names=["labels"]
)

# Metrics fn
def compute_metrics(p):
    preds = p.predictions.argmax(-1)
    return {"eval_accuracy": accuracy_score(p.label_ids, preds)}

# Initialize Trainer & train
trainer = Trainer(
    model=peft_model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    tokenizer=tok,
    data_collator=collator,
    compute_metrics=compute_metrics,
    callbacks=[EpochLogger()]
)

trainer.train()
trainer.save_model("best_model")
tok.save_pretrained("best_model")


