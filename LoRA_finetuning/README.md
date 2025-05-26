# 🔧 Finetuning RoBERTa with AdaLoRA on AG News

This project demonstrates parameter-efficient fine-tuning of [`roberta-base`](https://huggingface.co/roberta-base) using **AdaLoRA (Adaptive Low-Rank Adaptation)** for multi-class text classification on the **AG News** dataset. The model classifies news into four categories:
- 🌍 World
- 🏈 Sports
- 💼 Business
- 🧪 Sci/Tech

The entire training process was built and logged within a single Jupyter Notebook: `finetuning-lora.ipynb`.

---

## 📂 Repository Contents

| File/Folder                  | Description |
|-----------------------------|-------------|
| `finetuning-lora.ipynb`     | 📒 Main notebook for training, evaluation, logging, and submission generation |
| `train.py`                  | 🧠 Optional standalone training script |
| `inference.py`              | 🔍 Script for final inference on unlabeled data |
| `best_model/`               | 🗂️ Saved adapter-merged model & tokenizer |
| `best_model-submission.csv` | 📄 Final predictions in Kaggle-ready format |
| `results/epoch_metrics.xlsx`| 📊 Epoch-wise training/evaluation metrics (loss & accuracy) |

---

## 🧠 Highlights

- ✅ **AdaLoRA** enables dynamic rank adaptation for better generalization under parameter constraints.
- ⚡ Fine-tuned using `AutoModelForSequenceClassification` with only **trainable adapter layers**.
- 🧪 Accuracy tracked at every epoch and logged to Excel.
- 💾 Final model saved with `.merge_and_unload()` for efficient deployment.

---

## 📊 Training Details

| Setting               | Value              |
|----------------------|--------------------|
| Model                | `roberta-base`     |
| PEFT Method          | `AdaLoRA`          |
| Epochs               | 3                  |
| Batch Size (train)   | 32                 |
| Optimizer            | AdamW              |
| Metric Tracked       | Accuracy (Eval)    |
| Mixed Precision      | `fp16=True`        |

---

## 📦 Submission

- Inference results are saved to `best_model-submission.csv`.
- Format:  
  | ID | label |
  |----|-------|
  | 0  | 2     |
  | 1  | 0     |
  | .. | ..    |

---

## 🧾 References

- [AdaLoRA: Parameter-Efficient Fine-Tuning with Adaptive Rank](https://arxiv.org/abs/2303.10512)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/index)
- [AG News Dataset](https://huggingface.co/datasets/ag_news)

---