# Text Classification Pipeline with Hugging Face Transformers

A text classification pipeline that fine-tunes **DistilBERT** to classify news posts from the **20 Newsgroups** dataset into one of 20 topic categories, built with Hugging Face Transformers and PyTorch, and trained on a GPU in Google Colab.

## Overview

This project covers the full workflow of fine-tuning a pretrained transformer for a downstream NLP task:
- Loading and exploring the 20 Newsgroups dataset
- Tokenizing raw text with a pretrained transformer tokenizer
- Fine-tuning `distilbert-base-uncased` for multi-class classification
- Evaluating model accuracy on a held-out test set
- Running inference on both custom text and real test examples

## Dataset

**20 Newsgroups** — 18,846 newsgroup posts across 20 topics (e.g. `rec.autos`, `sci.space`, `talk.politics.mideast`). Loaded via `sklearn.datasets.fetch_20newsgroups`, no manual download needed. Headers, footers, and quoted replies are stripped so the model learns from the actual post content rather than metadata. Split 80/20 into train and test sets.

## Model

**DistilBERT** (`distilbert-base-uncased`) — a distilled, lighter version of BERT that retains most of BERT's language understanding at a fraction of the size and inference time. A fresh classification head is attached on top and fine-tuned for this 20-class task.

## Pipeline

1. **Data loading & exploration** — load the dataset, inspect sample texts, labels, and class balance
2. **Train/test split** — 80/20 split of the data
3. **Tokenization** — convert raw text into `input_ids` and `attention_mask` tensors using the DistilBERT tokenizer (truncated/padded to max length 256)
4. **Dataset wrapping** — package tokenized data into a PyTorch `Dataset`
5. **Model loading** — load `distilbert-base-uncased` with a 20-way classification head
6. **Training** — fine-tune using Hugging Face's `Trainer` API on a Colab T4 GPU
7. **Evaluation** — measure accuracy and loss on the test set via `trainer.evaluate()`
8. **Inference** — classify both custom example sentences and real test-set posts, comparing predictions against true labels

## Requirements

```
transformers
torch
scikit-learn
numpy
```

Install with:
```bash
pip install -r requirements.txt
```

## Usage

Built and trained in **Google Colab** using a T4 GPU runtime (Runtime > Change runtime type > GPU).

1. Open the notebook and run all cells in order.
2. Confirm GPU is active — the first cell checks `torch.cuda.is_available()`.
3. Training runs via `trainer.train()`.
4. Evaluate with `trainer.evaluate()`.
5. Run the final cells to classify custom text and real test examples.

> Note: after GPU training, tokenized inputs must be moved to the same device as the model (`inputs = {k: v.to(model.device) for k, v in inputs.items()}`), since the tokenizer creates tensors on CPU by default. Skipping this causes a `RuntimeError: Expected all tensors to be on the same device`.

## Results

| Metric | Value |
|---|---|
| Hardware | Colab T4 GPU |
| Epochs | 1 |
| Batch size | 32 |
| Max sequence length | 256 |
| Training loss (final) | 1.41 |
| Eval loss | 1.35 |
| **Eval accuracy** | **65.0%** |

For reference, random guessing across 20 balanced classes would score ~5% accuracy, so 65% after a single epoch indicates the model learned meaningful patterns. Spot-checking predictions against real test examples and true labels confirmed the model gets clearly on-topic predictions right (e.g. correctly identifying a baseball post as `rec.sport.baseball`), with some expected confusion between closely related technical categories (e.g. `sci.electronics` vs `comp.windows.x`).

## Project Structure

```
.
├── textclasshugface.ipynb   # Full pipeline: data loading through inference
├── requirements.txt          # Python dependencies
└── README.md
```

## Future Improvements

- Train for more epochs and on the full dataset (loss was still decreasing at epoch 1)
- Try alternate base models (`bert-base-uncased`, `roberta-base`)
- Add a confusion matrix to see which categories get confused most often
- Wrap inference in a small API (Flask/FastAPI) for real-time classification
- Tune learning rate and sequence length further

## Acknowledgments

Built as a learning project following the Hugging Face `Trainer` API workflow, using the classic 20 Newsgroups dataset.
