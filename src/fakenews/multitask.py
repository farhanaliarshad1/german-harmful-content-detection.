"""Unified multi-task BERT: one shared German encoder, one head per task.

Each row of the merged corpus is annotated for only one task (fake news from
DeFaktS, hate speech from GAHD, toxicity from DeTox); the others are -1. The
loss for a head is computed only on rows where that task is labelled.

Usage:
    python -m fakenews.multitask --data data/merged_dataset.csv --epochs 3
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer, DataCollatorWithPadding, get_linear_schedule_with_warmup, set_seed

from fakenews.data import clean_text

logger = logging.getLogger("fakenews.multitask")
TASKS = ["isFake", "isHateSpeech", "isToxic"]


class MultiTaskDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, max_length: int):
        self.enc = tokenizer(df["text"].tolist(), truncation=True, max_length=max_length)
        self.labels = df[TASKS].to_numpy(dtype=np.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return {"input_ids": self.enc["input_ids"][i], "attention_mask": self.enc["attention_mask"][i], "labels": self.labels[i]}


class MultiTaskBert(nn.Module):
    def __init__(self, model_name: str, n_tasks: int = len(TASKS), dropout: float = 0.3):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        self.heads = nn.Linear(self.encoder.config.hidden_size, n_tasks)  # one logit per task

    def forward(self, input_ids, attention_mask):
        cls = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0]
        return self.heads(self.dropout(cls))


def masked_bce(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Mean BCE per task over labelled rows only, then averaged over tasks present in the batch."""
    mask = labels != -1
    losses = nn.functional.binary_cross_entropy_with_logits(logits, labels.clamp(min=0), reduction="none")
    per_task = (losses * mask).sum(0) / mask.sum(0).clamp(min=1)
    return per_task[mask.any(0)].mean()


@torch.inference_mode()
def evaluate(model, loader, device) -> dict:
    model.eval()
    all_logits, all_labels = [], []
    for batch in loader:
        labels = batch.pop("labels")
        all_logits.append(model(**{k: v.to(device) for k, v in batch.items()}).cpu())
        all_labels.append(labels)
    logits, labels = torch.cat(all_logits), torch.cat(all_labels)
    preds = (logits > 0).int()
    report = {}
    for t, task in enumerate(TASKS):
        m = labels[:, t] != -1
        y, yhat = labels[m, t].int().numpy(), preds[m, t].numpy()
        p, r, f1, _ = precision_recall_fscore_support(y, yhat, average="binary", zero_division=0)
        report[task] = {"n": int(m.sum()), "accuracy": round(accuracy_score(y, yhat), 4),
                        "precision": round(p, 4), "recall": round(r, 4), "f1": round(f1, 4)}
    return report


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default="data/merged_dataset.csv")
    p.add_argument("--model_name", default="bert-base-german-cased")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--max_length", type=int, default=128)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output_dir", default="outputs/multitask-bert")
    a = p.parse_args()

    set_seed(a.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = pd.read_csv(a.data).dropna(subset=["text"])
    df["text"] = df["text"].map(clean_text)
    df = df.drop_duplicates(subset="text")
    # stratify on which task a row belongs to, so every task is represented in test
    source = df[TASKS].ne(-1).idxmax(axis=1)
    train_df, test_df = train_test_split(df, test_size=0.2, stratify=source, random_state=a.seed)

    tok = AutoTokenizer.from_pretrained(a.model_name)
    collate = DataCollatorWithPadding(tok)
    train_dl = DataLoader(MultiTaskDataset(train_df, tok, a.max_length), batch_size=a.batch_size, shuffle=True, collate_fn=collate)
    test_dl = DataLoader(MultiTaskDataset(test_df, tok, a.max_length), batch_size=a.batch_size * 2, collate_fn=collate)

    model = MultiTaskBert(a.model_name).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    total = len(train_dl) * a.epochs
    sched = get_linear_schedule_with_warmup(optim, int(0.1 * total), total)

    for epoch in range(a.epochs):
        model.train()
        running = 0.0
        for batch in train_dl:
            labels = batch.pop("labels").to(device)
            loss = masked_bce(model(**{k: v.to(device) for k, v in batch.items()}), labels)
            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            sched.step()
            running += loss.item()
        logger.info("epoch %d/%d  mean train loss %.4f", epoch + 1, a.epochs, running / len(train_dl))

    report = evaluate(model, test_dl, device)
    out = Path(a.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out / "multitask_bert.pt")
    (out / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info("TEST %s", json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
