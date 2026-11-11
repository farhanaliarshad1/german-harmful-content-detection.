"""Unified text-to-text model for all three tasks (task-prefixed mT5).

The course version trained English ``t5-base`` to generate
"fake_news: 1 hate_speech: -1 toxicity: -1", i.e. it learned to *generate* the
missing-label placeholder, and pad tokens were included in the loss. This
version fixes both:

* every labelled (row, task) pair becomes one example:  "hate_speech: <text>" -> "1"
  so the model never sees -1 targets;
* padding in the targets is replaced by -100 so it is ignored by the loss;
* a multilingual checkpoint (mT5) is used because the data is German.

Usage:
    python -m harmdetect.t5_unified --data data/merged_dataset.csv --model_name google/mt5-small --epochs 3
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, get_linear_schedule_with_warmup, set_seed

from harmdetect.data import clean_text

logger = logging.getLogger("harmdetect.t5_unified")
LABEL_COLUMNS = {"fake_news": "isFake", "hate_speech": "isHateSpeech", "toxicity": "isToxic"}


def to_text2text(df: pd.DataFrame) -> pd.DataFrame:
    """Explode the merged corpus into (source, target, task) examples, skipping unlabelled tasks."""
    rows = []
    for task, col in LABEL_COLUMNS.items():
        part = df[df[col] != -1]
        rows.append(pd.DataFrame({
            "source": task + ": " + part["text"],
            "target": part[col].astype(int).astype(str),
            "task": task,
        }))
    return pd.concat(rows, ignore_index=True)


def make_loader(ex: pd.DataFrame, tok, max_len: int, batch_size: int, shuffle: bool) -> DataLoader:
    def collate(batch):
        enc = tok([b["source"] for b in batch], truncation=True, max_length=max_len, padding=True, return_tensors="pt")
        lab = tok([b["target"] for b in batch], padding=True, return_tensors="pt").input_ids
        lab[lab == tok.pad_token_id] = -100  # ignore padding in the loss
        enc["labels"] = lab
        return enc
    return DataLoader(ex.to_dict("records"), batch_size=batch_size, shuffle=shuffle, collate_fn=collate)


@torch.inference_mode()
def evaluate(model, tok, ex: pd.DataFrame, device, max_len: int, batch_size: int = 64) -> dict:
    model.eval()
    preds = []
    for i in range(0, len(ex), batch_size):
        enc = tok(ex["source"].iloc[i : i + batch_size].tolist(), truncation=True, max_length=max_len,
                  padding=True, return_tensors="pt").to(device)
        out = model.generate(**enc, max_new_tokens=3)
        preds.extend(tok.batch_decode(out, skip_special_tokens=True))
    ex = ex.assign(pred=[p.strip()[:1] for p in preds])
    report = {}
    for task, g in ex.groupby("task"):
        y = g["target"].astype(int)
        valid = g["pred"].isin(["0", "1"])
        # an invalid generation is scored as wrong (the opposite of the gold label), never as a free 0/1
        yhat = g["pred"].where(valid, (1 - y).astype(str)).astype(int)
        p, r, f1, _ = precision_recall_fscore_support(y, yhat, average="binary", zero_division=0)
        report[task] = {"n": len(g), "invalid": int((~valid).sum()),
                        "accuracy": round(accuracy_score(y, yhat), 4), "precision": round(p, 4),
                        "recall": round(r, 4), "f1": round(f1, 4)}
    return report


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default="data/merged_dataset.csv")
    p.add_argument("--model_name", default="google/mt5-small")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--max_length", type=int, default=128)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output_dir", default="outputs/t5-unified")
    a = p.parse_args()

    set_seed(a.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = pd.read_csv(a.data).dropna(subset=["text"])
    df["text"] = df["text"].map(clean_text)
    df = df.drop_duplicates(subset="text")
    source = df[list(LABEL_COLUMNS.values())].ne(-1).idxmax(axis=1)
    train_df, test_df = train_test_split(df, test_size=0.2, stratify=source, random_state=a.seed)
    train_ex, test_ex = to_text2text(train_df), to_text2text(test_df)

    tok = AutoTokenizer.from_pretrained(a.model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(a.model_name).to(device)
    loader = make_loader(train_ex, tok, a.max_length, a.batch_size, shuffle=True)
    optim = torch.optim.AdamW(model.parameters(), lr=a.lr)
    total = len(loader) * a.epochs  # schedule matches the actual number of epochs
    sched = get_linear_schedule_with_warmup(optim, int(0.06 * total), total)

    for epoch in range(a.epochs):
        model.train()
        running = 0.0
        for batch in loader:
            loss = model(**batch.to(device)).loss
            optim.zero_grad()
            loss.backward()
            optim.step()
            sched.step()
            running += loss.item()
        logger.info("epoch %d/%d  mean train loss %.4f", epoch + 1, a.epochs, running / len(loader))

    report = evaluate(model, tok, test_ex, device, a.max_length)
    out = Path(a.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out / "model")
    tok.save_pretrained(out / "model")
    (out / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info("TEST %s", json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
