"""Model-side utilities: torch dataset, class-weighted Trainer and metrics."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import Dataset
from transformers import Trainer


class TextClassificationDataset(Dataset):
    """Tokenises lazily; padding is done per batch by ``DataCollatorWithPadding``."""

    def __init__(self, texts: Sequence[str], labels: Sequence[int] | None, tokenizer, max_length: int = 128):
        self.encodings = tokenizer(list(texts), truncation=True, max_length=max_length)
        self.labels = None if labels is None else list(labels)

    def __len__(self) -> int:
        return len(self.encodings["input_ids"])

    def __getitem__(self, idx: int) -> dict:
        item = {k: v[idx] for k, v in self.encodings.items()}
        if self.labels is not None:
            item["labels"] = int(self.labels[idx])
        return item


def class_weights(labels: Sequence[int]) -> torch.Tensor:
    classes = np.unique(labels)
    weights = compute_class_weight("balanced", classes=classes, y=np.asarray(labels))
    return torch.tensor(weights, dtype=torch.float)


class WeightedLossTrainer(Trainer):
    """``Trainer`` with class-weighted cross-entropy (no-op when weights are None)."""

    def __init__(self, *args, class_weights: torch.Tensor | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        weight = None if self.class_weights is None else self.class_weights.to(outputs.logits.device)
        loss = torch.nn.functional.cross_entropy(outputs.logits, labels, weight=weight)
        return (loss, outputs) if return_outputs else loss


def compute_metrics(eval_pred) -> dict:
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    p, r, f1, _ = precision_recall_fscore_support(labels, preds, average="binary", zero_division=0)
    _, _, macro_f1, _ = precision_recall_fscore_support(labels, preds, average="macro", zero_division=0)
    return {
        "accuracy": accuracy_score(labels, preds),
        "precision": p,
        "recall": r,
        "f1": f1,
        "macro_f1": macro_f1,
    }
