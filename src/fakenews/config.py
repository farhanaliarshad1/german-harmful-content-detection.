"""Typed experiment config loaded from YAML."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path

import yaml


@dataclass
class TrainConfig:
    # data
    data_path: str = "data/fake_news.csv"
    text_col: str = "text"
    label_col: str = "binary_label"
    val_size: float = 0.1
    test_size: float = 0.2
    balance: str = "class_weight"  # one of: none | class_weight | oversample

    # model
    model_name: str = "distilbert-base-german-cased"
    max_length: int = 128

    # optimisation
    epochs: int = 5
    batch_size: int = 16
    learning_rate: float = 3e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    early_stopping_patience: int = 2
    seed: int = 42

    # debugging: cap the number of training rows (0 = use all)
    max_train_samples: int = 0

    output_dir: str = "outputs/distilbert-german-defakts"

    @classmethod
    def from_yaml(cls, path: str | Path, **overrides) -> "TrainConfig":
        with open(path, encoding="utf-8") as f:
            values = yaml.safe_load(f) or {}
        values.update({k: v for k, v in overrides.items() if v is not None})
        known = {f.name for f in fields(cls)}
        unknown = set(values) - known
        if unknown:
            raise ValueError(f"Unknown config keys: {sorted(unknown)}")
        cfg = cls(**values)
        if cfg.balance not in {"none", "class_weight", "oversample"}:
            raise ValueError(f"balance must be none|class_weight|oversample, got {cfg.balance!r}")
        return cfg

    def to_dict(self) -> dict:
        return asdict(self)
