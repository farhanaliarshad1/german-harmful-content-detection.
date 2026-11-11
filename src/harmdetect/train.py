"""Fine-tune a transformer for one harmful-content task (fake news, hate speech or toxicity).

Usage:
    python -m harmdetect.train --config configs/fake_news_distilbert.yaml
    python -m harmdetect.train --config configs/hate_speech_xlmr.yaml
    python -m harmdetect.train --config configs/toxicity_mbert.yaml
    python -m harmdetect.train --config configs/fake_news_distilbert.yaml --model_name deepset/gbert-base
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    TrainingArguments,
    set_seed,
)

from harmdetect.config import TrainConfig
from harmdetect.data import load_binary_csv, oversample_minority, split_data
from harmdetect.modeling import TextClassificationDataset, WeightedLossTrainer, class_weights, compute_metrics

logger = logging.getLogger("harmdetect.train")


def run(cfg: TrainConfig) -> dict:
    set_seed(cfg.seed)
    labels = dict(enumerate(cfg.label_names))
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ---- data ---------------------------------------------------------------
    df = load_binary_csv(cfg.data_path, cfg.text_col, cfg.label_col)
    splits = split_data(df, cfg.val_size, cfg.test_size, cfg.seed)
    train_df = splits.train
    if cfg.max_train_samples:
        train_df = train_df.sample(min(cfg.max_train_samples, len(train_df)), random_state=cfg.seed)
    if cfg.balance == "oversample":
        train_df = oversample_minority(train_df, cfg.seed)
    weights = class_weights(train_df["label"]) if cfg.balance == "class_weight" else None
    logger.info("[%s] train=%d val=%d test=%d | balance=%s", cfg.task, len(train_df), len(splits.val), len(splits.test), cfg.balance)

    # ---- model --------------------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model_name, num_labels=2, id2label=labels, label2id={v: k for k, v in labels.items()}
    )
    make_ds = lambda d: TextClassificationDataset(d["text"], d["label"], tokenizer, cfg.max_length)  # noqa: E731
    train_ds, val_ds, test_ds = make_ds(train_df), make_ds(splits.val), make_ds(splits.test)

    args = TrainingArguments(
        output_dir=str(out / "checkpoints"),
        num_train_epochs=cfg.epochs,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size * 2,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        warmup_ratio=cfg.warmup_ratio,
        lr_scheduler_type="cosine",
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        logging_steps=50,
        fp16=torch.cuda.is_available(),
        seed=cfg.seed,
        report_to=[],
    )
    trainer = WeightedLossTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,  # early stopping / model selection on VALIDATION, never on test
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=cfg.early_stopping_patience)],
        class_weights=weights,
    )
    trainer.train()

    # ---- final, one-shot test evaluation -----------------------------------
    pred = trainer.predict(test_ds)
    y_pred = np.argmax(pred.predictions, axis=-1)
    y_true = splits.test["label"].to_numpy()
    metrics = {k.replace("test_", ""): round(float(v), 4) for k, v in pred.metrics.items()
               if k.replace("test_", "") in {"accuracy", "precision", "recall", "f1", "macro_f1"}}
    report = {
        "config": cfg.to_dict(),
        "sizes": {"train": len(train_df), "val": len(splits.val), "test": len(splits.test)},
        "test_metrics": metrics,
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "classification_report": classification_report(
            y_true, y_pred, target_names=list(labels.values()), output_dict=True, zero_division=0
        ),
    }

    trainer.save_model(str(out / "model"))
    tokenizer.save_pretrained(str(out / "model"))
    (out / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    splits.test.assign(pred=y_pred).to_csv(out / "test_predictions.csv", index=False)

    logger.info("TEST %s", metrics)
    logger.info("Saved model + metrics to %s", out)
    return report


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    for name in ("data_path", "model_name", "output_dir", "balance"):
        p.add_argument(f"--{name}")
    p.add_argument("--epochs", type=int)
    p.add_argument("--max_train_samples", type=int)
    a = vars(p.parse_args())
    run(TrainConfig.from_yaml(a.pop("config"), **a))


if __name__ == "__main__":
    main()
