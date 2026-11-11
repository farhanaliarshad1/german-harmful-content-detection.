"""Run a trained fake-news classifier on raw text or a CSV file.

Usage:
    python -m fakenews.predict --model outputs/distilbert-german-defakts/model --text "Die Regierung verschweigt die Wahrheit!"
    python -m fakenews.predict --model outputs/.../model --csv posts.csv --out predictions.csv
"""

from __future__ import annotations

import argparse

import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from fakenews.data import clean_text


class FakeNewsClassifier:
    def __init__(self, model_dir: str, device: str | None = None, max_length: int = 128):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(self.device).eval()
        self.max_length = max_length

    @torch.inference_mode()
    def predict_proba(self, texts: list[str], batch_size: int = 32) -> list[float]:
        """Probability that each text is fake news / disinformation."""
        probs: list[float] = []
        for i in range(0, len(texts), batch_size):
            batch = [clean_text(t) for t in texts[i : i + batch_size]]
            enc = self.tokenizer(batch, truncation=True, max_length=self.max_length, padding=True, return_tensors="pt")
            logits = self.model(**enc.to(self.device)).logits
            probs.extend(torch.softmax(logits, dim=-1)[:, 1].cpu().tolist())
        return probs


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--text")
    g.add_argument("--csv")
    p.add_argument("--text_col", default="text")
    p.add_argument("--out", default="predictions.csv")
    p.add_argument("--threshold", type=float, default=0.5)
    a = p.parse_args()

    clf = FakeNewsClassifier(a.model)
    if a.text:
        prob = clf.predict_proba([a.text])[0]
        print(f"{'FAKE' if prob >= a.threshold else 'NEUTRAL'}  (p_fake={prob:.3f})")
        return
    df = pd.read_csv(a.csv)
    df["p_fake"] = clf.predict_proba(df[a.text_col].astype(str).tolist())
    df["is_fake_pred"] = (df["p_fake"] >= a.threshold).astype(int)
    df.to_csv(a.out, index=False)
    print(f"Wrote {len(df)} predictions to {a.out}")


if __name__ == "__main__":
    main()
