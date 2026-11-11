"""SHAP explanations for the fake-news classifier.

Produces per-post HTML text plots and a corpus-level ranking of the words that
push predictions towards "fake". Word-level aggregation (instead of raw WordPiece
pieces like ``##hr``) makes the ranking readable.

Usage:
    python -m fakenews.explain --model outputs/distilbert-german-defakts/model \
        --data outputs/distilbert-german-defakts/test_predictions.csv --n 200
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from fakenews.predict import FakeNewsClassifier


def main() -> None:
    import shap  # optional dependency

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--data", required=True, help="CSV with a 'text' column")
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--n_html", type=int, default=10, help="how many per-post HTML plots to write")
    p.add_argument("--out_dir", default="outputs/shap")
    a = p.parse_args()

    clf = FakeNewsClassifier(a.model)
    texts = pd.read_csv(a.data)["text"].astype(str).sample(frac=1.0, random_state=42).head(a.n).tolist()

    masker = shap.maskers.Text(r"\W+")  # word-level masking
    explainer = shap.Explainer(lambda xs: np.array(clf.predict_proba(list(xs))), masker, output_names=["fake"])
    values = explainer(texts)

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for i in range(min(a.n_html, len(texts))):
        (out / f"post_{i:03d}.html").write_text(shap.plots.text(values[i], display=False), encoding="utf-8")

    contrib, count = defaultdict(float), defaultdict(int)
    for sv in values:
        for token, v in zip(sv.data, sv.values):
            word = token.strip().lower()
            if len(word) > 2:
                contrib[word] += float(v)
                count[word] += 1
    ranking = pd.DataFrame(
        [(w, contrib[w] / count[w], count[w]) for w in contrib if count[w] >= 3],
        columns=["word", "mean_shap_towards_fake", "occurrences"],
    ).sort_values("mean_shap_towards_fake", ascending=False)
    ranking.to_csv(out / "word_importance.csv", index=False)

    print("Top words pushing towards FAKE:\n", ranking.head(15).to_string(index=False))
    print("\nTop words pushing towards NEUTRAL:\n", ranking.tail(15).iloc[::-1].to_string(index=False))


if __name__ == "__main__":
    main()
