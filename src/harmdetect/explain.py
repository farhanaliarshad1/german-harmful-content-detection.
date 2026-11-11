"""SHAP explanations for any trained classifier (fake news, hate speech, toxicity).

Produces per-post HTML text plots and a corpus-level ranking of the words that
push predictions towards the harmful class. Masking is done at word level, so
the ranking shows readable words instead of sub-word pieces such as ``##hr``.

Usage:
    python -m harmdetect.explain --model outputs/fake_news-distilbert-german/model \
        --data outputs/fake_news-distilbert-german/test_predictions.csv --n 200
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from harmdetect.predict import HarmClassifier


def word_importance(values, min_count: int = 3) -> pd.DataFrame:
    """Average SHAP contribution of each (lower-cased) word over a set of explanations."""
    contrib, count = defaultdict(float), defaultdict(int)
    for sv in values:
        for token, v in zip(sv.data, sv.values):
            word = token.strip().lower()
            if len(word) > 2:
                contrib[word] += float(v)
                count[word] += 1
    return pd.DataFrame(
        [(w, contrib[w] / count[w], count[w]) for w in contrib if count[w] >= min_count],
        columns=["word", "mean_shap_towards_positive", "occurrences"],
    ).sort_values("mean_shap_towards_positive", ascending=False)


def main() -> None:
    import shap  # optional dependency

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--data", required=True, help="CSV with a 'text' column")
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--n_html", type=int, default=10, help="how many per-post HTML plots to write")
    p.add_argument("--out_dir", default=None, help="default: <model dir>/../shap")
    a = p.parse_args()

    clf = HarmClassifier(a.model)
    out = Path(a.out_dir) if a.out_dir else Path(a.model).parent / "shap"
    texts = pd.read_csv(a.data)["text"].astype(str).sample(frac=1.0, random_state=42).head(a.n).tolist()

    masker = shap.maskers.Text(r"\W+")
    explainer = shap.Explainer(
        lambda xs: np.array(clf.predict_proba(list(xs))), masker, output_names=[clf.positive_label]
    )
    values = explainer(texts)

    out.mkdir(parents=True, exist_ok=True)
    for i in range(min(a.n_html, len(texts))):
        (out / f"post_{i:03d}.html").write_text(shap.plots.text(values[i], display=False), encoding="utf-8")
    ranking = word_importance(values)
    ranking.to_csv(out / "word_importance.csv", index=False)

    print(f"Top words pushing towards {clf.positive_label.upper()}:\n", ranking.head(15).to_string(index=False))
    print("\nTop words pushing towards NOT harmful:\n", ranking.tail(15).iloc[::-1].to_string(index=False))


if __name__ == "__main__":
    main()
