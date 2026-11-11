"""DeFaktS data pipeline: loading, cleaning, leakage-safe splitting and balancing.

The DeFaktS corpus (German tweets annotated for disinformation framing) is loaded
from a CSV with a text column and a binary label column (1 = fake/disinformation,
0 = neutral). All cleaning keeps casing because the downstream model is cased.
"""

from __future__ import annotations

import html
import logging
import re
from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

URL_RE = re.compile(r"https?://\S+|www\.\S+")
MENTION_RE = re.compile(r"@\w{1,30}")
WHITESPACE_RE = re.compile(r"\s+")

URL_TOKEN = "[URL]"
USER_TOKEN = "@user"


def clean_text(text: str) -> str:
    """Normalise a single tweet.

    - unescape HTML entities (``&amp;`` -> ``&``)
    - replace links with ``[URL]`` so the model cannot memorise shortened t.co links
    - anonymise user mentions to ``@user``
    - collapse whitespace
    """
    text = html.unescape(str(text))
    text = URL_RE.sub(URL_TOKEN, text)
    text = MENTION_RE.sub(USER_TOKEN, text)
    return WHITESPACE_RE.sub(" ", text).strip()


def load_defakts(path: str, text_col: str = "text", label_col: str = "binary_label") -> pd.DataFrame:
    """Load and clean DeFaktS. Returns a frame with columns ``text`` and ``label``."""
    raw = pd.read_csv(path)
    missing = {text_col, label_col} - set(raw.columns)
    if missing:
        raise ValueError(f"{path} is missing columns {missing}; found {list(raw.columns)}")

    df = raw[[text_col, label_col]].rename(columns={text_col: "text", label_col: "label"})
    n0 = len(df)
    df = df.dropna()
    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    df = df[df["label"].isin([0, 1])].astype({"label": int})
    df["text"] = df["text"].map(clean_text)
    df = df[df["text"].str.len() > 0]

    # Exact duplicates (retweets, copy-pasted posts) would leak across splits.
    # If the same text carries conflicting labels, drop it entirely.
    conflicting = df.groupby("text")["label"].nunique()
    df = df[~df["text"].isin(conflicting[conflicting > 1].index)]
    df = df.drop_duplicates(subset="text").reset_index(drop=True)

    logger.info(
        "Loaded %d rows -> %d after cleaning/dedup (label dist: %s)",
        n0, len(df), df["label"].value_counts().to_dict(),
    )
    return df


@dataclass
class Splits:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame


def split_data(df: pd.DataFrame, val_size: float = 0.1, test_size: float = 0.2, seed: int = 42) -> Splits:
    """Stratified train/val/test split.

    The validation set is used for early stopping and checkpoint selection; the
    test set is touched exactly once, for the final report.
    """
    train_val, test = train_test_split(df, test_size=test_size, stratify=df["label"], random_state=seed)
    rel_val = val_size / (1.0 - test_size)
    train, val = train_test_split(train_val, test_size=rel_val, stratify=train_val["label"], random_state=seed)
    return Splits(train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True))


def oversample_minority(train: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """Random-oversample the minority class. Apply to the TRAIN split only.

    Oversampling before splitting puts copies of the same tweet into train and
    test, which inflates test scores (the bug in the original course notebook).
    """
    counts = train["label"].value_counts()
    if counts.nunique() == 1:
        return train
    majority = counts.idxmax()
    parts = [train[train["label"] == majority]]
    for label, n in counts.items():
        if label == majority:
            continue
        minority = train[train["label"] == label]
        extra = minority.sample(counts.max() - n, replace=True, random_state=seed)
        parts.extend([minority, extra])  # keep every original row, add random duplicates
    return pd.concat(parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)
