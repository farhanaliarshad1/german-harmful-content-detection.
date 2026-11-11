"""Data pipeline shared by all tasks: loading, cleaning, leakage-safe splitting, balancing.

Every task is a binary text-classification CSV (1 = harmful, 0 = not harmful):

=============  =========================  ============  =================
task           dataset                    text column   label column
=============  =========================  ============  =================
fake_news      DeFaktS (German tweets)    text          binary_label
hate_speech    GAHD (adversarial German)  text          label
toxicity       DeTox / GermEval 2018+21   text          toxic
=============  =========================  ============  =================

Cleaning is deliberately light: casing, punctuation, digits and emojis are kept
because German capitalisation and emoji/punctuation carry signal (e.g. sarcasm
markers in hate speech). Only links, mentions, HTML entities and whitespace are
normalised.
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


def load_binary_csv(path: str, text_col: str = "text", label_col: str = "label") -> pd.DataFrame:
    """Load and clean a binary classification CSV. Returns columns ``text`` and ``label``."""
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


# backwards-compatible alias
def load_defakts(path: str, text_col: str = "text", label_col: str = "binary_label") -> pd.DataFrame:
    return load_binary_csv(path, text_col, label_col)
