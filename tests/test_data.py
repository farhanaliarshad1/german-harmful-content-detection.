import pandas as pd
import pytest

from harmdetect.data import clean_text, load_binary_csv, load_defakts, oversample_minority, split_data
from harmdetect.llm_baseline import parse_labels
from harmdetect.prompts import build_messages


def test_clean_text_normalises_urls_mentions_entities():
    raw = "@max_mustermann Lest das!!  https://t.co/AbC123 &amp; teilt   es"
    assert clean_text(raw) == "@user Lest das!! [URL] & teilt es"


@pytest.fixture
def csv(tmp_path):
    rows = [(f"Tweet Nummer {i}", int(i % 3 == 0)) for i in range(300)]
    rows += [("Duplikat", 1), ("Duplikat", 1), ("Widerspruch", 0), ("Widerspruch", 1), (None, 1), ("x", "bad")]
    path = tmp_path / "defakts.csv"
    pd.DataFrame(rows, columns=["text", "binary_label"]).to_csv(path, index=False)
    return path


def test_load_drops_duplicates_conflicts_and_invalid(csv):
    df = load_defakts(str(csv))
    assert df["text"].is_unique
    assert "Widerspruch" not in set(df["text"])
    assert set(df["label"]) == {0, 1}
    assert len(df) == 301  # 300 tweets + one copy of "Duplikat"


def test_split_is_disjoint_and_oversampling_stays_in_train(csv):
    df = load_defakts(str(csv))
    s = split_data(df, val_size=0.1, test_size=0.2, seed=0)
    train_texts, val_texts, test_texts = set(s.train.text), set(s.val.text), set(s.test.text)
    assert not (train_texts & test_texts) and not (train_texts & val_texts) and not (val_texts & test_texts)
    assert len(s.train) + len(s.val) + len(s.test) == len(df)

    balanced = oversample_minority(s.train, seed=0)
    counts = balanced["label"].value_counts()
    assert counts[0] == counts[1]
    assert set(balanced.text) == train_texts  # no val/test text leaks into training


@pytest.mark.parametrize(
    "answer,expected",
    [("0,1,1", (0, 1, 1)), (" 1, 0 ,0\n", (1, 0, 0)), ("Labels: 0,0,1", (0, 0, 1)), ("1", None), ("unsure", None), (None, None)],
)
def test_parse_labels(answer, expected):
    assert parse_labels(answer) == expected


def test_few_shot_messages_end_with_user_text():
    msgs = build_messages("Testpost", few_shot=True)
    assert msgs[0]["role"] == "system" and msgs[-1] == {"role": "user", "content": "Testpost"}
    assert len(build_messages("Testpost", few_shot=False)) == 2


def test_generic_loader_for_hate_and_toxicity_schemas(tmp_path):
    for label_col in ("label", "toxic"):
        path = tmp_path / f"{label_col}.csv"
        pd.DataFrame({"text": ["Hallo Welt", "Du Idiot 🤥"], label_col: [0, 1]}).to_csv(path, index=False)
        df = load_binary_csv(str(path), label_col=label_col)
        assert list(df.columns) == ["text", "label"] and df["text"].iloc[1] == "Du Idiot 🤥"  # emojis kept
