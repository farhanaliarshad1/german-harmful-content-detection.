import torch

from harmdetect.multitask import masked_bce
from harmdetect.t5_unified import to_text2text


def test_masked_bce_ignores_unlabelled_tasks():
    logits = torch.tensor([[2.0, 99.0, -99.0], [-1.0, 99.0, -99.0]])
    labels = torch.tensor([[1.0, -1.0, -1.0], [0.0, -1.0, -1.0]])  # only task 0 labelled
    expected = torch.nn.functional.binary_cross_entropy_with_logits(logits[:, 0], labels[:, 0])
    assert torch.allclose(masked_bce(logits, labels), expected)


def test_masked_bce_averages_over_present_tasks():
    logits = torch.zeros(2, 3)
    labels = torch.tensor([[1.0, -1.0, -1.0], [-1.0, 0.0, -1.0]])
    # every labelled logit is 0 -> each task loss is ln(2); task 2 absent
    assert torch.allclose(masked_bce(logits, labels), torch.log(torch.tensor(2.0)))


def test_t5_examples_skip_unlabelled_tasks():
    import pandas as pd

    df = pd.DataFrame({"text": ["a", "b"], "isFake": [1, -1], "isHateSpeech": [-1, 0], "isToxic": [-1, -1]})
    ex = to_text2text(df)
    assert sorted(ex["source"]) == ["fake_news: a", "hate_speech: b"]
    assert set(ex["target"]) == {"0", "1"}  # never "-1"
