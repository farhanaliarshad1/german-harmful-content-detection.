import torch

from fakenews.multitask import masked_bce


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
