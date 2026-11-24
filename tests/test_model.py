import torch

from homer_ai.model import HomerGreekLLM


def test_output_shape(tiny_model):
    ids = torch.tensor([1, 2, 3, 4, 5])
    assert tiny_model(ids).shape == (5, 20)


def test_parameter_count_for_book_1():
    model = HomerGreekLLM(vocab_size=137, embed_dim=32)
    assert sum(p.numel() for p in model.parameters()) == 11_840


def test_causal_later_tokens_do_not_change_earlier_logits(tiny_model):
    a = torch.tensor([1, 2, 3, 4, 5, 6])
    b = a.clone()
    b[4:] = torch.tensor([9, 8])  # change only the last two letters
    with torch.no_grad():
        logits_a, logits_b = tiny_model(a), tiny_model(b)
    assert torch.allclose(logits_a[:4], logits_b[:4], atol=1e-6)
    assert not torch.allclose(logits_a[4:], logits_b[4:])
