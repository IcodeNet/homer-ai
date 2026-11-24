import dataclasses

import pytest
import torch

from homer_ai.generation import context_size, next_scores
from homer_ai.gpt import GPTConfig, GPTModel


@pytest.fixture
def config():
    return GPTConfig(vocab_size=20, context_length=8, emb_dim=16, n_heads=2, n_layers=2)


@pytest.fixture
def gpt(config):
    torch.manual_seed(0)
    return GPTModel(config).eval()


def formula(c):
    """Weights, counted by hand: embeddings, then per block, then the end."""
    e = c.emb_dim
    attention = 3 * e * e + e * e + e  # q, k, v (no bias), output projection
    feed_forward = (e * 4 * e + 4 * e) + (4 * e * e + e)
    block = attention + feed_forward + 2 * 2 * e  # two layer norms
    return (
        c.vocab_size * e + c.context_length * e + c.n_layers * block + 2 * e
    ) + e * c.vocab_size  # token + position tables, blocks, final norm, head


def test_output_shape(gpt):
    ids = torch.randint(0, 20, (3, 5))
    assert gpt(ids).shape == (3, 5, 20)


def test_parameter_count_follows_the_formula(config, gpt):
    assert sum(p.numel() for p in gpt.parameters()) == formula(config)


def test_parameter_count_of_the_chapter_4_model():
    # the 2-block, 32-wide model counted in 04_05
    config = GPTConfig(
        vocab_size=137, context_length=24, emb_dim=32, n_heads=2, n_layers=2
    )
    assert sum(p.numel() for p in GPTModel(config).parameters()) == 34_816


def test_causal_later_letters_do_not_change_earlier_logits(gpt):
    a = torch.tensor([[1, 2, 3, 4, 5, 6]])
    b = a.clone()
    b[0, 4:] = torch.tensor([9, 8])  # change only the last two letters
    with torch.no_grad():
        logits_a, logits_b = gpt(a), gpt(b)
    assert torch.allclose(logits_a[0, :4], logits_b[0, :4], atol=1e-6)
    assert not torch.allclose(logits_a[0, 4:], logits_b[0, 4:])


def test_more_letters_than_the_context_is_refused(gpt):
    with pytest.raises(ValueError, match="context length"):
        gpt(torch.zeros(1, 9, dtype=torch.long))


def test_config_is_frozen_and_checks_the_heads():
    config = GPTConfig(vocab_size=10)
    with pytest.raises(dataclasses.FrozenInstanceError):
        config.emb_dim = 3
    with pytest.raises(ValueError, match="divisible"):
        GPTConfig(vocab_size=10, emb_dim=30, n_heads=4)
    assert GPTConfig(**config.to_dict()) == config


def test_next_scores_reads_only_the_context(gpt, tiny_model):
    long_text = list(range(1, 20)) * 2  # 38 letters, more than the 8 the GPT reads
    with torch.no_grad():
        assert context_size(gpt) == 8
        assert next_scores(gpt, long_text).shape == (20,)
        assert torch.equal(
            next_scores(gpt, long_text), next_scores(gpt, long_text[-8:])
        )
        assert context_size(tiny_model) == 24  # the small model
        assert next_scores(tiny_model, long_text).shape == (20,)
