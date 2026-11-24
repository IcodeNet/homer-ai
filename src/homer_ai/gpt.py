"""The GPT you build in chapter 4 (notebooks 04_01 to 04_05), as ready-to-use code.

The picture, for one text of `n` letters (`B` texts at once):

    IDs (B, n) -> token + position rows (B, n, emb_dim)
               -> n_layers x TransformerBlock (B, n, emb_dim)
               -> final layer norm -> output head -> logits (B, n, vocab_size)

`model.py` is the small one-head model of chapter 1. This file is the full model:
several heads, position rows, feed-forward steps, layer norms and shortcuts.
Unlike `model.py`, it takes a *batch* of texts: IDs of shape `(batch, letters)`.
"""

from dataclasses import asdict, dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class GPTConfig:
    """Every size of the model in one place (see 04_00)."""

    vocab_size: int  # how many different tokens (137 for Book 1 of the Odyssey)
    context_length: int = 64  # the most letters the model can look at
    emb_dim: int = 128  # numbers per letter (the "width")
    n_heads: int = 4  # attention heads; must divide emb_dim
    n_layers: int = 4  # how many blocks are stacked
    drop_rate: float = 0.1  # share of numbers switched off while training
    qkv_bias: bool = False  # does the query/key/value layer add a bias?

    def __post_init__(self):
        if self.emb_dim % self.n_heads:
            raise ValueError(
                f"emb_dim ({self.emb_dim}) must be divisible by "
                f"n_heads ({self.n_heads})"
            )

    def to_dict(self):
        """A plain dict, to save inside a checkpoint."""
        return asdict(self)


class MultiHeadAttention(nn.Module):
    """Causal attention with several heads, all computed at once (04_04)."""

    def __init__(self, emb_dim, n_heads, context_length, drop_rate, qkv_bias=False):
        super().__init__()
        self.n_heads = n_heads
        self.head_size = emb_dim // n_heads
        self.query = nn.Linear(emb_dim, emb_dim, bias=qkv_bias)
        self.key = nn.Linear(emb_dim, emb_dim, bias=qkv_bias)
        self.value = nn.Linear(emb_dim, emb_dim, bias=qkv_bias)
        self.out_proj = nn.Linear(emb_dim, emb_dim)  # the output projection
        self.dropout = nn.Dropout(drop_rate)
        mask = torch.triu(torch.ones(context_length, context_length), diagonal=1).bool()
        self.register_buffer("mask", mask)  # True = hidden (the future)

    def forward(self, x):
        batch, length, emb_dim = x.shape

        def split(t):  # (batch, length, emb_dim) -> (batch, heads, length, head_size)
            return t.view(batch, length, self.n_heads, self.head_size).transpose(1, 2)

        q, k, v = split(self.query(x)), split(self.key(x)), split(self.value(x))
        scores = q @ k.transpose(2, 3) / self.head_size**0.5
        scores = scores.masked_fill(self.mask[:length, :length], float("-inf"))
        weights = self.dropout(torch.softmax(scores, dim=-1))
        mixed = weights @ v
        return self.out_proj(mixed.transpose(1, 2).reshape(batch, length, emb_dim))


class FeedForward(nn.Sequential):
    """Two linear layers with a GELU curve between them (04_02)."""

    def __init__(self, emb_dim):
        super().__init__(
            nn.Linear(emb_dim, 4 * emb_dim),
            nn.GELU(),
            nn.Linear(4 * emb_dim, emb_dim),
        )


class TransformerBlock(nn.Module):
    """Norm, attention, shortcut; norm, feed-forward, shortcut (04_04)."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.attention = MultiHeadAttention(
            cfg.emb_dim, cfg.n_heads, cfg.context_length, cfg.drop_rate, cfg.qkv_bias
        )
        self.feed_forward = FeedForward(cfg.emb_dim)
        self.norm1 = nn.LayerNorm(cfg.emb_dim)  # layer norm (04_01)
        self.norm2 = nn.LayerNorm(cfg.emb_dim)
        self.dropout = nn.Dropout(cfg.drop_rate)

    def forward(self, x):
        x = x + self.dropout(self.attention(self.norm1(x)))  # shortcut (04_03)
        return x + self.dropout(self.feed_forward(self.norm2(x)))


class GPTModel(nn.Module):
    """The full model (04_05): IDs `(batch, letters)` in, logits out."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.token_embedding = nn.Embedding(cfg.vocab_size, cfg.emb_dim)
        self.position_embedding = nn.Embedding(cfg.context_length, cfg.emb_dim)
        self.dropout = nn.Dropout(cfg.drop_rate)
        self.blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg.n_layers)]
        )
        self.final_norm = nn.LayerNorm(cfg.emb_dim)
        self.output_head = nn.Linear(cfg.emb_dim, cfg.vocab_size, bias=False)

    @property
    def context_length(self):
        return self.cfg.context_length

    def forward(self, ids):
        """ids: `(batch, letters)`, at most `context_length` letters. Returns the
        logits `(batch, letters, vocab_size)`: one score per token, per letter."""
        if ids.shape[1] > self.cfg.context_length:
            raise ValueError(
                f"{ids.shape[1]} letters is more than the context length "
                f"({self.cfg.context_length}); feed in only the last ones"
            )
        positions = torch.arange(ids.shape[1], device=ids.device)
        x = self.token_embedding(ids) + self.position_embedding(positions)
        x = self.blocks(self.dropout(x))
        return self.output_head(self.final_norm(x))
