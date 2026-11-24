import torch
from torch import nn
from torch.nn import functional as F


class HomerGreekLLM(nn.Module):
    def __init__(self, vocab_size, embed_dim=32):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.query = nn.Linear(embed_dim, embed_dim, bias=False)
        self.key = nn.Linear(embed_dim, embed_dim, bias=False)
        self.value = nn.Linear(embed_dim, embed_dim, bias=False)
        self.head = nn.Linear(embed_dim, vocab_size, bias=False)

    def forward(self, idx):
        seq_len = idx.shape[0]
        x = self.embedding(idx)
        Q, K, V = self.query(x), self.key(x), self.value(x)

        scores = (Q @ K.T) / (x.shape[-1] ** 0.5)
        mask = torch.triu(torch.ones(seq_len, seq_len) * float("-inf"), diagonal=1)
        weights = F.softmax(scores + mask, dim=-1)

        context = weights @ V
        return self.head(context)
