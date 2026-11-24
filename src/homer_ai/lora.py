"""LoRA: train two small tables next to a frozen layer (built step by step in 08_04).

A linear layer has a weight table `W` of shape `(outputs, inputs)`. Fine-tuning
changes it to `W + ΔW`. LoRA (low-rank adaptation) does not learn the big `ΔW`.
It learns two thin tables whose product has the same shape:

    ΔW = B @ A        A: (rank, inputs)    B: (outputs, rank)

With 768 inputs, 768 outputs and rank 8, that is 2 * 8 * 768 = 12,288 numbers
instead of 589,824. `W` itself stays frozen.
"""

import math

import torch
from torch import nn


class LoRALinear(nn.Module):
    """A frozen `nn.Linear` plus a trainable low-rank change, `scale * B @ A`.

    `B` starts at zero, so at first the layer gives exactly what the frozen
    layer gave. `alpha / rank` scales the change (a common choice is
    `alpha = 2 * rank`).
    """

    def __init__(self, linear: nn.Linear, rank: int = 8, alpha: float = 16.0):
        super().__init__()
        if rank < 1:
            raise ValueError(f"rank must be at least 1, not {rank}")
        self.linear = linear
        for parameter in linear.parameters():
            parameter.requires_grad = False  # frozen: the optimizer leaves it alone
        self.A = nn.Parameter(torch.empty(rank, linear.in_features))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))  # small random numbers
        self.B = nn.Parameter(torch.zeros(linear.out_features, rank))
        self.scale = alpha / rank

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x) + self.scale * (x @ self.A.T @ self.B.T)

    def merged(self) -> nn.Linear:
        """One plain `nn.Linear` with the change added into the weight table."""
        old = self.linear
        new = nn.Linear(old.in_features, old.out_features, bias=old.bias is not None)
        with torch.no_grad():
            new.weight.copy_(old.weight + self.scale * (self.B @ self.A))
            if old.bias is not None:
                new.bias.copy_(old.bias)
        return new


def apply_lora(
    module: nn.Module, rank: int = 8, alpha: float = 16.0, skip: tuple[str, ...] = ()
) -> int:
    """Wrap every `nn.Linear` inside `module` in a `LoRALinear`, in place.

    Children whose name is in `skip` are left alone (for example a new
    classification head, which we train in full). Returns how many layers were
    wrapped.
    """
    wrapped = 0
    for name, child in module.named_children():
        if name in skip:
            continue
        if isinstance(child, nn.Linear):
            setattr(module, name, LoRALinear(child, rank, alpha))
            wrapped += 1
        elif not isinstance(child, LoRALinear):
            wrapped += apply_lora(child, rank, alpha, skip)
    return wrapped


def count_parameters(model: nn.Module) -> tuple[int, int]:
    """(numbers that training may change, all numbers) of a model."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable, total
