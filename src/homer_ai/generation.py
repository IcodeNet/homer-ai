"""One next-letter step that works for both models.

`model.py` takes one text as a 1-D list of IDs and always reads 24 letters.
`gpt.py` takes a batch and reads `context_length` letters. This file hides that
difference, so writing code (lessons, server) needs no `if` of its own.
"""

import torch

from homer_ai.gpt import GPTModel

SMALL_MODEL_CONTEXT = 24  # the small model reads 24 letters (02_04)


def context_size(model):
    """How many letters the model can read at once."""
    return model.context_length if isinstance(model, GPTModel) else SMALL_MODEL_CONTEXT


def next_scores(model, ids):
    """The scores (logits) for the letter after `ids`: a tensor of `vocab_size`.

    Only the last `context_size(model)` letters are read. Does not switch the
    model to `eval()` and does not turn off gradients: the caller does that.
    """
    recent = torch.tensor(list(ids)[-context_size(model) :], dtype=torch.long)
    if isinstance(model, GPTModel):
        return model(recent[None])[0, -1]  # add the batch dimension, then drop it
    return model(recent)[-1]
