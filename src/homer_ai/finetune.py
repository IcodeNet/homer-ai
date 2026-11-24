"""Fine-tune GPT-2 to sort text messages into spam and not spam (08_02, 08_04).

The data is the SMS Spam Collection (Almeida and Gómez Hidalgo, 2011), from the
UCI Machine Learning Repository, licence CC BY 4.0. It is downloaded once to the
cache folder outside the project, like GPT-2 itself (see `pretrained.py`).

Every function here is built, step by step, in notebook 08_02.
"""

import random
import time
import zipfile
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from homer_ai.gpt import GPTModel
from homer_ai.pretrained import CACHE, download

SPAM_URL = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
SPAM_FILE = CACHE / "sms_spam" / "sms_spam_collection.zip"
SPAM_BYTES = 203_415
SPAM_SHA256 = "1587ea43e58e82b14ff1f5425c88e17f8496bfcdb67a583dbff9eefaf9963ce3"

LABELS = ("ham", "spam")  # class 0 = ham (a normal message), class 1 = spam
PAD_ID = 50256  # GPT-2's <|endoftext|>, used to fill short messages

Message = tuple[str, str]  # (label, text)


def download_sms_spam(path: Path = SPAM_FILE) -> Path:
    """The SMS Spam Collection zip (0.2 MB), downloaded once and checked."""
    return download(SPAM_URL, path, SPAM_BYTES, SPAM_SHA256)


def load_sms_spam(path: Path | None = None) -> list[Message]:
    """All 5,574 messages as `(label, text)`; the label is "ham" or "spam"."""
    path = Path(path) if path else download_sms_spam()
    with zipfile.ZipFile(path) as z:
        lines = z.read("SMSSpamCollection").decode("utf-8").splitlines()
    return [tuple(line.split("\t", 1)) for line in lines if "\t" in line]


def balance(messages: list[Message], seed: int = 123) -> list[Message]:
    """Keep every message of the rarest label, and as many of each other label.

    Which ones are kept is random but fixed by `seed`. The result is shuffled.
    """
    rng = random.Random(seed)
    by_label: dict[str, list[Message]] = {}
    for message in messages:
        by_label.setdefault(message[0], []).append(message)
    smallest = min(len(group) for group in by_label.values())
    kept = [m for group in by_label.values() for m in rng.sample(group, smallest)]
    rng.shuffle(kept)
    return kept


def split(
    messages: list[Message], train: float = 0.7, val: float = 0.1
) -> tuple[list[Message], list[Message], list[Message]]:
    """Cut a (shuffled) list into training, validation and test parts."""
    a = int(len(messages) * train)
    b = a + int(len(messages) * val)
    return messages[:a], messages[a:b], messages[b:]


def encode_messages(
    messages: list[Message], encoding, length: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Messages as one table of token IDs `(messages, length)`, and their labels.

    A message longer than `length` tokens is cut; a shorter one is filled up
    with `PAD_ID`, so every row has the same length. The labels are 0 or 1.
    """
    rows = []
    for _label, text in messages:
        ids = encoding.encode(text)[:length]
        rows.append(ids + [PAD_ID] * (length - len(ids)))
    labels = [LABELS.index(label) for label, _text in messages]
    return torch.tensor(rows), torch.tensor(labels)


def make_classifier(model: GPTModel, n_classes: int = 2, seed: int = 123) -> GPTModel:
    """Turn a text-writing GPT into a classifier, in place.

    Freezes every weight, swaps the output head (one score per token of the
    vocabulary) for a new one with `n_classes` scores, and unfreezes the last
    block and the final layer norm. The new head is trainable too.
    """
    for parameter in model.parameters():
        parameter.requires_grad = False
    torch.manual_seed(seed)
    model.output_head = nn.Linear(model.cfg.emb_dim, n_classes)
    for part in (model.blocks[-1], model.final_norm):
        for parameter in part.parameters():
            parameter.requires_grad = True
    return model


def class_scores(model: GPTModel, ids: torch.Tensor) -> torch.Tensor:
    """One row of class scores per message: the scores at the LAST token.

    The last token is the only one that has seen the whole message (the causal
    mask hides later tokens from earlier ones, see 03_03).
    """
    return model(ids)[:, -1, :]


def accuracy(
    model: GPTModel, ids: torch.Tensor, labels: torch.Tensor, batch_size: int = 32
) -> float:
    """The share of messages the model labels correctly (0.0 to 1.0)."""
    was_training = model.training
    model.eval()
    right = 0
    with torch.no_grad():
        for i in range(0, len(ids), batch_size):
            guesses = class_scores(model, ids[i : i + batch_size]).argmax(dim=-1)
            right += (guesses == labels[i : i + batch_size]).sum().item()
    model.train(was_training)
    return right / len(ids)


def train_classifier(
    model: GPTModel,
    ids: torch.Tensor,
    labels: torch.Tensor,
    epochs: int = 1,
    lr: float = 5e-5,
    batch_size: int = 8,
    seed: int = 123,
    report_every: int = 20,
) -> list[tuple[int, float]]:
    """Train the weights that are not frozen. Returns `(step, average loss)` pairs.

    One step is one batch of `batch_size` messages: guess, measure the loss,
    work out the nudges, nudge (the loop of 05_01). The messages are shuffled
    each epoch, in an order fixed by `seed`.
    """
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=0.1)
    order = torch.Generator().manual_seed(seed)
    history, recent, step = [], [], 0
    started = time.time()
    model.train()
    for epoch in range(epochs):
        shuffled = torch.randperm(len(ids), generator=order)
        for i in range(0, len(ids) - batch_size + 1, batch_size):
            batch = shuffled[i : i + batch_size]
            loss = F.cross_entropy(class_scores(model, ids[batch]), labels[batch])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            recent.append(loss.item())
            step += 1
            if step % report_every == 0:
                history.append((step, sum(recent) / len(recent)))
                print(
                    f"epoch {epoch + 1} step {step:4} | loss {history[-1][1]:.3f} "
                    f"| {time.time() - started:5.1f}s"
                )
                recent = []
    model.eval()
    return history
