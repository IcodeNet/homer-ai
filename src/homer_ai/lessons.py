"""Helpers the course notebooks share, so each notebook can show only its own idea.

These are not learner exercises: `model.py`, `tokenizer.py` and `train_homer.py`
are. Every function here is built, step by step, in the notebook named in its
docstring; the notebooks import the finished version after that. The one
exception is `check`, which the "Your turn" cells of every notebook call to
say whether an answer is right.

Nothing here ever writes `homer_model.pt`: the model file is only read.
"""

import math

import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR

from homer_ai.checkpoint import load_checkpoint
from homer_ai.generation import next_scores
from homer_ai.model import HomerGreekLLM
from homer_ai.tokenizer import CharTokenizer
from homer_ai.train_homer import (
    BOOKS_DIR,
    GPT_MODEL_FILE,
    MODEL_FILE,
    load_books,
)

CHUNK = 24  # the model reads 24 characters at a time (02_04)
HOMER = BOOKS_DIR.parents[1]  # the project's data/homer folder
# the ready-made model that ships with the repo (models/ is next to data/)
READY_MADE_MODEL = BOOKS_DIR.parents[3] / "models" / "odyssey_book1.pt"
READY_MADE_GPT = BOOKS_DIR.parents[3] / "models" / "odyssey_book1_gpt.pt"


def _plain(value):
    """Tensors and tuples as plain lists, so two answers compare by their numbers."""
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return value


def _same(got, want, tol):
    """True if two plain values are equal; numbers may differ by a rounding error."""
    if isinstance(want, list):
        return (
            isinstance(got, list)
            and len(got) == len(want)
            and all(_same(g, w, tol) for g, w in zip(got, want, strict=True))
        )
    numbers = (int, float)
    if isinstance(want, float) or isinstance(got, float):
        if not (isinstance(got, numbers) and isinstance(want, numbers)):
            return False
        return got == want or math.isclose(got, want, rel_tol=tol, abs_tol=tol)
    return got == want


def _shown(value):
    """A value ready to print: tensors as lists, floats rounded to 4 places."""
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, float):
        return round(value, 4)
    if isinstance(value, tuple):
        return tuple(_shown(v) for v in value)
    if isinstance(value, list):
        return [_shown(v) for v in value]
    return value


def _text(value, show=None, limit=70):
    """A value as one short line for a message. `show` makes it readable first."""
    try:
        if show is not None:
            value = show(value)
        text = repr(_shown(value))
    except Exception:  # an answer of an unexpected kind: show it as it is
        text = repr(value)
    return text if len(text) <= limit else text[: limit - 1] + "…"


def check(
    name, got, want=None, *, test=None, expected=None, coach="", show=None, tol=1e-4
):
    """Check one answer of a "Your turn" exercise and print what to do next.

    `got` is the learner's answer and `want` the right one. Tensors, lists and
    tuples are compared by their numbers, and floats may differ by `tol`. For
    an answer with more than one right value, pass `test` (a function that
    takes the answer and returns True or False) and `expected` (the right
    answer in words). `show` turns both values into something readable for
    the message, for example `tokenizer.decode`.

    It prints one of three things and never raises:

        ⏳ the answer is still None
        ✅ right
        ❌ wrong: what was expected and what it got, then 🚨 and `coach`,
           a line about the most likely mistake
    """
    try:
        if got is None:
            print(
                f"⏳ {name}. Not filled in yet: replace the None in the cell above, "
                "run it, then run this check again."
            )
            return
        if test is not None:
            right = bool(test(got))
            want_text = expected or "something else"
        else:
            right = _same(_plain(got), _plain(want), tol)
            want_text = expected or _text(want, show)
        if right:
            print(f"✅ {name}: {_text(got, show)}")
            return
        print(f"❌ {name}: expected {want_text}, got {_text(got, show)}")
    except Exception as error:  # e.g. an answer of a kind the test can't read
        print(f"❌ {name}: the check couldn't read your answer ({error})")
    if coach:
        print(f"🚨 {coach}")


def load_homer_model(fallback_untrained=False):
    """Load the small trained model, read-only (see 05_06). Returns (tokenizer, model).

    Your own `src/homer_ai/homer_model.pt` is used if it exists. Otherwise the
    ready-made `models/odyssey_book1.pt` is used, so every notebook also works
    for a learner who has not trained yet. One line says which one was loaded.

    The tokenizer comes from the vocabulary saved in the file, so its numbers
    match training (`checkpoint.load_checkpoint` also reads format 1 files:
    weights + text, e.g. models/odyssey_book1.pt). If
    neither file is usable, the tokenizer is built from Book 1 and the model is
    None (or, with `fallback_untrained=True`, a new untrained one, so checks that
    do not need training can still run).
    """
    for path in (MODEL_FILE, READY_MADE_MODEL):
        if not path.exists():
            continue
        try:
            model, tokenizer, meta = load_checkpoint(path)
        except (RuntimeError, KeyError) as error:  # saved with other settings
            print(f"{path.name}: cannot be loaded ({type(error).__name__}): skipped")
            continue
        if meta["kind"] != "char-1head":
            print(f"{path.name}: not the small model: skipped")
            continue
        rows = tokenizer.vocab_size
        if path == MODEL_FILE:
            print(f"Using your own model (src/homer_ai/homer_model.pt, {rows} tokens)")
        else:
            print(
                f"Using the ready-made model from models/ ({rows} tokens). "
                "Train your own with: uv run homer-train (about 8 minutes); "
                "the GPT is: uv run homer-train --model gpt (about 3 minutes)"
            )
        return tokenizer, model

    print(
        "no trained model found: make one with `uv run homer-train` (about 8 minutes;"
        " the GPT is `uv run homer-train --model gpt`, about 3 minutes)"
    )
    tokenizer = CharTokenizer(load_books(BOOKS_DIR)[0])
    if not fallback_untrained:
        return tokenizer, None
    torch.manual_seed(0)
    print("using a new, untrained model instead")
    model = HomerGreekLLM(tokenizer.vocab_size, embed_dim=32)
    model.eval()
    return tokenizer, model


def load_homer_gpt():
    """Load the trained GPT of chapter 4, read-only. Returns (tokenizer, model).

    Your own `src/homer_ai/homer_gpt.pt` (from `uv run homer-train --model gpt`)
    comes first, else the ready-made `models/odyssey_book1_gpt.pt`. One line says
    which one was loaded. The sizes come from the file, so nothing is hard-coded.
    """
    for path in (GPT_MODEL_FILE, READY_MADE_GPT):
        if not path.exists():
            continue
        model, tokenizer, meta = load_checkpoint(path)
        if meta["kind"] != "gpt":
            continue
        if path == GPT_MODEL_FILE:
            print("Using your own GPT (src/homer_ai/homer_gpt.pt)")
        else:
            print(
                "Using the ready-made GPT from models/. Train your own with: "
                "uv run homer-train --model gpt (about 3 minutes); "
                "the small model is: uv run homer-train (about 8 minutes)"
            )
        return tokenizer, model
    raise FileNotFoundError(
        "no GPT found: train one with `uv run homer-train --model gpt`"
        " (about 3 minutes)"
    )


def import_mcp_server():
    """Import `homer_mcp_server` so a notebook can look at it (06_03). Never starts it.

    The server picks its model with `find_model_file()`: `$HOMER_MODEL`, else your
    own `homer_gpt.pt`, else the ready-made `models/odyssey_book1_gpt.pt`, else the
    small models in the same order. Importing the file loads nothing and starts
    nothing. Nothing is copied or written.
    """
    from homer_ai import homer_mcp_server

    path = homer_mcp_server.find_model_file()
    if path.parent == READY_MADE_MODEL.parent:
        print(f"Using the ready-made model from models/ ({path.name})")
    else:
        print(f"Using your own model ({path.name})")
    return homer_mcp_server


def load_book_files(folder, pattern="book_*.txt"):
    """Read every file of a folder that matches the pattern (07_01)."""
    return [f.read_text(encoding="utf-8") for f in sorted(folder.glob(pattern))]


def split_90_10(data):
    """The first 90% to train on, the last 10% hidden for checking (05_05)."""
    n = int(0.9 * len(data))
    return data[:n], data[n:]


def evaluate(model, data, chunk_size=CHUNK):
    """Average loss on `data`, measuring only: no learning (05_05)."""
    losses = []
    with torch.no_grad():
        for i in range(0, len(data) - chunk_size, chunk_size):
            inputs = data[i : i + chunk_size]
            targets = data[i + 1 : i + chunk_size + 1]
            losses.append(F.cross_entropy(model(inputs), targets).item())
    return sum(losses) / len(losses)


def fit(train_data, vocab_size, seed, epochs=4):
    """A new model trained with the loop of 05_01; the seed fixes the start (05_04)."""
    torch.manual_seed(seed)
    model = HomerGreekLLM(vocab_size=vocab_size, embed_dim=32)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    for _ in range(epochs):
        for i in range(0, len(train_data) - CHUNK, CHUNK):
            inputs = train_data[i : i + CHUNK]
            targets = train_data[i + 1 : i + CHUNK + 1]
            loss = F.cross_entropy(model(inputs), targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return model


def practice_data():
    """The practice text of chapter 5: the first 5 lines of Book 1 (05_01).

    Returns `(text, tokenizer, data)`. The tokenizer knows only the characters
    of these 5 lines, and `data` is the text as numbers.
    """
    text = "\n".join(load_books(BOOKS_DIR)[0].splitlines()[:5])
    tokenizer = CharTokenizer(text)
    return text, tokenizer, torch.tensor(tokenizer.encode(text))


def tiny_model(seed, epochs, lr=0.01, chunk_size=CHUNK, cosine=False):
    """A new small model trained on the practice text. Saves nothing.

    Returns `(model, tokenizer, losses)`; `losses` is the average loss of each
    epoch. The loop is the one built in 05_01. `cosine=True` adds the
    learning-rate schedule of 05_02: one notch down per epoch, from `lr` to 0.
    """
    _, tokenizer, data = practice_data()
    torch.manual_seed(seed)
    model = HomerGreekLLM(vocab_size=tokenizer.vocab_size, embed_dim=32)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs) if cosine else None
    average_losses = []
    for _epoch in range(epochs):
        losses = []
        for i in range(0, len(data) - chunk_size, chunk_size):
            inputs = data[i : i + chunk_size]
            targets = data[i + 1 : i + chunk_size + 1]
            loss = F.cross_entropy(model(inputs), targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        if scheduler is not None:
            scheduler.step()  # once per epoch
        average_losses.append(sum(losses) / len(losses))
    return model, tokenizer, average_losses


def tiny_train(seed, epochs, lr=0.01, chunk_size=CHUNK, cosine=False):
    """The average loss of each epoch of a tiny training run (05_01, 05_02)."""
    return tiny_model(seed, epochs, lr, chunk_size, cosine)[2]


def write(model, tokenizer, prompt, n_new, temperature=0.8, top_k=5, seed=0):
    """Write `n_new` more letters after `prompt` (06_02).

    Works for the small model and for the GPT: each reads as many letters as it
    can (24 and `context_length`). `top_k=1` is greedy; `top_k=None` keeps every
    letter. A line break is shown as a visible mark, so the text prints on one line.
    """
    torch.manual_seed(seed)
    ids = tokenizer.encode(prompt)
    model.eval()
    with torch.no_grad():
        for _ in range(n_new):
            scores = next_scores(model, ids) / temperature
            if top_k is not None:  # keep only the k best letters
                kth_best = torch.topk(scores, top_k).values[-1]
                scores = scores.masked_fill(scores < kth_best, float("-inf"))
            chances = torch.softmax(scores, dim=-1)
            ids.append(torch.multinomial(chances, num_samples=1).item())
    return tokenizer.decode(ids).replace("\n", "⏎")


def draw_weights(fig, rect, grid, letters, caption):
    """One attention heatmap (03_01).

    Row = the letter that looks, column = the letter it looks at.
    """
    from homer_ai.studio_style import GOOD_BAD, INK, MUTED, text_on

    ax = fig.add_axes(rect)
    ax.set_facecolor("none")
    ax.imshow(grid, cmap=GOOD_BAD, vmin=-1, vmax=1)
    for i in range(len(letters)):
        for j in range(len(letters)):
            ax.text(
                j,
                i,
                f"{grid[i][j]:.2f}",
                ha="center",
                va="center",
                color=text_on(grid[i][j], 1),
                fontsize=11,
                family="monospace",
                fontweight="bold",
            )
    ax.set_xticks(range(len(letters)), letters, fontsize=15, color=INK)
    ax.set_yticks(range(len(letters)), letters, fontsize=15, color=INK)
    ax.xaxis.tick_top()
    ax.tick_params(length=0)
    for side in ax.spines.values():
        side.set_visible(False)
    ax.set_xlabel("the letter it looks AT", color=MUTED)
    ax.xaxis.set_label_position("top")
    ax.set_ylabel("the letter that LOOKS", color=MUTED)
    ax.set_title(caption, color=MUTED, fontsize=10, pad=46)
