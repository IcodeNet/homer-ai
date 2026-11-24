"""Save and load a trained model together with everything needed to use it.

Checkpoint format 2 is one file with:

    format       2
    kind         "char-1head" (model.py) or "gpt" (gpt.py)
    config       the sizes needed to rebuild the model (a dict)
    chars        the vocabulary in order, special tokens included
    model_state  the weights
    train_meta   how it was trained: seed, epochs, lr, losses, books, versions

Format 1 (weights + text, e.g. `models/odyssey_book1.pt`) has only
`model_state` and `rhapsody_text`; `load_checkpoint` reads it too.
Files are read with `weights_only=True`, so loading cannot run code.
"""

import subprocess
from pathlib import Path

import torch

from homer_ai.gpt import GPTConfig, GPTModel
from homer_ai.model import HomerGreekLLM
from homer_ai.tokenizer import CharTokenizer

GREEK_DIR = Path(__file__).parents[2] / "data" / "homer"  # <poem>/greek/book_NN.txt
SMALL_EMBED_DIM = 32  # what format-1 files were always trained with


def git_sha():
    """The short git commit of the code, or None if git is not available."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).parent,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None if out.returncode == 0 else None


def save_checkpoint(path, model, tokenizer, kind, config, train_meta):
    """Write format 2. `config` is a dict that `build_model` understands."""
    notes = {
        "git_sha": git_sha(),
        "torch_version": str(torch.__version__),  # plain text: loads with weights_only
        **train_meta,
    }
    torch.save(
        {
            "format": 2,
            "kind": kind,
            "config": dict(config),
            "chars": list(tokenizer.chars),
            "model_state": model.state_dict(),
            "train_meta": notes,
        },
        path,
    )


def build_model(kind, config):
    """A new, untrained model of this kind and config."""
    if kind == "char-1head":
        return HomerGreekLLM(config["vocab_size"], embed_dim=config["embed_dim"])
    if kind == "gpt":
        return GPTModel(GPTConfig(**config))
    raise ValueError(f"unknown model kind: {kind!r}")


def load_checkpoint(path):
    """Read a checkpoint (read-only). Returns `(model, tokenizer, meta)`.

    The model is in `eval()` mode. `meta` is a dict with at least `format`, `kind`,
    `config` and `train_meta`; for format 1 the saved training text is `meta["text"]`.
    """
    checkpoint = torch.load(path, weights_only=True)
    if "format" in checkpoint:
        kind, config = checkpoint["kind"], checkpoint["config"]
        tokenizer = CharTokenizer.from_chars(checkpoint["chars"])
        meta = {
            "format": checkpoint["format"],
            "kind": kind,
            "config": config,
            "train_meta": checkpoint.get("train_meta", {}),
        }
    else:  # format 1: only weights and the training text
        text = checkpoint["rhapsody_text"]
        tokenizer = CharTokenizer(text)
        kind = "char-1head"
        config = {"vocab_size": tokenizer.vocab_size, "embed_dim": SMALL_EMBED_DIM}
        meta = {
            "format": 1,
            "kind": kind,
            "config": config,
            "train_meta": {},
            "text": text,
        }
    model = build_model(kind, config)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, tokenizer, meta


def training_text(meta):
    """The text the model was trained on (for showing real lines, 06_03).

    Format 1 saved it inside the file. Format 2 saves only the names of the
    books, so they are read again from `data/homer/`.
    """
    if "text" in meta:
        return meta["text"]
    train_meta = meta["train_meta"]
    folder = GREEK_DIR / train_meta.get("poem", "odyssey") / "greek"
    books = train_meta.get("books", ["book_01.txt"])
    return "\n".join((folder / name).read_text(encoding="utf-8") for name in books)
