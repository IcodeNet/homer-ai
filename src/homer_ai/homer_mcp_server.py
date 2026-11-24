"""An MCP server that offers Homer to apps like Claude as two tools.

`consult_homer_llm` lets our small model continue a Greek text. The model is the
GPT of chapter 4 if its file is there, else the small one-head model (see
DEFAULT_MODEL_FILES for the order, or set $HOMER_MODEL).

`search_homer` finds real passages of the Odyssey and the Iliad for a question
(chapter 9), with their book and line, so the app can answer from the text.

Start it with `uv run homer-serve`. It talks over stdin/stdout, so everything
we want to *print* (logs, errors) goes to stderr.
"""

import os
import sys
import time
from pathlib import Path
from typing import Literal, NamedTuple

import torch
from mcp.server.mcpserver import MCPServer

from homer_ai.checkpoint import load_checkpoint, training_text
from homer_ai.generation import next_scores
from homer_ai.retrieval import search
from homer_ai.tokenizer import CharTokenizer

HERE = Path(__file__).parent
MODELS = HERE.parents[1] / "models"  # the ready-made models that ship with the repo
# The order we look in (first file that exists wins), after $HOMER_MODEL:
# the GPT first (yours, then the ready-made one), then the small model.
DEFAULT_MODEL_FILES = (
    HERE / "homer_gpt.pt",  # the GPT you trained yourself
    MODELS / "odyssey_book1_gpt.pt",  # the ready-made GPT
    HERE / "homer_model.pt",  # the small model you trained yourself
    MODELS / "odyssey_book1.pt",  # the ready-made small model
)

MAX_MATCHING_LINES = 10
MIN_CHARS, MAX_CHARS = 1, 200  # how many new characters one call may ask for
MIN_K, MAX_K = 1, 10  # how many passages one search may ask for
MAX_PASSAGE_CHARS = 1200  # a longer passage is cut here, and marked with …

mcp = MCPServer("HomerOdysseyExpert")


class Loaded(NamedTuple):
    path: Path
    tokenizer: CharTokenizer
    model: torch.nn.Module  # the small model or the GPT; see `kind`
    text: str  # the training text (Odyssey Book 1)
    kind: str  # "gpt" or "char-1head"


def find_model_file():
    """Pick the model file: $HOMER_MODEL, else the first of DEFAULT_MODEL_FILES."""
    from_env = os.environ.get("HOMER_MODEL")
    if from_env:
        if not Path(from_env).is_file():
            raise FileNotFoundError(f"HOMER_MODEL points to a missing file: {from_env}")
        return Path(from_env)
    for path in DEFAULT_MODEL_FILES:
        if path.is_file():
            return path
    raise FileNotFoundError(
        "No model file found. Train one with `uv run homer-train --model gpt` "
        "(a few minutes), or set HOMER_MODEL=/path/to/model.pt."
    )


def load_model(path):
    """Read a checkpoint (read-only) and rebuild the tokenizer and the model.

    The model's sizes and the vocabulary come from the file itself, so a small
    model, a GPT and an older file all load the same way.
    """
    model, tokenizer, meta = load_checkpoint(path)
    return Loaded(Path(path), tokenizer, model, training_text(meta), meta["kind"])


_loaded = None


def load():
    """Load the model once; later calls reuse it. Importing this file loads nothing."""
    global _loaded
    if _loaded is None:
        _loaded = load_model(find_model_file())
    return _loaded


def find_lines(text, seed, limit=MAX_MATCHING_LINES):
    """Return up to `limit` lines of the text that contain the seed (ignoring case)."""
    seed = seed.strip().casefold()
    if not seed:
        return []
    matches = [line for line in text.splitlines() if seed in line.casefold()]
    return matches[:limit]


def clamp_chars(n):
    """Keep the number of new characters between MIN_CHARS and MAX_CHARS."""
    return max(MIN_CHARS, min(MAX_CHARS, n))


@mcp.tool()
def consult_homer_llm(greek_seed: str, chars_to_generate: int = 30) -> str:
    """
    Continue a Greek text with a tiny character-level model trained only on
    Homer's Odyssey, Book 1: a small GPT (under a million weights) if one
    is available, else an even smaller one-head model. The continuation looks like
    Greek but is often not real Greek; this tool cannot answer questions or
    explain grammar. It also returns up to 10 real lines of Book 1 that contain
    the seed.

    greek_seed: a Greek word or phrase to continue.
    chars_to_generate: how many characters to add, from 1 to 200 (default 30);
    values outside that range are clamped.
    """
    started = time.perf_counter()
    loaded = load()
    tokenizer, model, text = loaded.tokenizer, loaded.model, loaded.text
    chars_to_generate = clamp_chars(chars_to_generate)

    # Characters the model has never seen become <|unk|>
    ids = tokenizer.encode(greek_seed)
    if not ids:  # an empty seed: start from the first letter of the text
        ids = tokenizer.encode(text[0])

    generated = list(ids)
    with torch.no_grad():
        for _ in range(chars_to_generate):
            # next_scores reads only as many recent letters as the model can
            # (its context length), and works for both kinds of model
            scores = next_scores(model, generated)
            generated.append(torch.argmax(scores).item())

    model_output = tokenizer.decode(generated)

    lines = find_lines(text, greek_seed)
    reference = "\n".join(lines) if lines else "(no line contains this seed)"

    # stderr, never stdout: stdout is the channel to the MCP client
    print(
        f"homer: seed_chars={len(greek_seed)} chars={chars_to_generate} "
        f"ms={(time.perf_counter() - started) * 1000:.0f} model={loaded.path.name} "
        f"kind={loaded.kind}",
        file=sys.stderr,
    )
    return (
        f"Custom PyTorch HomerLLM ({loaded.kind}) Continuation:\n{model_output}\n\n"
        f"Lines containing the seed (up to {MAX_MATCHING_LINES}):\n{reference}"
    )


def clamp_k(k):
    """Keep the number of passages between MIN_K and MAX_K."""
    return max(MIN_K, min(MAX_K, k))


@mcp.tool()
def search_homer(
    question: str,
    poem: Literal["odyssey", "iliad", "both"] = "both",
    language: Literal["english", "greek"] = "english",
    k: int = 5,
) -> str:
    """
    Find real passages of Homer's Odyssey and Iliad that match a question, each
    with its reference "poem book:line", so you can answer from the text and
    cite it. The English is Samuel Butler's prose translation, which uses Roman
    names (Ulysses, Jove, Minerva); Greek names in the question are matched to
    them. The line numbers are the lines of this project's files (in English one
    line is one paragraph), not the verse numbers of printed editions.

    The search matches words (BM25), not meaning: it finds passages that share
    rare words with the question, and misses a passage that says the same thing
    in other words. If nothing fits, ask again with words the text itself would
    use. The passages are not an answer: read them, and say so if they do not
    hold one.

    question: what to look for, in English (or Greek words for language="greek";
    accents may be left out).
    poem: "odyssey", "iliad" or "both" (default).
    language: "english" (default) or "greek" (the original, 5 verses a passage).
    k: how many passages to return, from 1 to 10 (default 5); values outside
    that range are clamped.
    """
    started = time.perf_counter()
    k = clamp_k(k)
    found = search(question, poem, language, k)  # builds the index on first use

    # stderr, never stdout: stdout is the channel to the MCP client
    print(
        f"homer: search question_chars={len(question)} poem={poem} "
        f"language={language} k={k} found={len(found)} "
        f"ms={(time.perf_counter() - started) * 1000:.0f}",
        file=sys.stderr,
    )
    if not found:
        return "No passage shares a word with this question."
    parts = []
    for passage, score in found:
        text = passage.text
        if len(text) > MAX_PASSAGE_CHARS:
            text = text[:MAX_PASSAGE_CHARS] + "…"
        parts.append(f"[{passage.ref}] (score {score:.1f})\n{text}")
    return "\n\n".join(parts)


def main():
    """Entry point of `homer-serve`: load the model, then serve over stdio.

    The search index is not built here: it is built by the first search (well
    under a second), so the server starts as fast as it does with one tool.
    """
    try:
        loaded = load()
    except FileNotFoundError as error:
        sys.exit(f"homer-serve: {error}")
    print(f"homer: loaded {loaded.path} ({loaded.kind})", file=sys.stderr)
    mcp.run()


if __name__ == "__main__":
    main()
