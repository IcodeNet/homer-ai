# 0005. CPU-only, laptop-first

Status: Accepted

## Context

The target reader is a developer with no machine-learning background and an ordinary laptop, working in short sessions (README, "Who is this for?"). We can't assume a GPU (a graphics chip that does many sums at once), and a course whose exercises mean an hour of waiting loses its readers.

Numbers from `models/README.md`:

| Model | Weights | Training text | Time on a MacBook CPU |
|---|---|---|---|
| GPT (`gpt.py`) | 835,072 | first 90% of Book 1: 17,730 characters | 2.9 minutes (200 epochs, 3,400 steps) |
| Small one-head model (`model.py`) | 11,840 | all of Book 1: 19,701 characters | about 8 minutes (800 epochs) |

## Decision

Design every part for a laptop CPU, with no GPU code path:

- Train on Odyssey Book 1 only (`BOOKS = ["book_01.txt"]` in `train_homer.py`), about 20,000 characters.
- Keep the GPT small: 4 blocks, 4 heads, 128 numbers per letter, context of 64 letters, dropout 0.3 (`GPT_SETTINGS`).
- Use a character tokenizer ([0001](0001-character-level-tokenizer.md)) so the vocabulary and output layer stay small.
- On Linux, and so in CI, install the small CPU-only PyTorch from `https://download.pytorch.org/whl/cpu` (`[tool.uv.sources]` in `pyproject.toml`), instead of the multi-gigabyte CUDA build. macOS and Windows wheels from PyPI are CPU already.
- Ship ready-made models ([0003](0003-ship-small-models-in-git.md)) so no notebook needs a training run first.

## Consequences

- A reader can train the GPT in about 3 minutes and re-run every notebook. CI runs all notebooks on a standard Ubuntu runner (45 minute limit in `ci.yml`).
- The model is weak by design. With about 18,000 training letters it memorises part of Book 1: held-out loss 2.27 against train loss 1.51, and it does not write real sentences (`models/README.md`). The README says so plainly.
- On so little text, a bigger step size and less dropout make things worse, not better: in a trial recorded in `models/README.md`, learning rate 0.003 with dropout 0.1 reached train loss 0.17 but held-out loss 3.8.
- Scaling up has a limit: a much larger model would need GPU support, `device` handling in `train_homer.py`, and a different torch install. None of that exists today.
- Timings are for one MacBook CPU. Other machines will differ.
