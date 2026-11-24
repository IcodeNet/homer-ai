# 0001. A character-level tokenizer instead of BPE for Homeric Greek

Status: Accepted

## Context

A model reads numbers, not letters, so text must first be cut into pieces (tokens) and each piece given a number. The two common choices are one token per character, and BPE (byte pair encoding), which merges frequent character pairs into longer tokens.

The course teaches readers who have never trained a model. It uses a small amount of text (Odyssey Book 1, 19,701 characters, per `models/README.md`) and has to run on a laptop CPU.

A ready-made BPE exists: GPT-2's, available through `tiktoken`. Its merges were learned from mostly English text. Notebook 07_05 (`notebooks/07_advanced_iliad_and_combining/07_05_english_and_greek.ipynb`) measures it on the first 1,000 characters of Book 1 in each language:

| Text | GPT-2 tokens | Characters per token |
|---|---|---|
| English, 1,000 characters | 227 | 4.41 |
| Greek, 1,000 characters | 1,284 | 0.78 |

So for Greek, GPT-2 needs more tokens than there are letters. It often falls back to pieces of single letters. (The figures are written in `scripts/course/ch07_iliad.py`, in the 07_05 cell "What you saw".)

## Decision

Use `CharTokenizer` (`src/homer_ai/tokenizer.py`): every distinct character of the training text is one token, in sorted order, plus two special tokens at the end, `<|unk|>` and `<|endoftext|>`. For Book 1 this gives a vocabulary of 137 tokens (`models/README.md`). The special tokens go last so no existing ID changes.

The course still teaches BPE (notebook 02_03 builds a small one, and A_05 covers `tiktoken`), so readers understand what we chose not to use.

## Consequences

- The tokenizer is about 50 lines, with nothing hidden. A reader can encode and decode a word by hand (`ἄνδρα` becomes five numbers and back).
- The output layer is small: 137 scores per step. A random guess costs `ln 137 = 4.92` (README table), which makes the loss numbers easy to read.
- Any character not in the vocabulary becomes `<|unk|>`. The MCP server relies on this for seeds with unseen letters.
- Sequences are long: one token per letter, so the model's context of 64 letters (GPT, `models/README.md`) covers only a short stretch of verse. The model learns spelling patterns, not words, and does not write real sentences.
- A model trained this way is not comparable token for token with models that use BPE. Loss is per character here.
- A BPE trained on Greek would give shorter sequences. The course lists it as a next step (the closing cell of 07_05). It would change the vocabulary, the checkpoint `chars` field, and every loss number in the README, so it would need a new checkpoint kind or format.
