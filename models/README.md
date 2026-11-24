# Ready-made models

These are trained models that ship with the course, so every notebook works before you've trained anything yourself. They're shared under CC BY-SA 4.0, like the Greek text they learned from ([LICENSE.md](LICENSE.md)).

| File | What it is | Size |
|---|---|---|
| `odyssey_book1_gpt.pt` | **the GPT of chapter 4** (`src/homer_ai/gpt.py`), trained on the first 90% of Book 1 of the Odyssey (`data/homer/odyssey/greek/book_01.txt`); checkpoint format 2 | 3.4 MB |
| `odyssey_book1.pt` | the small one-head model of chapter 1 (`src/homer_ai/model.py`), trained on all of Book 1: 19,701 characters, vocabulary of 137 tokens, 11,840 weights; checkpoint format 1 | 88 KB |

## The GPT: what is it, and how was it made?

Nothing special: we ran `uv run homer-train --model gpt` once, then copied `src/homer_ai/homer_gpt.pt` to `models/odyssey_book1_gpt.pt`.

| | |
|---|---|
| Model | `GPTModel`: 4 blocks, 4 heads, 128 numbers per letter, context of 64 letters, dropout 0.3, no bias in query/key/value |
| Weights | 835,072 |
| Vocabulary | 137 tokens: the characters of Book 1, `<|unk|>`, `<|endoftext|>` |
| Training text | the first 90% of Book 1 (17,730 characters); the last 10% (1,971 characters) is held out |
| Settings | seed 42, AdamW (weight decay 0.01), peak learning rate 0.001 with a cosine schedule down to 0 (one notch per step), 16 windows of 64 letters per step, **200 epochs** (an epoch is about one pass over the training text: 17 steps, so 3,400 steps in all) |
| Time | **2.9 minutes** on a MacBook CPU (no GPU) |
| Train loss (no dropout) | **1.51** |
| Held-out loss (last 10%) | **2.27** (perplexity about 9.7) |
| For comparison | random guess 4.92; bigram counts, held-out 2.99 (`uv run python scripts/baselines.py`) |

So, what is this really? The held-out loss is a fair number, and it clearly beats the bigram. But the gap to the train loss shows the model memorises part of Book 1 (it only has about 18,000 letters to learn from), and it doesn't write real sentences.
Dropout 0.3 and a short run are what stop the held-out loss from turning back up. In one trial, a run at learning rate 0.003 with dropout 0.1 got the train loss down to 0.17, but the held-out loss went to 3.8.

To reproduce it (this does **not** touch the ready-made files; it writes `src/homer_ai/homer_gpt.pt`):

```
uv run homer-train --model gpt
uv run python scripts/baselines.py     # prints the held-out loss of the ready-made GPT
```

The run is seeded, so you should get the same numbers on the same machine and PyTorch version. Other machines may differ in the last digits.

## The small model

We ran `uv run python src/homer_ai/train_homer.py` with its defaults: seed 42, `embed_dim=32`, AdamW with learning rate 0.01 and a cosine schedule,
chunks of 24 characters, **800 epochs** (about 8 minutes on a laptop). The average loss at the end was about **2.38**, measured on the same Book 1 it trained on.

## What is inside a file?

Each one is a PyTorch checkpoint: a file that saves a model's weights plus what you need to use them again. The GPT file (format 2) holds `format`, `kind`, `config` (the sizes), `chars` (the vocabulary in order), `model_state` (the weights) and `train_meta` (seed, epochs, learning rate, losses, books, PyTorch version, git commit).
The small model's file (format 1) holds only `model_state` and `rhapsody_text` (the training text, from which the tokenizer is rebuilt; a *rhapsody* is one book of a Homeric poem, here Book 1). `homer_ai.checkpoint.load_checkpoint` reads both. See [05_06](../notebooks/05_training_on_the_odyssey/05_06_saving_and_loading.ipynb).

## Which one do the notebooks use?

Yours, if you have one. For the GPT: `src/homer_ai/homer_gpt.pt` if it exists, otherwise `models/odyssey_book1_gpt.pt`. For the small model: `src/homer_ai/homer_model.pt`, otherwise `models/odyssey_book1.pt`.
Each notebook prints one line telling you which one it loaded.

The MCP server (`uv run homer-serve`, i.e. `src/homer_ai/homer_mcp_server.py`) looks in this order and takes the first file that exists:
the file named in the `HOMER_MODEL` environment variable, your own `homer_gpt.pt`, `models/odyssey_book1_gpt.pt`, your own `homer_model.pt`, `models/odyssey_book1.pt`.
If none exists, it stops with a message on how to train.
