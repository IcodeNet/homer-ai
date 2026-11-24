"""Measure three numbers for the README table "What the model achieves".

Run: uv run python scripts/baselines.py

Loss is the average cross-entropy in nats (lower is better), the same number
the training loop prints. Nothing is trained and no file is written.
Measures the shipped files only, never your own homer_model.pt or homer_gpt.pt.

  1. Random guess:       ln(vocab size).
  2. Bigram baseline:    count how often each character follows each other one
                         (add-one smoothing), then score the next-character guesses.
                         Reported in-sample (fit and scored on all of Book 1) and
                         held-out (fit on the first 90%, scored on the last 10%).
  3. The small model:    models/odyssey_book1.pt, scored with lessons.evaluate on
                         all of Book 1. It was trained on 100% of Book 1, so this
                         is an in-sample number.
  4. The GPT:            models/odyssey_book1_gpt.pt, scored on the last 10% of
                         Book 1 (held-out: it trained on the first 90% only) and,
                         for comparison, on the first 90%.
"""

import math

import torch

from homer_ai.checkpoint import load_checkpoint
from homer_ai.lessons import READY_MADE_GPT, READY_MADE_MODEL, evaluate
from homer_ai.tokenizer import CharTokenizer
from homer_ai.train_homer import BOOKS_DIR, evaluate_gpt, load_books


def bigram_loss(train_ids, test_ids, vocab_size):
    """Average -ln(p(next | current)) on test_ids, counts taken from train_ids."""
    counts = [[1] * vocab_size for _ in range(vocab_size)]  # add-one smoothing
    for current, following in zip(train_ids, train_ids[1:], strict=False):
        counts[current][following] += 1
    total = 0.0
    for current, following in zip(test_ids, test_ids[1:], strict=False):
        total -= math.log(counts[current][following] / sum(counts[current]))
    return total / (len(test_ids) - 1)


def main():
    text = load_books(BOOKS_DIR)[0]
    tokenizer = CharTokenizer(text)
    ids = tokenizer.encode(text)
    vocab = tokenizer.vocab_size
    cut = int(0.9 * len(ids))  # the same 90/10 split as lessons.split_90_10

    random_guess = math.log(vocab)
    bigram_in = bigram_loss(ids, ids, vocab)
    bigram_out = bigram_loss(ids[:cut], ids[cut:], vocab)

    # always the shipped file, never your own homer_model.pt
    model, _, _ = load_checkpoint(READY_MADE_MODEL)
    model_in = evaluate(model, torch.tensor(ids))  # lessons.evaluate; no learning
    gpt, _, meta = load_checkpoint(READY_MADE_GPT)
    data = torch.tensor(ids)
    gpt_train = evaluate_gpt(gpt, data[:cut])
    gpt_val = evaluate_gpt(gpt, data[cut:])

    print(f"Book 1: {len(ids)} characters, vocabulary {vocab}")
    print(f"random guess, ln({vocab})                      {random_guess:.2f}")
    print(f"bigram, in-sample (all of Book 1)            {bigram_in:.2f}")
    print(f"bigram, held-out (fit 90%, test last 10%)    {bigram_out:.2f}")
    print(f"models/odyssey_book1.pt, in-sample           {model_in:.2f}")
    print(
        f"models/odyssey_book1_gpt.pt, first 90%       {gpt_train:.2f}  (trained on it)"
    )
    print(f"models/odyssey_book1_gpt.pt, last 10%        {gpt_val:.2f}  (held-out)")
    assert (
        abs(gpt_val - meta["train_meta"]["val_loss"]) < 1e-4
    )  # same as when it trained


if __name__ == "__main__":
    main()
