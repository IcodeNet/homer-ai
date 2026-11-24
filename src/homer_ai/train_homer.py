import argparse
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR

from homer_ai.checkpoint import save_checkpoint as write_checkpoint
from homer_ai.gpt import GPTConfig, GPTModel
from homer_ai.model import HomerGreekLLM
from homer_ai.tokenizer import CharTokenizer

# Files live next to this script, wherever you run it from
HERE = Path(__file__).parent
# The Greek text: one file per book in data/homer/<poem>/greek/ (book_01.txt, ...).
# To train on the Iliad instead, point this at data/homer/iliad/greek.
BOOKS_DIR = HERE.parents[1] / "data" / "homer" / "odyssey" / "greek"
# Which books to train on. Book 1 only, so training takes about 8 minutes.
# Widen it (e.g. ["book_01.txt", "book_02.txt"]) to train on more text.
BOOKS = ["book_01.txt"]
MODEL_FILE = HERE / "homer_model.pt"  # the small one-head model (--model char)
GPT_MODEL_FILE = HERE / "homer_gpt.pt"  # the GPT of chapter 4 (--model gpt)

# The GPT's settings: small enough to train on a laptop CPU in a few minutes.
GPT_SETTINGS = {
    "context_length": 64,  # letters the model reads at once
    "emb_dim": 128,  # numbers per letter
    "n_heads": 4,
    "n_layers": 4,
    # Book 1 is only 20,000 letters, so a bigger model memorises it quickly:
    # a high dropout keeps the held-out loss falling instead of turning upwards.
    "drop_rate": 0.3,
}
GPT_TRAINING = {
    "seed": 42,
    "epochs": 200,  # an epoch here = about one pass over the training text
    "lr": 0.001,  # peak learning rate; a cosine schedule slides it down to 0
    "batch_size": 16,  # windows of text per step
    "eval_every": 20,  # print the held-out loss every N epochs
}


def load_books(folder, books=BOOKS):
    """Read the chosen book files from the folder, in order: one text per book."""
    return [(folder / name).read_text(encoding="utf-8") for name in books]


def load_data(folder):
    """Read the Greek text and turn every character into a number."""
    books = load_books(folder)
    text = "\n".join(books)  # all books as one text, to build the vocabulary

    tokenizer = CharTokenizer(text)

    # Encode each book on its own, with <|endoftext|> between books
    ids = []
    for book in books:
        if ids:  # not the first book: mark where the previous one ended
            ids.append(tokenizer.eot_id)
        ids.extend(tokenizer.encode(book))

    data = torch.tensor(ids, dtype=torch.long)
    return data, tokenizer, text


def train(model, optimizer, data, chunk_size=24, epochs=800):
    """Train on chunk_size-character windows of Homeric Greek."""
    # Big steps early, small steps late: slide lr smoothly down to ~0
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    for epoch in range(epochs):
        total_loss = 0
        num_chunks = 0
        for i in range(0, len(data) - chunk_size, chunk_size):
            inputs = data[i : i + chunk_size]
            targets = data[i + 1 : i + chunk_size + 1]

            logits = model(inputs)
            loss = F.cross_entropy(logits, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            num_chunks += 1

        scheduler.step()  # once per epoch: move one notch along the schedule

        if (epoch + 1) % 50 == 0:
            lr = scheduler.get_last_lr()[0]
            # Average per chunk, so runs on different amounts of text compare fairly
            avg_loss = total_loss / num_chunks
            print(
                f"Epoch {epoch + 1}/{epochs} | Avg loss: {avg_loss:.4f} | lr: {lr:.5f}"
            )


def save_checkpoint(model, text, path, train_meta=None):
    """Save the trained brain + the vocabulary (checkpoint format 2)."""
    tokenizer = CharTokenizer(text)
    config = {
        "vocab_size": tokenizer.vocab_size,
        "embed_dim": model.embedding.embedding_dim,
    }
    write_checkpoint(path, model, tokenizer, "char-1head", config, train_meta or {})


def split_ids(data, fraction=0.9):
    """The first 90% of the numbers to train on, the last 10% to check on."""
    cut = int(fraction * len(data))
    return data[:cut], data[cut:]


def random_batch(data, context_length, batch_size):
    """`batch_size` random windows: inputs and the same windows shifted by one."""
    starts = torch.randint(0, len(data) - context_length - 1, (batch_size,))
    inputs = torch.stack([data[i : i + context_length] for i in starts])
    targets = torch.stack([data[i + 1 : i + context_length + 1] for i in starts])
    return inputs, targets


def evaluate_gpt(model, data, batch_size=64):
    """Average loss of the GPT on `data`, measuring only: no learning, no dropout.

    The text is cut into back-to-back windows of `context_length` letters.
    """
    was_training = model.training
    model.eval()
    size = model.context_length
    starts = range(0, len(data) - size, size)
    total, count = 0.0, 0
    with torch.no_grad():
        for first in range(0, len(starts), batch_size):
            batch = starts[first : first + batch_size]
            inputs = torch.stack([data[i : i + size] for i in batch])
            targets = torch.stack([data[i + 1 : i + size + 1] for i in batch])
            logits = model(inputs)
            loss = F.cross_entropy(
                logits.flatten(0, 1), targets.flatten(), reduction="sum"
            )
            total += loss.item()
            count += targets.numel()
    model.train(was_training)
    return total / count


def train_gpt(model, train_data, val_data, seed, epochs, lr, batch_size, eval_every):
    """Train the GPT on random windows of `train_data`, checking on `val_data`.

    Returns a dict with the final losses (both measured without dropout).
    """
    torch.manual_seed(seed)
    size = model.context_length
    steps_per_epoch = max(1, len(train_data) // (size * batch_size))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    # one notch of the cosine schedule per step: big steps early, tiny ones late
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs * steps_per_epoch)

    model.train()  # dropout on while learning
    for epoch in range(epochs):
        total = 0.0
        for _ in range(steps_per_epoch):
            inputs, targets = random_batch(train_data, size, batch_size)
            logits = model(inputs)
            loss = F.cross_entropy(logits.flatten(0, 1), targets.flatten())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            total += loss.item()

        if (epoch + 1) % eval_every == 0:
            val = evaluate_gpt(model, val_data)
            print(
                f"Epoch {epoch + 1}/{epochs}"
                f" | train loss: {total / steps_per_epoch:.4f}"
                f" | held-out loss: {val:.4f} | lr: {scheduler.get_last_lr()[0]:.5f}"
            )

    return {
        "final_train_loss": evaluate_gpt(model, train_data),
        "val_loss": evaluate_gpt(model, val_data),
    }


def main_char():
    """The small one-head model (chapter 1): 24-letter chunks, all of Book 1."""
    torch.manual_seed(42)

    data, tokenizer, text = load_data(BOOKS_DIR)
    model = HomerGreekLLM(vocab_size=tokenizer.vocab_size, embed_dim=32)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)

    vocab = tokenizer.vocab_size
    print(f"Training on {len(data)} Greek characters (Vocab size: {vocab})...")
    train(model, optimizer, data)

    meta = {
        "seed": 42,
        "epochs": 800,
        "lr": 0.01,
        "chunk": 24,
        "train_chars": len(data),
        "val_chars": 0,  # trained on all of Book 1: no held-out text
        "books": BOOKS,
        "poem": "odyssey",
    }
    save_checkpoint(model, text, MODEL_FILE, meta)
    print(f"Saved trained model to {MODEL_FILE}")


def main_gpt():
    """The GPT of chapter 4: the first 90% of Book 1 to learn, the last 10% to check."""
    settings = dict(GPT_TRAINING)
    data, tokenizer, _ = load_data(BOOKS_DIR)
    train_data, val_data = split_ids(data)

    torch.manual_seed(settings["seed"])
    config = GPTConfig(vocab_size=tokenizer.vocab_size, **GPT_SETTINGS)
    model = GPTModel(config)
    n_weights = sum(p.numel() for p in model.parameters())
    print(
        f"Training a GPT ({n_weights:,} weights) on {len(train_data)} Greek characters,"
        f" checking on the last {len(val_data)} (Vocab size: {tokenizer.vocab_size})..."
    )
    started = time.perf_counter()
    losses = train_gpt(model, train_data, val_data, **settings)
    seconds = time.perf_counter() - started
    print(
        f"Done in {seconds / 60:.1f} minutes"
        f" | train loss {losses['final_train_loss']:.4f}"
        f" | held-out loss {losses['val_loss']:.4f}"
    )

    meta = {
        **settings,
        "chunk": config.context_length,
        "train_chars": len(train_data),
        "val_chars": len(val_data),
        **losses,
        "train_seconds": round(seconds),
        "books": BOOKS,
        "poem": "odyssey",
    }
    write_checkpoint(GPT_MODEL_FILE, model, tokenizer, "gpt", config.to_dict(), meta)
    print(f"Saved trained model to {GPT_MODEL_FILE}")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Train a Homer model on Book 1.")
    parser.add_argument(
        "--model",
        choices=["char", "gpt"],
        default="char",
        help="char: the small one-head model (default, about 8 minutes, "
        "writes homer_model.pt); gpt: the GPT of chapter 4 "
        "(writes homer_gpt.pt)",
    )
    args = parser.parse_args(argv)
    main_gpt() if args.model == "gpt" else main_char()


if __name__ == "__main__":
    main()
