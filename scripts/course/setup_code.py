# ruff: noqa: E501
"""Code-cell snippets that more than one notebook of a chapter shares."""

# =====================================================================
# 01_foundations/01_04_our_model_as_a_black_box.ipynb
# =====================================================================
BLACK_BOX_SETUP = r"""import torch

from homer_ai.lessons import load_homer_model
from homer_ai.model import HomerGreekLLM

# read-only: your homer_model.pt, or else the ready-made one
tokenizer, trained = load_homer_model()

torch.manual_seed(42)
untrained = HomerGreekLLM(tokenizer.vocab_size, embed_dim=32)
untrained.eval()  # eval(): "just guessing" mode, not training"""

GENERATE = r'''def write(model, tokenizer, prompt, n_new):
    """Guess the next character, add it, repeat: always the most likely one."""
    ids = tokenizer.encode(prompt)
    with torch.no_grad():  # just guessing, not learning
        for _ in range(n_new):
            scores = model(torch.tensor(ids[-24:], dtype=torch.long))
            best = torch.argmax(scores[-1])  # argmax: where the highest score is
            ids.append(best.item())
    return tokenizer.decode(ids)


print("untrained:", write(untrained, tokenizer, "ἄνδρα", 40))
if trained is not None:
    print("trained:  ", write(trained, tokenizer, "ἄνδρα", 40))'''

# =====================================================================
# Shared code for chapter 3 (load the real model read-only; draw attention heatmaps)
# =====================================================================
CH3_LOAD = r"""import torch

from homer_ai.lessons import load_homer_model

# read only: we load the saved model, we never save
tokenizer, model = load_homer_model(fallback_untrained=True)"""

CH3_HEATMAP = r"""import matplotlib.pyplot as plt

from homer_ai.lessons import draw_weights  # the heatmap function, drawn in 03_01
from homer_ai.studio_style import studio_figure"""

# =====================================================================
# Shared code for chapter 4 (the tokenizer and the five rows of ἄνδρα)
# =====================================================================
CH4_LOAD = r"""import torch
from torch import nn

from homer_ai.tokenizer import CharTokenizer
from homer_ai.train_homer import BOOKS_DIR, load_books

tokenizer = CharTokenizer(load_books(BOOKS_DIR)[0])
print("vocabulary size:", tokenizer.vocab_size)"""

CH4_ROWS = r"""torch.manual_seed(0)
embedding = nn.Embedding(tokenizer.vocab_size, 32)  # a new, untrained table
ids = torch.tensor([tokenizer.encode("ἄνδρα")])  # shape (1, 5): 1 text, 5 letters
x = embedding(ids)  # shape (1, 5, 32): 5 letters, 32 numbers each
print("ids:", tuple(ids.shape), " x:", tuple(x.shape))"""

# =====================================================================
# Shared demo code for chapter 5 (a tiny training run on 5 lines)
# =====================================================================
TINY_SETUP = r"""import torch
import torch.nn.functional as F

from homer_ai.model import HomerGreekLLM
from homer_ai.tokenizer import CharTokenizer
from homer_ai.train_homer import BOOKS_DIR, load_books

# A short practice text: the first 5 lines of Book 1
practice_text = "\n".join(load_books(BOOKS_DIR)[0].splitlines()[:5])
tokenizer = CharTokenizer(practice_text)
data = torch.tensor(tokenizer.encode(practice_text))


def tiny_train(seed, epochs, lr=0.01, chunk_size=24):
    torch.manual_seed(seed)
    model = HomerGreekLLM(vocab_size=tokenizer.vocab_size, embed_dim=32)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
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
        average_losses.append(sum(losses) / len(losses))
    return average_losses"""

# =====================================================================
# Chapter 6: Making it talk (06_00 .. 06_03)
# =====================================================================
CH6_SETUP = r"""import torch

from homer_ai.lessons import load_homer_model

# read-only: we only load the saved model, we never save
tokenizer, model = load_homer_model()  # yours, or else the ready-made one"""

CH6_WINDOW = CH6_SETUP.replace(
    "import load_homer_model", "import CHUNK, load_homer_model"
)  # CHUNK = 24: the model reads at most this many letters (02_04)

# =====================================================================
# Chapter 7: Advanced: the Iliad, and combining both (07_00 .. 07_05)
# =====================================================================
CH7_FIND = r"""# data/homer: load_book_files reads all the books of a folder
from homer_ai.lessons import HOMER, load_book_files"""

CH7_TOOLS = r"""import torch

from homer_ai.lessons import HOMER, evaluate, fit, load_book_files, split_90_10
from homer_ai.tokenizer import CharTokenizer

SLICE = 10000  # characters of Book 1 we use: small, so every cell runs in seconds

odyssey = load_book_files(HOMER / "odyssey" / "greek")  # 24 books, one text each
iliad = load_book_files(HOMER / "iliad" / "greek")"""

# =====================================================================
# Chapter 8: Advanced: fine-tuning a pretrained model (08_00 .. 08_04)
# =====================================================================
CH8_GPT2 = r"""import tiktoken
import torch

from homer_ai.pretrained import load_gpt2

encoding = tiktoken.get_encoding("gpt2")
model = load_gpt2()  # GPT-2 small inside our own GPTModel (08_01); read from the cache"""

CH8_SPAM = r"""import tiktoken

from homer_ai.finetune import balance, encode_messages, load_sms_spam, split

# the data of 08_02, prepared the same way: balanced, split, padded to 64 tokens
encoding = tiktoken.get_encoding("gpt2")
train, val, test = split(balance(load_sms_spam(), seed=123))
train_ids, train_labels = encode_messages(train, encoding, 64)
val_ids, val_labels = encode_messages(val, encoding, 64)
test_ids, test_labels = encode_messages(test, encoding, 64)
print(len(train), "training,", len(val), "validation,", len(test), "test messages")"""
