import pytest

from homer_ai.lessons import READY_MADE_MODEL
from homer_ai.model import HomerGreekLLM
from homer_ai.tokenizer import CharTokenizer
from homer_ai.train_homer import BOOKS_DIR, load_books


@pytest.fixture(autouse=True)
def use_ready_made_model(monkeypatch):
    """Every test uses models/odyssey_book1.pt, never your own homer_model.pt."""
    monkeypatch.setenv("HOMER_MODEL", str(READY_MADE_MODEL))


@pytest.fixture(scope="session")
def book1():
    return load_books(BOOKS_DIR)[0]


@pytest.fixture(scope="session")
def tokenizer(book1):
    return CharTokenizer(book1)


@pytest.fixture
def tiny_model():
    import torch

    torch.manual_seed(0)
    return HomerGreekLLM(vocab_size=20, embed_dim=8).eval()
