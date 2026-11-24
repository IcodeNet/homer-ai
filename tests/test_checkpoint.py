import pytest
import torch

from homer_ai.checkpoint import load_checkpoint, save_checkpoint, training_text
from homer_ai.gpt import GPTConfig, GPTModel
from homer_ai.lessons import READY_MADE_MODEL
from homer_ai.model import HomerGreekLLM
from homer_ai.tokenizer import CharTokenizer

META = {"seed": 1, "epochs": 2, "books": ["book_01.txt"], "poem": "odyssey"}


def test_gpt_round_trip_gives_identical_logits(tmp_path, tokenizer):
    config = GPTConfig(
        vocab_size=tokenizer.vocab_size,
        context_length=16,
        emb_dim=16,
        n_heads=2,
        n_layers=1,
    )
    torch.manual_seed(0)
    model = GPTModel(config).eval()
    path = tmp_path / "gpt.pt"
    save_checkpoint(path, model, tokenizer, "gpt", config.to_dict(), META)

    loaded, loaded_tokenizer, meta = load_checkpoint(path)
    ids = torch.tensor([tokenizer.encode("ἄνδρα μοι")])
    with torch.no_grad():
        assert torch.equal(model(ids), loaded(ids))
    assert isinstance(loaded, GPTModel)
    assert loaded.cfg == config
    assert loaded_tokenizer.chars == tokenizer.chars
    assert (meta["format"], meta["kind"]) == (2, "gpt")
    assert meta["train_meta"]["seed"] == 1
    assert "torch_version" in meta["train_meta"]


def test_small_model_round_trip(tmp_path, tokenizer):
    torch.manual_seed(0)
    model = HomerGreekLLM(tokenizer.vocab_size, embed_dim=32).eval()
    config = {"vocab_size": tokenizer.vocab_size, "embed_dim": 32}
    path = tmp_path / "small.pt"
    save_checkpoint(path, model, tokenizer, "char-1head", config, META)

    loaded, _, meta = load_checkpoint(path)
    ids = torch.tensor(tokenizer.encode("ἄνδρα μοι"))
    with torch.no_grad():
        assert torch.equal(model(ids), loaded(ids))
    assert meta["kind"] == "char-1head"


def test_the_saved_file_loads_with_weights_only(tmp_path, tokenizer):
    model = HomerGreekLLM(tokenizer.vocab_size, embed_dim=32)
    path = tmp_path / "small.pt"
    config = {"vocab_size": tokenizer.vocab_size, "embed_dim": 32}
    save_checkpoint(path, model, tokenizer, "char-1head", config, META)
    saved = torch.load(path, weights_only=True)
    assert set(saved) == {
        "format",
        "kind",
        "config",
        "chars",
        "model_state",
        "train_meta",
    }


def test_format_1_file_loads(tokenizer, book1):
    saved = torch.load(READY_MADE_MODEL, weights_only=True)
    assert "format" not in saved  # really format 1
    model, loaded_tokenizer, meta = load_checkpoint(READY_MADE_MODEL)
    assert meta["format"] == 1
    assert meta["kind"] == "char-1head"
    assert loaded_tokenizer.chars == tokenizer.chars
    assert sum(p.numel() for p in model.parameters()) == 11_840
    assert training_text(meta) == book1
    assert not model.training  # eval mode


def test_tokenizer_from_chars_keeps_every_id(tokenizer):
    rebuilt = CharTokenizer.from_chars(tokenizer.chars)
    word = "ἄνδρα μοι ἔννεπε !"
    assert rebuilt.encode(word) == tokenizer.encode(word)
    assert (rebuilt.unk_id, rebuilt.eot_id) == (135, 136)
    assert rebuilt.vocab_size == 137


def test_unknown_kind_is_refused():
    from homer_ai.checkpoint import build_model

    with pytest.raises(ValueError, match="unknown model kind"):
        build_model("nope", {})
