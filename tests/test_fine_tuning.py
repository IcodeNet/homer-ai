"""Tests of the chapter 8 helpers (notebooks/08_advanced_fine_tuning/*.py).

Nothing here downloads: GPT-2 and the spam data are replaced by tiny stand-ins
written to a temporary folder.
"""

import json
import struct
import zipfile

import pytest
import torch
from torch import nn

from homer_ai import finetune, lora, pretrained
from homer_ai.gpt import GPTConfig, GPTModel

TINY = GPTConfig(
    vocab_size=11, context_length=6, emb_dim=8, n_heads=2, n_layers=2, qkv_bias=True
)


def write_safetensors(path, tensors):
    """A safetensors file made by hand: header length, JSON header, raw bytes."""
    header, data = {"__metadata__": {"format": "pt"}}, b""
    for name, tensor in tensors.items():
        raw = tensor.numpy().tobytes()
        header[name] = {
            "dtype": "F32",
            "shape": list(tensor.shape),
            "data_offsets": [len(data), len(data) + len(raw)],
        }
        data += raw
    text = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(text)) + text + data)
    return path


def gpt2_style_weights(cfg):
    """Random tensors with GPT-2's names and layouts, for a tiny model."""
    torch.manual_seed(0)
    e = cfg.emb_dim
    weights = {
        "wte.weight": torch.randn(cfg.vocab_size, e),
        "wpe.weight": torch.randn(cfg.context_length, e),
        "ln_f.weight": torch.randn(e),
        "ln_f.bias": torch.randn(e),
    }
    for n in range(cfg.n_layers):
        h = f"h.{n}."
        weights[h + "attn.bias"] = torch.ones(
            1, 1, cfg.context_length, cfg.context_length
        )
        weights[h + "attn.c_attn.weight"] = torch.randn(e, 3 * e)
        weights[h + "attn.c_attn.bias"] = torch.randn(3 * e)
        weights[h + "attn.c_proj.weight"] = torch.randn(e, e)
        weights[h + "attn.c_proj.bias"] = torch.randn(e)
        weights[h + "mlp.c_fc.weight"] = torch.randn(e, 4 * e)
        weights[h + "mlp.c_fc.bias"] = torch.randn(4 * e)
        weights[h + "mlp.c_proj.weight"] = torch.randn(4 * e, e)
        weights[h + "mlp.c_proj.bias"] = torch.randn(e)
        for norm in ("ln_1", "ln_2"):
            weights[h + norm + ".weight"] = torch.randn(e)
            weights[h + norm + ".bias"] = torch.randn(e)
    return weights


@pytest.fixture
def tiny_file(tmp_path):
    return write_safetensors(tmp_path / "tiny.safetensors", gpt2_style_weights(TINY))


# ---------------------------------------------------------------- pretrained.py
def test_safetensors_header_and_tensors_round_trip(tiny_file):
    written = gpt2_style_weights(TINY)
    header, start = pretrained.read_safetensors_header(tiny_file)
    assert "__metadata__" not in header
    assert header["wte.weight"]["shape"] == [11, 8]
    assert start > 8
    read = pretrained.read_safetensors(tiny_file)
    assert read.keys() == written.keys()
    for name, tensor in written.items():
        assert torch.equal(read[name], tensor), name


def test_read_safetensors_rejects_a_wrong_size(tmp_path):
    header = {"x": {"dtype": "F32", "shape": [3], "data_offsets": [0, 8]}}
    text = json.dumps(header).encode()
    path = tmp_path / "bad.safetensors"
    path.write_bytes(struct.pack("<Q", len(text)) + text + b"\0" * 8)
    with pytest.raises(ValueError, match="x"):
        pretrained.read_safetensors(path)


def test_gpt2_names_map_onto_our_model(tiny_file):
    theirs = pretrained.read_safetensors(tiny_file)
    model = pretrained.load_gpt2_into(GPTModel(TINY), theirs)
    state = model.state_dict()
    e = TINY.emb_dim
    assert torch.equal(state["token_embedding.weight"], theirs["wte.weight"])
    assert torch.equal(state["output_head.weight"], theirs["wte.weight"])  # tied
    # the fused query/key/value layer is cut in three and transposed
    fused = theirs["h.1.attn.c_attn.weight"]
    assert torch.equal(state["blocks.1.attention.query.weight"], fused[:, :e].T)
    assert torch.equal(state["blocks.1.attention.key.weight"], fused[:, e : 2 * e].T)
    assert torch.equal(state["blocks.1.attention.value.weight"], fused[:, 2 * e :].T)
    assert torch.equal(
        state["blocks.1.attention.value.bias"], theirs["h.1.attn.c_attn.bias"][2 * e :]
    )
    assert torch.equal(
        state["blocks.0.feed_forward.0.weight"], theirs["h.0.mlp.c_fc.weight"].T
    )
    assert model(torch.tensor([[1, 2, 3]])).shape == (1, 3, 11)


def test_load_gpt2_into_rejects_a_model_of_another_size(tiny_file):
    other = GPTConfig(
        vocab_size=11,
        context_length=6,
        emb_dim=16,
        n_heads=2,
        n_layers=2,
        qkv_bias=True,
    )
    with pytest.raises(RuntimeError, match="size mismatch"):
        pretrained.load_gpt2_into(
            GPTModel(other), pretrained.read_safetensors(tiny_file)
        )


def test_load_gpt2_into_rejects_a_model_with_more_layers(tiny_file):
    deeper = GPTConfig(
        vocab_size=11, context_length=6, emb_dim=8, n_heads=2, n_layers=3, qkv_bias=True
    )
    with pytest.raises(ValueError, match="missing"):
        pretrained.load_gpt2_into(
            GPTModel(deeper), pretrained.read_safetensors(tiny_file)
        )


def test_download_uses_the_cached_file_and_never_the_network(tmp_path):
    path = tmp_path / "file.bin"
    path.write_bytes(b"12345")
    url = "http://127.0.0.1:9/never-fetched"
    assert pretrained.download(url, path, expected_bytes=5) == path
    assert pretrained.sha256_of(path) == (
        "5994471abb01112afcc18159f6cc74b4f511b99806da59b3caf5a9c173cacfc5"
    )


def test_gpt2_config_is_gpt2_small():
    cfg = pretrained.GPT2_CONFIG
    assert (cfg.vocab_size, cfg.context_length, cfg.emb_dim) == (50257, 1024, 768)
    assert (cfg.n_heads, cfg.n_layers, cfg.qkv_bias) == (12, 12, True)


# ---------------------------------------------------------------- lora.py
def test_lora_starts_as_the_frozen_layer_and_counts_its_numbers():
    torch.manual_seed(0)
    plain = nn.Linear(6, 4)
    layer = lora.LoRALinear(plain, rank=2, alpha=4)
    x = torch.randn(3, 6)
    assert torch.equal(layer(x), plain(x))  # B is all zeros at the start
    assert lora.count_parameters(layer) == (2 * 6 + 4 * 2, 6 * 4 + 4 + 2 * 6 + 4 * 2)


def test_lora_only_the_small_tables_get_nudged():
    torch.manual_seed(0)
    layer = lora.LoRALinear(nn.Linear(6, 4), rank=2)
    layer(torch.randn(3, 6)).sum().backward()
    assert layer.linear.weight.grad is None
    assert layer.B.grad is not None and layer.A.grad is not None


def test_lora_merged_gives_the_same_answers():
    torch.manual_seed(0)
    layer = lora.LoRALinear(nn.Linear(6, 4), rank=2, alpha=4)
    with torch.no_grad():
        layer.B.normal_()
    x = torch.randn(5, 6)
    assert torch.allclose(layer.merged()(x), layer(x), atol=1e-6)


def test_lora_rank_must_be_positive():
    with pytest.raises(ValueError):
        lora.LoRALinear(nn.Linear(2, 2), rank=0)


def test_apply_lora_wraps_every_linear_but_the_skipped_ones():
    model = finetune.make_classifier(GPTModel(TINY))
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in model.output_head.parameters():
        parameter.requires_grad = True
    wrapped = lora.apply_lora(model, rank=2, skip=("output_head",))
    assert wrapped == TINY.n_layers * 6  # query, key, value, out_proj, 2 feed-forward
    assert isinstance(model.output_head, nn.Linear)
    e, rank = TINY.emb_dim, 2
    per_block = 4 * (2 * rank * e) + 2 * (rank * e + rank * 4 * e)
    head = e * 2 + 2
    assert lora.count_parameters(model)[0] == TINY.n_layers * per_block + head
    assert lora.apply_lora(model, rank=2, skip=("output_head",)) == 0  # not twice
    assert model(torch.tensor([[1, 2, 3]])).shape == (1, 3, 2)


# ---------------------------------------------------------------- finetune.py
class WordEncoding:
    """A stand-in for tiktoken: one ID per word."""

    def encode(self, text):
        return [len(word) % 10 for word in text.split()]


@pytest.fixture
def messages():
    spam = [("spam", f"win a prize now {n}") for n in range(5)]
    ham = [("ham", f"see you at {n}") for n in range(20)]
    return spam + ham


def test_load_sms_spam_reads_the_zip(tmp_path):
    path = tmp_path / "spam.zip"
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("SMSSpamCollection", "ham\tsee you\tat 5\nspam\tWIN now\n")
    assert finetune.load_sms_spam(path) == [
        ("ham", "see you\tat 5"),
        ("spam", "WIN now"),
    ]


def test_balance_keeps_as_many_ham_as_spam(messages):
    kept = finetune.balance(messages, seed=1)
    assert sorted(label for label, _ in kept) == ["ham"] * 5 + ["spam"] * 5
    assert kept == finetune.balance(messages, seed=1)  # same seed, same choice
    assert set(kept) <= set(messages)


def test_split_loses_nothing(messages):
    train, val, test = finetune.split(messages, 0.7, 0.1)
    assert (len(train), len(val), len(test)) == (17, 2, 6)
    assert train + val + test == messages


def test_encode_messages_pads_and_cuts():
    ids, labels = finetune.encode_messages(
        [("ham", "a bb"), ("spam", "a bb ccc dddd")], WordEncoding(), length=3
    )
    assert ids.tolist() == [[1, 2, finetune.PAD_ID], [1, 2, 3]]
    assert labels.tolist() == [0, 1]


def test_make_classifier_swaps_the_head_and_freezes_the_rest():
    model = finetune.make_classifier(GPTModel(TINY), n_classes=2)
    assert model.output_head.out_features == 2
    trainable = {name for name, p in model.named_parameters() if p.requires_grad}
    assert all(
        name.startswith(("blocks.1.", "final_norm.", "output_head."))
        for name in trainable
    )
    assert "blocks.0.attention.query.weight" not in trainable
    assert not model.token_embedding.weight.requires_grad
    ids = torch.randint(0, 11, (4, 5))
    assert finetune.class_scores(model, ids).shape == (4, 2)


def test_train_classifier_learns_a_toy_rule_and_leaves_frozen_weights_alone():
    torch.manual_seed(0)
    model = finetune.make_classifier(GPTModel(TINY).eval())
    frozen_before = model.blocks[0].attention.query.weight.clone()
    ids = torch.randint(0, 10, (64, 5))
    labels = (ids[:, -1] >= 5).long()  # the rule: is the last token 5 or more?
    history = finetune.train_classifier(
        model, ids, labels, epochs=30, lr=0.01, report_every=8
    )
    assert history[-1][1] < history[0][1]
    assert finetune.accuracy(model, ids, labels) > 0.9
    assert torch.equal(model.blocks[0].attention.query.weight, frozen_before)
    assert not model.training
