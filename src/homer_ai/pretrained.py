"""Load OpenAI's GPT-2 weights into our own `GPTModel` (built step by step in 08_01).

GPT-2 small is one file of 548 MB. It is downloaded once, to a cache folder
outside the project (`~/.cache/homer-ai/`, or the folder named by the
environment variable `HOMER_CACHE`), checked, and then only read.

The file is in the *safetensors* format, which is simple enough to read with
plain Python:

    8 bytes   N, the length of the header (little-endian unsigned integer)
    N bytes   the header: JSON, one entry per tensor (dtype, shape, data_offsets)
    the rest  the numbers of every tensor, as raw bytes

Nothing here writes into the project folder.
"""

import hashlib
import json
import os
import struct
import sys
import urllib.request
from pathlib import Path

import torch

from homer_ai.gpt import GPTConfig, GPTModel

CACHE = Path(os.environ.get("HOMER_CACHE") or Path.home() / ".cache" / "homer-ai")

GPT2_URL = "https://huggingface.co/openai-community/gpt2/resolve/main/model.safetensors"
GPT2_FILE = CACHE / "gpt2" / "model.safetensors"
GPT2_BYTES = 548_105_171
GPT2_SHA256 = "248dfc3911869ec493c76e65bf2fcf7f615828b0254c12b473182f0f81d3a707"

# GPT-2 small, in the words of our own GPTConfig (see 04_00)
GPT2_CONFIG = GPTConfig(
    vocab_size=50257,  # GPT-2's BPE tokens (02_03)
    context_length=1024,
    emb_dim=768,
    n_heads=12,
    n_layers=12,
    drop_rate=0.0,  # no dropout: we fine-tune for a short time only
    qkv_bias=True,  # GPT-2's query/key/value layers have a bias; ours had none
)

# safetensors dtype name -> (torch dtype, bytes per number)
DTYPES = {
    "F32": (torch.float32, 4),
    "F16": (torch.float16, 2),
    "I64": (torch.int64, 8),
    "U8": (torch.uint8, 1),
}


def sha256_of(path: Path) -> str:
    """The SHA-256 fingerprint of a file, read in pieces of 1 MB."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while piece := f.read(1 << 20):
            digest.update(piece)
    return digest.hexdigest()


def download(
    url: str,
    path: Path,
    expected_bytes: int | None = None,
    expected_sha256: str | None = None,
) -> Path:
    """Download `url` to `path` once, showing progress, then check it.

    If `path` already exists with the expected size, nothing is downloaded. The
    download goes to a `.part` file first, so a broken download is never taken
    for the real file. Raises `ValueError` if the size or the SHA-256 is wrong.
    """
    path = Path(path)
    if path.exists() and expected_bytes in (None, path.stat().st_size):
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "python-urllib"})
    with urllib.request.urlopen(request) as response, open(part, "wb") as f:
        total = int(response.headers.get("Content-Length") or 0) or expected_bytes
        done, next_report = 0, 0
        while piece := response.read(1 << 20):
            f.write(piece)
            done += len(piece)
            if total and done >= next_report:
                print(
                    f"\r{path.name}: {done / 1e6:6.0f} of {total / 1e6:.0f} MB", end=""
                )
                sys.stdout.flush()
                next_report += total // 20
        print(f"\r{path.name}: {done / 1e6:6.0f} MB downloaded        ")
    if expected_bytes is not None and part.stat().st_size != expected_bytes:
        size = part.stat().st_size
        part.unlink()
        raise ValueError(f"{path.name}: got {size:,} bytes, not {expected_bytes:,}")
    if expected_sha256 is not None and sha256_of(part) != expected_sha256:
        part.unlink()
        raise ValueError(f"{path.name}: the SHA-256 does not match; download again")
    part.replace(path)
    return path


def download_gpt2(path: Path = GPT2_FILE) -> Path:
    """GPT-2 small's weights (548 MB), downloaded once and checked."""
    return download(GPT2_URL, path, GPT2_BYTES, GPT2_SHA256)


def read_safetensors_header(path: Path) -> tuple[dict[str, dict], int]:
    """The header of a safetensors file, and the byte where the numbers start.

    The header maps each tensor name to `{"dtype", "shape", "data_offsets"}`.
    The extra `__metadata__` entry, if there is one, is left out.
    """
    with open(path, "rb") as f:
        (length,) = struct.unpack("<Q", f.read(8))  # <Q: little-endian, 8 bytes
        header = json.loads(f.read(length))
    header.pop("__metadata__", None)
    return header, 8 + length


def read_safetensors(path: Path) -> dict[str, torch.Tensor]:
    """Every tensor of a safetensors file, by name, read with plain Python."""
    header, start = read_safetensors_header(path)
    tensors = {}
    with open(path, "rb") as f:
        for name, entry in header.items():
            dtype, size = DTYPES[entry["dtype"]]
            begin, end = entry["data_offsets"]
            count = 1
            for n in entry["shape"]:
                count *= n
            if end - begin != count * size:
                raise ValueError(f"{name}: {end - begin} bytes for {count} numbers")
            f.seek(start + begin)
            raw = bytearray(f.read(end - begin))
            if count == 0:
                tensors[name] = torch.empty(entry["shape"], dtype=dtype)
            else:
                tensors[name] = torch.frombuffer(raw, dtype=dtype).reshape(
                    entry["shape"]
                )
    return tensors


def gpt2_to_ours(weights: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Rename (and reshape) GPT-2's tensors to the names of our `GPTModel`.

    Three things differ from our model:

    - GPT-2 keeps query, key and value in ONE layer, `attn.c_attn`, with a
      weight of shape `(emb, 3 * emb)`. We cut it into three.
    - GPT-2's layers store a weight as `(inputs, outputs)`; `nn.Linear` wants
      `(outputs, inputs)`. So every such weight is transposed (`.T`).
    - GPT-2 has no separate output head: it reuses the token table `wte`.
      We copy `wte` into our `output_head`.
    """
    emb = weights["wte.weight"].shape[1]
    ours = {
        "token_embedding.weight": weights["wte.weight"],
        "position_embedding.weight": weights["wpe.weight"],
        "final_norm.weight": weights["ln_f.weight"],
        "final_norm.bias": weights["ln_f.bias"],
        "output_head.weight": weights["wte.weight"],
    }
    n_layers = 1 + max(int(k.split(".")[1]) for k in weights if k.startswith("h."))
    for n in range(n_layers):
        theirs, mine = f"h.{n}.", f"blocks.{n}."
        qkv_w = weights[theirs + "attn.c_attn.weight"].T  # (3 * emb, emb)
        qkv_b = weights[theirs + "attn.c_attn.bias"]  # (3 * emb,)
        for i, part in enumerate(("query", "key", "value")):
            rows = slice(i * emb, (i + 1) * emb)
            ours[f"{mine}attention.{part}.weight"] = qkv_w[rows]
            ours[f"{mine}attention.{part}.bias"] = qkv_b[rows]
        for theirs_name, my_name in (
            ("attn.c_proj", "attention.out_proj"),
            ("mlp.c_fc", "feed_forward.0"),
            ("mlp.c_proj", "feed_forward.2"),
        ):
            ours[f"{mine}{my_name}.weight"] = weights[f"{theirs}{theirs_name}.weight"].T
            ours[f"{mine}{my_name}.bias"] = weights[f"{theirs}{theirs_name}.bias"]
        for theirs_name, my_name in (("ln_1", "norm1"), ("ln_2", "norm2")):
            for kind in ("weight", "bias"):
                ours[f"{mine}{my_name}.{kind}"] = weights[
                    f"{theirs}{theirs_name}.{kind}"
                ]
    return {name: tensor.clone() for name, tensor in ours.items()}


def load_gpt2_into(model: GPTModel, weights: dict[str, torch.Tensor]) -> GPTModel:
    """Copy GPT-2's weights into `model`, checking every name and shape.

    `load_state_dict` raises if a name is unknown or a shape differs. The only
    entries we leave alone are the causal masks: they hold no learned numbers.
    """
    result = model.load_state_dict(gpt2_to_ours(weights), strict=False)
    not_masks = [k for k in result.missing_keys if not k.endswith("attention.mask")]
    if not_masks or result.unexpected_keys:
        raise ValueError(f"missing {not_masks}, unexpected {result.unexpected_keys}")
    return model


def load_gpt2(path: Path | None = None) -> GPTModel:
    """Our `GPTModel` with GPT-2 small's weights, in `eval()` mode.

    Downloads the file first if it is not in the cache yet.
    """
    path = Path(path) if path else download_gpt2()
    model = GPTModel(GPT2_CONFIG)
    load_gpt2_into(model, read_safetensors(path))
    return model.eval()


def continue_text(
    model: GPTModel,
    encoding,
    prompt: str,
    n_new: int,
    top_k: int | None = None,
    seed: int = 123,
    stop_id: int | None = None,
) -> str:
    """The prompt plus up to `n_new` new tokens (the writing loop of 06_01).

    With `top_k=None` the most likely token is taken every time (greedy). With
    a number, one of the `top_k` most likely tokens is drawn by chance (06_02).
    `encoding` is a tiktoken encoding. Writing stops early at `stop_id`.
    """
    torch.manual_seed(seed)
    ids = torch.tensor([encoding.encode(prompt)])
    with torch.no_grad():
        for _ in range(n_new):
            scores = model(ids[:, -model.context_length :])[0, -1]
            if top_k is None:
                best = scores.argmax().view(1)
            else:
                top_scores, top_ids = torch.topk(scores, top_k)
                pick = torch.multinomial(torch.softmax(top_scores, dim=-1), 1)
                best = top_ids[pick]
            if best.item() == stop_id:
                break
            ids = torch.cat([ids, best.view(1, 1)], dim=1)
    return encoding.decode(ids[0].tolist())
