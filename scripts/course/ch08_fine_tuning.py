# ruff: noqa: E501
"""Chapter 8, Advanced: fine-tuning a pretrained model (GPT-2)."""

from .cells import (
    background,
    code,
    learn,
    md,
    q,
    questions,
    recap,
    save,
    saw,
    step_intro,
    title,
    where,
    words,
)
from .diagrams import DIAGRAM_LORA, DIAGRAM_PRETRAINED
from .setup_code import CH8_GPT2, CH8_SPAM

FOLDER = "08_advanced_fine_tuning/"
HELPERS = "The helper code lives in `src/homer_ai/`: `pretrained.py`, `finetune.py` and `lora.py`, imported like the rest of the project."


def build():
    # ---------------------------------------------------------------- 08_00
    save(
        FOLDER + "08_00_chapter_overview.ipynb",
        [
            title(
                "08_00 Advanced: fine-tuning a pretrained model (overview)",
                "section 5.5, chapters 6 and 7, and appendix E",
            ),
            where(
                None,
                [
                    "src/homer_ai/gpt.py",
                    "src/homer_ai/pretrained.py",
                    "src/homer_ai/finetune.py",
                    "src/homer_ai/lora.py",
                ],
                "**after the main course**: a borrowed model, and a new job for it",
            ),
            learn(
                [
                    "The four ways to get a model to do a job, from cheapest to dearest",
                    "What fine-tuning is, and why we borrow GPT-2 for it instead of using our Homer model",
                    "What freezing, a classification head, instruction fine-tuning and LoRA are, in one picture each",
                    "What this chapter downloads, and what results to expect on a laptop",
                ]
            ),
            words(
                [
                    (
                        "pretrained model",
                        "a model somebody already trained, shared as a file of weights",
                        "GPT-2 small: one file of 548 MB",
                    ),
                    (
                        "fine-tuning",
                        "taking a pretrained model and training it a little more, on a small set of examples of one job",
                        "GPT-2, then 1,045 text messages marked spam or not",
                    ),
                    (
                        "GPT-2 small",
                        "the smallest of the GPT-2 models OpenAI published in 2019: 12 blocks, 124 million weights, trained on English web pages",
                        "the model this chapter borrows",
                    ),
                    (
                        "RAG",
                        "retrieval-augmented generation: a program first looks up text that fits the question and pastes it into the prompt, so the model answers from that text",
                        "a chatbot that searches your handbook before it answers",
                    ),
                ]
            ),
            *background(),
            q(
                "I want a model to do one job for me. What are my options?",
                "There are four, and it helps to see them as a ladder. Each step up costs more, so you climb only as far as you need.\n\n"
                "```\n"
                "4. train from scratch   your own model, your own data          weeks, a lot of data\n"
                "3. fine-tune            a pretrained model + your examples      minutes to hours\n"
                "2. RAG                  look up the right text, paste it in     no training\n"
                "1. prompt               just ask a big model                    no training\n"
                "```\n\n"
                "**Prompting** is asking well. **RAG** (retrieval-augmented generation) means a program looks up the text that fits the question and puts it in the prompt, so the model answers from it. "
                "**Fine-tuning** changes the weights of a model somebody already trained. **From scratch** is what chapters 1 to 7 did.\n\n"
                "We started at the top of the ladder on purpose, because that's how you learn what's inside. In real work you start at the bottom. This chapter is step 3.",
            ),
            q(
                "So what is fine-tuning, really?",
                "Here's the picture I keep in my head.\n\n"
                "Our Homer model is like a child we taught to babble Greek from scratch. GPT-2 is an adult who already speaks English. "
                '**Fine-tuning** is giving that adult a short course for one new job, say "spot spam messages". That\'s far cheaper than raising a new child for it.\n\n'
                "The rest of the chapter fits the same picture:\n\n"
                "| In the picture | In the model | Notebook |\n|---|---|---|\n"
                "| the adult who already speaks English | GPT-2's weights, loaded into our own `GPTModel` | [08_01](08_01_loading_gpt2.ipynb) |\n"
                "| they keep everything they know and learn only the new bit | **freezing**: most weights are locked, a few are trained | [08_02](08_02_spam_classifier.ipynb) |\n"
                "| we teach them to take requests politely | **instruction fine-tuning**: examples of a request and its answer | [08_03](08_03_instructions.ipynb) |\n"
                "| they take notes on a small notepad instead of rewiring their brain | **LoRA**: the weights stay locked, two thin tables beside them are trained | [08_04](08_04_lora.ipynb) |\n\n"
                "The picture has a limit, and I'll say it once. An adult understands the job. GPT-2 doesn't. It's still only guessing the next token, and fine-tuning just shifts those guesses. So when it fails, it doesn't fail like a person would. You'll see that in 08_03.",
            ),
            q(
                "Why not fine-tune our own Homer model?",
                "Because there's almost nothing in it to build on. Fine-tuning works when the model already knows a lot, and the short course only has to point that knowledge at a job.\n\n"
                "Our GPT read 17,730 characters of one book of Greek. GPT-2 read about 40 GB of English web pages. Let's put the two side by side. The next cell builds both models empty (no download yet), just to count.\n\n"
                "👉 **Predict first:** our GPT has 835,072 weights. How many times bigger is GPT-2 small: 10 times, 100 times, more?",
            ),
            code(
                r"""from homer_ai.gpt import GPTConfig, GPTModel
from homer_ai.pretrained import GPT2_CONFIG  # GPT-2 small, written as our own GPTConfig

ours = GPTConfig(vocab_size=137)  # the GPT of chapter 4


def count(config):
    return sum(p.numel() for p in GPTModel(config).parameters())


print(f"{'':16} {'our GPT':>12} {'GPT-2 small':>12}")
for name in ("vocab_size", "context_length", "emb_dim", "n_heads", "n_layers"):
    print(f"{name:16} {getattr(ours, name):>12,} {getattr(GPT2_CONFIG, name):>12,}")
n_ours, n_gpt2 = count(ours), count(GPT2_CONFIG)
print(f"{'weights':16} {n_ours:>12,} {n_gpt2:>12,}")
print(f"GPT-2 small is {n_gpt2 / n_ours:.0f} times bigger")"""
            ),
            saw(
                "About 195 times bigger: 163,037,184 weights against 835,072. Every row is the same *kind* of setting, though. GPT-2 small is our `GPTModel` with bigger numbers: 12 blocks where we have 4, 768 numbers per token where we have 128. "
                "That's why its weights fit into our class, which is what [08_01](08_01_loading_gpt2.ipynb) does.\n\n"
                '(You may know GPT-2 small as "124 million". Built our way it has 163 million, because GPT-2 uses one table twice and we keep two copies. 08_01 counts it out.)'
            ),
            q(
                "Why GPT-2, of all models?",
                "Three reasons. It's small enough for a laptop: one file of 548 MB, and it runs on a CPU. OpenAI published its weights under a modified MIT licence, so anyone may use them. "
                "And it's the model our chapter 4 copies, block for block, so there's nothing new to learn before we can load it.\n\n"
                "It's also old (2019) and weak next to today's models. That's fine for us. The steps are the same for a big model, they just need a bigger computer.",
            ),
            q(
                "What do I need on my computer?",
                "- **One download of 548 MB**, GPT-2's weights. It goes to a cache folder outside the project, `~/.cache/homer-ai/gpt2/`, and is checked after downloading. 08_01 does it; the later notebooks reuse the file.\n"
                "- **One download of 0.2 MB**, the text messages for 08_02 and 08_04, to `~/.cache/homer-ai/sms_spam/`.\n"
                "- About **2 GB of free memory** while a notebook runs.\n"
                "- A few minutes per notebook. On my MacBook (M1 Pro, CPU only) the slowest one takes about two and a half minutes.\n\n"
                "Nothing is written into the project folder, and your Homer models aren't touched. "
                + HELPERS,
            ),
            q(
                "How good will the results be?",
                "Mixed, and I'd rather tell you now. These are the numbers from my laptop:\n\n"
                "| Notebook | What we measure | Result |\n|---|---|---|\n"
                "| 08_01 | does GPT-2 write English from inside our own class? | yes, fluent English |\n"
                "| 08_02 | spam or not, on 300 messages the model never saw | 93.7% right, after about a minute of training |\n"
                "| 08_03 | three small text jobs given as instructions | the easy one 28 of 30, the two harder ones 15 of 30 each |\n"
                "| 08_04 | spam again, with LoRA | 94.3% right, with 5 times fewer trained weights, but each step is slower |\n\n"
                "So sorting text works well, even on a CPU. Following instructions works only for the easiest job. GPT-2 small is too small, and a minute of training is too short, for more than that.",
            ),
            q(
                "Which notebook do I open for each idea?",
                "| Notebook | Question it answers |\n|---|---|\n"
                "| [`08_01_loading_gpt2.ipynb`](08_01_loading_gpt2.ipynb) | What's inside a weights file, and how do GPT-2's numbers get into our model? |\n"
                "| [`08_02_spam_classifier.ipynb`](08_02_spam_classifier.ipynb) | How do I turn a model that writes into one that sorts messages? |\n"
                "| [`08_03_instructions.ipynb`](08_03_instructions.ipynb) | How do I teach a model to answer a request, and how far does a small one get? |\n"
                "| [`08_04_lora.ipynb`](08_04_lora.ipynb) | How do I fine-tune by training far fewer numbers? |",
            ),
            recap(
                [
                    "Prompt, RAG, fine-tune, train from scratch: four ways to get a job done, each dearer than the one before.",
                    "Fine-tuning is a short course for a model that already knows a lot. Our Homer model knows too little, so we borrow GPT-2 small.",
                    "GPT-2 small is our `GPTModel` with bigger settings, so its weights fit into our own class.",
                    "Freezing, instruction fine-tuning and LoRA are three ways to run that short course.",
                ],
                "[08_01: Loading GPT-2's weights by hand](08_01_loading_gpt2.ipynb).",
            ),
        ],
    )

    # ---------------------------------------------------------------- 08_01
    save(
        FOLDER + "08_01_loading_gpt2.ipynb",
        [
            title(
                "08_01 Loading GPT-2's weights by hand",
                "section 5.5 (loading pretrained weights from OpenAI)",
            ),
            where(
                None,
                [
                    "src/homer_ai/gpt.py",
                    "src/homer_ai/pretrained.py",
                ],
                "**after the main course**: getting the adult who already speaks English",
            ),
            learn(
                [
                    "What a weights file really is: a list of names, shapes and raw numbers",
                    "How to read a safetensors file with plain Python, no extra library",
                    "How GPT-2's names and table layouts map onto our `GPTModel`",
                    "How to check the loading worked: the model writes English",
                ]
            ),
            words(
                [
                    (
                        "safetensors",
                        "a simple file format for weights: a short text header that lists every tensor, then all the numbers as raw bytes",
                        "`model.safetensors`, 548 MB",
                    ),
                    (
                        "header",
                        "the first part of a file, which describes the rest",
                        '`"wpe.weight": {"dtype": "F32", "shape": [1024, 768], …}`',
                    ),
                    (
                        "float32",
                        "a number with a decimal point, stored in 4 bytes; the usual type of a weight",
                        "1024 × 768 float32 numbers take 3,145,728 bytes",
                    ),
                    (
                        "SHA-256",
                        "a fingerprint of a file: 64 characters that change completely if one byte of the file changes",
                        "`248dfc39…a707` for GPT-2's file",
                    ),
                    (
                        "weight tying",
                        "using one table for two jobs: GPT-2's token table is also its output head",
                        "`wte` is used at the bottom and at the top",
                    ),
                ]
            ),
            *background(),
            q(
                'What does it mean to "load GPT-2"?',
                "A trained model is two things: the code that does the maths, and the numbers the maths uses (the weights). The code we already have, because `src/homer_ai/gpt.py` is built like GPT-2. So all we need is OpenAI's numbers.\n\n"
                "Think of our `GPTModel` as an empty form, and of GPT-2's file as a filled-in copy of nearly the same form, from another office. The boxes are the same, but they're named differently, and a few are laid out the other way round. "
                "This notebook copies each box across.\n\n"
                "```\n"
                "model.safetensors (548 MB)          our GPTModel (gpt.py)\n"
                "  wte.weight        ───────────▶    token_embedding.weight\n"
                "  wpe.weight        ───────────▶    position_embedding.weight\n"
                "  h.0.ln_1.weight   ───────────▶    blocks.0.norm1.weight\n"
                "  …                                 …\n"
                "```\n\n"
                "In the picture of [08_00](08_00_chapter_overview.ipynb), this is where we get the adult who already speaks English.",
            ),
            q(
                "Why by hand? Isn't there a library for this?",
                "There is. With the `transformers` library it's one line. But then you'd have a model you can't see inside, and the point of this course is that there's no magic in it. "
                "The file format is simple enough to read with the Python standard library, so we do that, and add no new dependency.",
            ),
            step_intro(),
            q(
                "Step 1: where do the weights come from?",
                "From Hugging Face, a site where people share models. The file is `model.safetensors` in the `openai-community/gpt2` repository. `download_gpt2` (in `pretrained.py`) fetches it once, to `~/.cache/homer-ai/gpt2/`, and shows progress. "
                "After that it only checks that the file is there.\n\n"
                "A download can break or be tampered with, so we check two things: the size, and the **SHA-256**, a fingerprint of the whole file. "
                + HELPERS,
            ),
            code(
                r"""from homer_ai.pretrained import GPT2_BYTES, GPT2_SHA256, download_gpt2, sha256_of

path = download_gpt2()  # 548 MB the first time; after that, nothing to do
print(path)
print(f"size:    {path.stat().st_size:,} bytes  (expected {GPT2_BYTES:,})")
print("SHA-256:", sha256_of(path))
print("matches: ", sha256_of(path) == GPT2_SHA256)"""
            ),
            saw(
                "548,105,171 bytes, and the fingerprint `248dfc39…a707` matches the one written in `pretrained.py`. So this is byte for byte the file we expect."
            ),
            q(
                "Step 2: what's inside the file?",
                "Three parts, one after the other:\n\n"
                "```\n"
                "┌─────────┬──────────────────────────────┬──────────────────────────────────┐\n"
                "│ 8 bytes │ N bytes                      │ the rest                         │\n"
                "│ N       │ the header: a JSON text that │ the numbers of every tensor,     │\n"
                "│         │ lists every tensor           │ as raw bytes, back to back       │\n"
                "└─────────┴──────────────────────────────┴──────────────────────────────────┘\n"
                "```\n\n"
                'The first 8 bytes are one whole number: how long the header is. `struct.unpack("<Q", …)` turns 8 bytes into that number (`<` means the smallest byte comes first, which is called little-endian; `Q` means an 8-byte whole number without a sign).\n\n'
                "👉 **Predict first:** the file is 548 MB. Will the header be a big part of it, or tiny?",
            ),
            code(
                r"""import json
import struct

with open(path, "rb") as f:  # "rb": read the raw bytes, not text
    (header_length,) = struct.unpack("<Q", f.read(8))
    header = json.loads(f.read(header_length))

print("header length:", f"{header_length:,} bytes")
print("entries:      ", len(header))
print()
print('"wpe.weight":', header["wpe.weight"])"""
            ),
            saw(
                "The header is 14,283 bytes, a tiny part of 548 MB. It has 161 entries: one per tensor (160 of them), and one small note called `__metadata__` that we skip. The entry for `wpe.weight` (GPT-2's position table) says: the numbers are `F32` (float32, 4 bytes each), the shape is 1024 × 768, and they sit at bytes 267,533,312 to 270,679,040 of the data part.\n\n"
                "Check it: 1024 × 768 × 4 = 3,145,728, and 270,679,040 − 267,533,312 = 3,145,728. It fits."
            ),
            q(
                "Step 3: how do I get one tensor out?",
                "Jump to where its bytes start, read them, and tell PyTorch what they are. The data part begins right after the header, at byte `8 + header_length`.\n\n"
                "`torch.frombuffer` takes raw bytes and reads them as numbers of the type you name. It gives one long row, so we `reshape` it to the shape the header promised.",
            ),
            code(
                r"""import torch

entry = header["wpe.weight"]
begin, end = entry["data_offsets"]
with open(path, "rb") as f:
    f.seek(8 + header_length + begin)  # jump to the first byte of this tensor
    raw = bytearray(f.read(end - begin))

wpe = torch.frombuffer(raw, dtype=torch.float32).reshape(entry["shape"])
print("bytes read:", f"{len(raw):,}")
print("shape:     ", tuple(wpe.shape))
print("position 0, first 4 numbers:", [round(v, 4) for v in wpe[0, :4].tolist()])"""
            ),
            saw(
                "3,145,728 bytes became a table of shape `(1024, 768)`: one row of 768 numbers for each of the 1,024 positions GPT-2 can read. Row 0 starts `-0.0188, -0.1974, 0.004, 0.0113`. "
                "Somebody's training run found those numbers in 2019, and now they're in your notebook.\n\n"
                "`read_safetensors` in `pretrained.py` is this cell in a loop over all 160 entries."
            ),
            q(
                "Step 4: which tensors are there, and do they match ours?",
                "Let's read them all and look at the names. GPT-2 calls its blocks `h.0` to `h.11`. We print block 0, and the few tensors outside the blocks.\n\n"
                "👉 **Predict first:** our attention has three layers, `query`, `key` and `value`. Do you expect to find three in GPT-2's list?",
            ),
            code(
                r"""from homer_ai.pretrained import read_safetensors

theirs = read_safetensors(path)  # a dict: name -> tensor
for name in sorted(theirs):
    if name.startswith("h.0.") or not name.startswith("h."):
        print(f"{name:26} {tuple(theirs[name].shape)}")"""
            ),
            saw(
                "There's no `query`, `key` or `value`. There's one layer, `h.0.attn.c_attn`, with a weight of shape `(768, 2304)`. And 2304 = 3 × 768. GPT-2 keeps the three side by side in **one** table. That's the first difference.\n\n"
                "The rest is the parts we know from chapter 4, under shorter names: `wte` (the token table, 50,257 × 768), `wpe` (the position table), `ln_1` and `ln_2` (the two layer norms of a block), `attn.c_proj` (attention's output projection), `mlp.c_fc` and `mlp.c_proj` (the two layers of the feed-forward step), and `ln_f` (the final layer norm).\n\n"
                "`h.0.attn.bias` is not a bias at all. It's the causal mask of [03_03](../03_attention/03_03_hiding_the_future.ipynb), saved as a table of ones and zeros. Our model builds its own mask, so we skip it."
            ),
            md(
                "Here's the whole mapping in one picture. The orange row is the one that needs cutting."
            ),
            code(DIAGRAM_PRETRAINED),
            q(
                "Step 5: how do we cut the one table into query, key and value?",
                "First a detail about layout. Our `nn.Linear` stores its weight as `(outputs, inputs)`. GPT-2's layers store it as `(inputs, outputs)`, the other way round. So every GPT-2 weight table has to be turned on its side with `.T` (transpose: rows become columns).\n\n"
                "For `c_attn` that gives `(2304, 768)`: 2,304 output rows. The first 768 rows are the query layer, the next 768 the key layer, the last 768 the value layer.\n\n"
                "```\n"
                "c_attn.weight  (768, 2304)   ── .T ──▶  (2304, 768)\n"
                "                                         rows    0 …  767   → query.weight  (768, 768)\n"
                "                                         rows  768 … 1535   → key.weight    (768, 768)\n"
                "                                         rows 1536 … 2303   → value.weight  (768, 768)\n"
                "```\n\n"
                "How do we know we cut it right? One fused layer and three separate layers must give the same numbers for the same input. Let's check on a random row of 768 numbers.",
            ),
            code(
                r"""from torch import nn

fused_w = theirs["h.0.attn.c_attn.weight"].T  # (2304, 768)
fused_b = theirs["h.0.attn.c_attn.bias"]  # (2304,)

torch.manual_seed(0)
x = torch.randn(768)  # stands for one token on its way into attention
together = fused_w @ x + fused_b  # GPT-2's way: 2304 numbers at once

parts = {}
for i, name in enumerate(["query", "key", "value"]):
    rows = slice(i * 768, (i + 1) * 768)
    layer = nn.Linear(768, 768)  # our way: one layer each
    layer.load_state_dict({"weight": fused_w[rows], "bias": fused_b[rows]})
    parts[name] = layer(x)
    shapes = tuple(layer.weight.shape), tuple(layer.bias.shape)
    print(f"{name:6} weight {shapes[0]}  bias {shapes[1]}")

ours = torch.cat([parts["query"], parts["key"], parts["value"]])
print("same numbers as the fused layer:", torch.allclose(together, ours, atol=1e-5))"""
            ),
            saw(
                "Three layers of `(768, 768)`, each with a bias, and together they give the same 2,304 numbers as GPT-2's one layer. So the cut is right.\n\n"
                "Notice the bias. Our Homer GPT has no bias in query, key and value (`qkv_bias=False`). GPT-2 has one, so the config for it says `qkv_bias=True`."
            ),
            q(
                "Step 6: how does everything get into our model?",
                "`gpt2_to_ours` in `pretrained.py` does steps 4 and 5 for all 12 blocks: it renames every tensor, transposes the weight tables, and cuts `c_attn` in three. The result is a dict with **our** names. "
                "Then PyTorch's `load_state_dict` copies it into the model, and complains if a name is unknown or a shape is wrong. That check is what makes me trust the mapping.\n\n"
                "One box is left over: our `output_head`. GPT-2 has no tensor for it. It reuses the token table `wte` for the output (that's called **weight tying**), so we copy `wte` into our head.\n\n"
                "👉 **Predict first:** GPT-2 small is known as a 124-million-weight model. Will our model report 124 million too?",
            ),
            code(
                r"""from homer_ai.gpt import GPTModel
from homer_ai.pretrained import GPT2_CONFIG, gpt2_to_ours

print(GPT2_CONFIG)
model = GPTModel(GPT2_CONFIG)  # our class, GPT-2's sizes, random numbers for now
result = model.load_state_dict(gpt2_to_ours(theirs), strict=False)
model.eval()

masks = [name for name in theirs if name.endswith(".attn.bias")]  # no learned numbers
learned = sum(t.numel() for name, t in theirs.items() if name not in masks)
in_ours = sum(p.numel() for p in model.parameters())
print()
untouched = result.missing_keys
print("left untouched in our model:", len(untouched), "tensors, e.g.", untouched[0])
print("names we don't have:        ", result.unexpected_keys)
print(f"learned numbers in the file: {learned:>12,}")
print(f"weights in our model:        {in_ours:>12,}")
print(f"difference:                  {in_ours - learned:>12,}  = 50,257 x 768")"""
            ),
            saw(
                "Every name found its place: the only 12 tensors left untouched are our own causal masks, one per block, and there are no unknown names.\n\n"
                "The file has 124,439,808 learned numbers. Our model has 163,037,184. The difference is exactly 50,257 × 768 = 38,597,376: the copy of `wte` in our output head. GPT-2 stores that table once and uses it twice; we store it twice. The numbers are the same, so the model computes the same thing."
            ),
            q(
                "Step 7: did it work? Can it write English?",
                "The real test. If one table were in the wrong place or the wrong way round, we'd get rubbish.\n\n"
                "GPT-2 doesn't read characters. It reads BPE tokens ([02_03](../02_text_to_numbers/02_03_bpe.ipynb)), and `tiktoken` has its exact vocabulary ([A_05](../A_toolbox/A_05_tiktoken_bpe.ipynb)). "
                "`continue_text` is the writing loop of [06_01](../06_making_it_talk/06_01_writing_greek.ipynb): score the next token, pick one, add it, repeat.\n\n"
                "We give the same prompt to two models: our class with random numbers, and our class with GPT-2's numbers. Greedy writing, so always the most likely token.\n\n"
                "👉 **Predict first:** what will the random one write?",
            ),
            code(
                r"""import tiktoken

from homer_ai.pretrained import continue_text

encoding = tiktoken.get_encoding("gpt2")
prompt = "Every effort moves you"
print("prompt as tokens:", [encoding.decode([i]) for i in encoding.encode(prompt)])

torch.manual_seed(0)
random_model = GPTModel(GPT2_CONFIG).eval()  # same class, untrained
print()
print("random weights:", repr(continue_text(random_model, encoding, prompt, 12)))
print("GPT-2 weights: ", repr(continue_text(model, encoding, prompt, 20)))"""
            ),
            saw(
                "The random model writes token soup: `Every effort moves youandalrarilywestNAT nerds Bearing…`. Same code, no knowledge.\n\n"
                "With GPT-2's numbers the same code writes `Every effort moves you forward.\\n\\nThe first step is to understand the importance of your work.` That's real English, and it's the same continuation other GPT-2 implementations give for this prompt with greedy writing. So the weights are in the right boxes."
            ),
            q(
                "What does it make of Homer?",
                "Let's give it Butler's first words, and let chance in with top-k ([06_02](../06_making_it_talk/06_02_less_repetition.ipynb)): one of the 40 most likely tokens each time.",
            ),
            code(
                r"""prompt = "Tell me, O Muse, of"
print("greedy:", repr(continue_text(model, encoding, prompt, 30)))
print("top-40:", repr(continue_text(model, encoding, prompt, 30, top_k=40, seed=1)))"""
            ),
            saw(
                'Greedy loops, as greedy does: `…of what you have done to me?\\n\\nO Muse, of what you have done to me?` With top-40 it writes fluent English that goes its own way, something like `…of what you have done." "Hush, yes. That said, you\'ve done very good.`\n\n'
                "So it writes proper sentences, which our Homer model never did. But it isn't Butler and it isn't the Odyssey. It continues in the style of the web pages it learned from. That's the adult who speaks English, with no course for any job yet."
            ),
            *questions(
                [
                    (
                        "Is it safe to open a weights file from the internet?",
                        "With safetensors, yes, and that's why the format exists. It holds only a JSON text and raw numbers, so reading it can't run any code. "
                        "Older `.pt` or `.bin` files can contain code that runs when you open them, which is why our own checkpoints are loaded with `weights_only=True` ([05_06](../05_training_on_the_odyssey/05_06_saving_and_loading.ipynb)). "
                        "The SHA-256 check on top tells us the file is the one we meant to get.",
                    ),
                    (
                        "Is our model now exactly GPT-2?",
                        "Very nearly. One small thing differs: between the two feed-forward layers, OpenAI's code uses a quick approximation of the GELU curve ([04_02](../04_building_the_model/04_02_feed_forward.ipynb)), and our `nn.GELU()` uses the exact curve. "
                        "The two curves differ by less than 0.001 everywhere, and the text above came out the same. For fine-tuning it makes no difference.",
                    ),
                    (
                        "Why 1,024 positions, when we only gave it a few tokens?",
                        "1,024 is the most it can read at once (its context length). Shorter text just uses the first rows of the position table, as in [02_06](../02_text_to_numbers/02_06_positions.ipynb).",
                    ),
                    (
                        "Can I load the bigger GPT-2 models the same way?",
                        "Yes. GPT-2 medium, large and XL use the same names and layout, with more blocks and wider rows. You'd change the URL and the sizes in the config. They're 1.5 GB to 6 GB, and slow on a CPU, so this course stays with small.",
                    ),
                ]
            ),
            recap(
                [
                    "A weights file is a header (names, shapes, where the bytes are) plus raw numbers. A safetensors file can be read with `struct`, `json` and `torch.frombuffer`.",
                    "GPT-2's tensors map one to one onto our `GPTModel`. Three things need care: other names, weight tables stored the other way round (`.T`), and query, key and value fused into one table.",
                    "GPT-2 ties its output head to its token table, so our model holds 163 million numbers for GPT-2's 124 million.",
                    "The check is to make it write: the same class gives token soup with random numbers, and English with GPT-2's.",
                    "`load_gpt2()` in `pretrained.py` does all of it in one call. The next notebooks start from there.",
                ],
                "[08_02: A spam classifier](08_02_spam_classifier.ipynb).",
            ),
        ],
    )

    # ---------------------------------------------------------------- 08_02
    save(
        FOLDER + "08_02_spam_classifier.ipynb",
        [
            title(
                "08_02 Classification fine-tuning: a spam filter",
                "chapter 6 (fine-tuning for classification)",
            ),
            where(
                None,
                [
                    "src/homer_ai/finetune.py",
                    "src/homer_ai/pretrained.py",
                    "src/homer_ai/gpt.py",
                ],
                "**after the main course**: the adult's short course for one job",
            ),
            learn(
                [
                    "What classification is, and how it differs from writing text",
                    "How to prepare labelled data: balance it, split it three ways, pad it to one length",
                    "How to swap the output head for a 2-score one, and freeze most of the model",
                    "How to measure accuracy and read a confusion table",
                    "How much one minute of training on a laptop CPU gets you",
                ]
            ),
            words(
                [
                    (
                        "classification",
                        "sorting each input into one of a fixed set of groups (classes)",
                        "a message is `spam` or `ham`",
                    ),
                    (
                        "label",
                        "the right answer that comes with a training example",
                        "`spam`",
                    ),
                    (
                        "ham",
                        "the data set's word for a normal message, one that isn't spam",
                        "`Ok lar... Joking wif u oni...`",
                    ),
                    (
                        "balanced",
                        "a data set with the same number of examples of each class",
                        "747 spam and 747 ham",
                    ),
                    (
                        "test set",
                        "examples kept aside until the very end, to measure the finished model once",
                        "300 messages",
                    ),
                    (
                        "padding",
                        "filler tokens added to short texts so every row of a batch has the same length",
                        "token 50256 repeated",
                    ),
                    (
                        "classification head",
                        "an output layer with one score per class instead of one per token",
                        "`nn.Linear(768, 2)`",
                    ),
                    (
                        "frozen",
                        "a weight that training isn't allowed to change (`requires_grad = False`)",
                        "blocks 0 to 10 of GPT-2",
                    ),
                    (
                        "accuracy",
                        "the share of examples the model gets right",
                        "281 of 300 = 93.7%",
                    ),
                    (
                        "confusion table",
                        "a small table that counts, for each true class, what the model said",
                        '"spam called ham: 12"',
                    ),
                ]
            ),
            *background(),
            q(
                "What job are we teaching, and why this one?",
                "**Classification**: sorting each input into one of a fixed set of groups. Ours is the classic one, a spam filter. A text message goes in, and one of two answers comes out: `spam`, or `ham` (the data set's word for a normal message).\n\n"
                "It's a good first job because the answer is short and easy to check. A written text can be good in a hundred ways. A label is right or wrong, so we can count.\n\n"
                "In the picture of [08_00](08_00_chapter_overview.ipynb): GPT-2 is the adult who already reads English. We give them a short course, a thousand example messages with the right answer on each. "
                "And while they take it, they keep nearly everything they know and learn only the new bit. That's what **freezing** is, further down.",
            ),
            q(
                "A GPT writes text. How can it sort messages?",
                "By changing one part. Everything up to the final layer norm stays. That part turns a message into 768 numbers per token, and those numbers already carry a lot about what the text says. "
                "Only the last layer changes: the output head.\n\n"
                "```\n"
                "writing:      … → final norm → output head (768 → 50,257 scores: which token comes next?)\n"
                "classifying:  … → final norm → new head    (768 → 2 scores: ham or spam?)\n"
                "```\n\n"
                "The new head starts with random numbers, so at first it guesses. Training it, plus a small part of the model under it, is the whole job.",
            ),
            step_intro(),
            q(
                "Step 1: what does the data look like?",
                "It's the **SMS Spam Collection** by Tiago Almeida and José María Gómez Hidalgo: 5,574 real English text messages, each marked `ham` or `spam` by hand. It comes from the UCI Machine Learning Repository, under the CC BY 4.0 licence (free to use if you name the authors). "
                "`load_sms_spam` downloads the 0.2 MB zip once to `~/.cache/homer-ai/sms_spam/`, checks its SHA-256, and reads it. "
                + HELPERS
                + "\n\n"
                "👉 **Predict first:** of 5,574 messages, how many do you think are spam: about half?",
            ),
            code(
                r"""from collections import Counter

from homer_ai.finetune import load_sms_spam

messages = load_sms_spam()  # a list of (label, text) pairs
print(len(messages), "messages:", dict(Counter(label for label, _ in messages)))
print()
for label, text in messages[1:4]:
    print(f"{label:5}| {text[:90]}")"""
            ),
            saw(
                "Not half at all: 4,827 ham and only 747 spam.\n\n"
                "```\n"
                "ham  | Ok lar... Joking wif u oni...\n"
                "spam | Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121 to…\n"
                "ham  | U dun say so early hor... U c already then say...\n"
                "```\n\n"
                "The ham is chatty and full of slang (much of it from Singapore). The spam shouts about prizes and phone numbers. A person tells them apart at a glance."
            ),
            q(
                "Step 2: why can't we train on it as it is?",
                'Because 87% of it is ham. A lazy model that answers "ham" every time would be right 87% of the time and would have learned nothing. We want a data set where guessing gives 50%, so that any score above 50% means something.\n\n'
                "So we **balance** it: keep all 747 spam, and pick 747 ham at random (with a fixed seed, so you get the same ones as I do). Then we cut the 1,494 messages into three parts:\n\n"
                "```\n"
                "training   70%   the model learns from these\n"
                "validation 10%   we peek at these while we work (05_05)\n"
                "test       20%   we look at these ONCE, at the very end\n"
                "```\n\n"
                "The **test set** is new. In chapter 5 we had only training and validation. But if we tune settings while watching the validation score, that score slowly stops being a fair exam. The test set is the exam nobody has seen.",
            ),
            code(
                r"""from homer_ai.finetune import balance, split

balanced = balance(messages, seed=123)
train, val, test = split(balanced, train=0.7, val=0.1)
for name, part in [("training", train), ("validation", val), ("test", test)]:
    counts = Counter(label for label, _ in part)
    ham, spam = counts["ham"], counts["spam"]
    print(f"{name:10} {len(part):>5} messages   ham {ham:>3}   spam {spam:>3}")"""
            ),
            saw(
                "1,045 for training, 149 for validation, 300 for the test. Each part is close to half ham and half spam (not exactly, because the cut is by position in a shuffled list)."
            ),
            q(
                "Step 3: how do messages of different lengths go into one tensor?",
                "A batch is a table, so every row needs the same length ([05_07](../05_training_on_the_odyssey/05_07_train_the_gpt.ipynb)). Messages don't do us that favour. So we pick one length and make every message fit: cut the long ones, and fill the short ones with a filler token. That's **padding**. "
                "As filler we use GPT-2's `<|endoftext|>`, token 50256.\n\n"
                "Here it is for two real messages and a length of 12:\n\n"
                "```\n"
                "text  : Ok lar... Joking wif u oni...\n"
                "ids   : 18690  26371  986  …  72  986              (11 tokens)\n"
                "padded: 18690  26371  986  …  72  986  50256       (12: one filler added)\n"
                "\n"
                "text  : Free entry in 2 a wkly comp to win FA Cup final tkts …\n"
                "ids   : 11146  5726  287  362  257  …              (50 tokens)\n"
                "padded: 11146  5726  287  362  257  …              (12: cut after the twelfth)\n"
                "```\n\n"
                "👉 **Predict first:** `Ok lar... Joking wif u oni...` has 6 words. Will it be 6 tokens?",
            ),
            code(
                r"""import tiktoken

from homer_ai.finetune import PAD_ID, encode_messages

encoding = tiktoken.get_encoding("gpt2")
for label, text in messages[1:3]:
    ids = encoding.encode(text)
    padded, _ = encode_messages([(label, text)], encoding, length=12)
    print("text  :", text[:60])
    print("tokens:", "|".join(encoding.decode([i]) for i in ids[:12]))
    print(f"ids   : {len(ids)} tokens, starting {ids[:4]}")
    print("padded:", padded[0].tolist())
    print()
print("the filler token:", PAD_ID, repr(encoding.decode([PAD_ID])))"""
            ),
            saw(
                "Eleven tokens, not six. GPT-2's BPE cuts slang into pieces (`Ok| lar|...| J|oking| w|if| u| on|i|...`; a token often starts with its space), so the short message needed one filler, 50256, to reach 12. The spam message has 50 tokens and was cut after the twelfth."
            ),
            q(
                "Which length do we pick for real?",
                "Long enough to keep nearly every message whole, and no longer, because every extra token costs time on every step. Let's look at the lengths in the training set.",
            ),
            code(
                r"""lengths = sorted(len(encoding.encode(text)) for _, text in train)
middle = lengths[len(lengths) // 2]
print("shortest:", lengths[0], " middle:", middle, " longest:", lengths[-1])

LENGTH = 64
longer = sum(n > LENGTH for n in lengths)
print(f"longer than {LENGTH} tokens: {longer} of {len(lengths)} messages")

train_ids, train_labels = encode_messages(train, encoding, LENGTH)
val_ids, val_labels = encode_messages(val, encoding, LENGTH)
test_ids, test_labels = encode_messages(test, encoding, LENGTH)
print("train_ids:", tuple(train_ids.shape), " train_labels:", tuple(train_labels.shape))
print("first 10 labels:", train_labels[:10].tolist(), " (0 = ham, 1 = spam)")"""
            ),
            saw(
                "The middle message has 32 tokens and the longest 87. With `LENGTH = 64` only 11 of the 1,045 messages get cut, and shorter rows make every step faster. "
                "The training set is now one table of shape `(1045, 64)`, with 1,045 labels: 0 for ham, 1 for spam."
            ),
            q(
                "Step 4: how do we swap the head?",
                "We load GPT-2 into our class ([08_01](08_01_loading_gpt2.ipynb), now one call), and replace `model.output_head` with a new `nn.Linear(768, 2)`.\n\n"
                "The model still gives scores for **every** token of the message. Which ones do we read? The **last** token's. Because of the causal mask ([03_03](../03_attention/03_03_hiding_the_future.ipynb)), a token only sees what comes before it, so the last one is the only one that has seen the whole message.\n\n"
                "```\n"
                "ids     (1, 64)        one message, 64 tokens\n"
                "model   (1, 64, 2)     2 scores at every token\n"
                "last    (1, 2)         the 2 scores at the last token:  [ham score, spam score]\n"
                "```",
            ),
            code(CH8_GPT2),
            code(
                r"""from torch import nn

print("head before:", model.output_head)
torch.manual_seed(123)
model.output_head = nn.Linear(768, 2)  # 2 scores: ham, spam. Random for now.
print("head after: ", model.output_head)

one = train_ids[:1]  # one message, shape (1, 64)
with torch.no_grad():
    scores = model(one)
print()
print("scores at every token:", tuple(scores.shape))
print("scores at the last one:", scores[:, -1, :])"""
            ),
            saw(
                "The head went from `768 → 50257` to `768 → 2`, and the output from one score per vocabulary token to two scores per position: `(1, 64, 2)`. The last position's two numbers are the model's answer. Right now they mean nothing, because the head is random."
            ),
            q(
                "Step 5: what do we freeze, and why?",
                "**Freezing** a weight means telling PyTorch not to change it: `requires_grad = False`. No nudge is worked out for it, and the optimizer leaves it alone.\n\n"
                "We freeze almost everything. The adult keeps what they know. We train only three parts at the top:\n\n"
                "```\n"
                "token + position tables      frozen\n"
                "blocks 0 … 10                frozen     ← everything GPT-2 knows about English\n"
                "block 11 (the last one)      TRAINED\n"
                "final layer norm             TRAINED\n"
                "new head (768 → 2)           TRAINED    ← starts random\n"
                "```\n\n"
                "Why not train only the head? You can, and it works a little. Raschka's book reports clearly better results when the last block may move too, because then it can reshape its 768 numbers for this one job. "
                "And why not train everything? It's slower, and with only 1,045 examples a fully free model can forget what it knew.\n\n"
                "There's a speed bonus too. Working out nudges runs backwards from the loss, and it can stop at the first frozen block. So we pay for the backward pass of one block, not twelve.\n\n"
                "👉 **Predict first:** one block of twelve, plus two small parts. What share of all the weights will be trained?",
            ),
            code(
                r"""for parameter in model.parameters():
    parameter.requires_grad = False  # freeze everything...
for part in (model.blocks[-1], model.final_norm, model.output_head):
    for parameter in part.parameters():
        parameter.requires_grad = True  # ...then unfreeze three parts

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"trained: {trainable:>11,}")
print(f"all:     {total:>11,}")
print(f"share:   {trainable / total:.1%}")"""
            ),
            saw(
                "7,090,946 of 124,441,346, so 5.7%. (The total is smaller than the 163 million of 08_01, because the big 50,257-row head is gone and a 2-row one took its place.)\n\n"
                "`make_classifier` in `finetune.py` is steps 4 and 5 in one call."
            ),
            q(
                "Step 6: how good is it before any training?",
                "We need a number to beat. **Accuracy** is the share of messages the model gets right. The model's answer is whichever of its two scores is higher (`argmax`).\n\n"
                "👉 **Predict first:** the head is random and the data is balanced. What accuracy do you expect?",
            ),
            code(
                r"""from homer_ai.finetune import accuracy

before = accuracy(model, val_ids, val_labels)
print(f"validation accuracy, untrained head: {before:.1%}")"""
            ),
            saw(
                "54.4%, which is coin-flipping on 149 messages. A random head knows nothing, as it should."
            ),
            q(
                "Step 7: what does the training loop look like?",
                "It's the loop of chapter 5: guess, measure the loss, work out the nudges, nudge. Only the first two lines changed.\n\n"
                "```python\n"
                "scores = model(batch_ids)[:, -1, :]            # (8, 2): the last token's 2 scores\n"
                "loss = F.cross_entropy(scores, batch_labels)   # the right answer is 0 or 1\n"
                "optimizer.zero_grad()\n"
                "loss.backward()\n"
                "optimizer.step()\n"
                "```\n\n"
                'The loss is still cross-entropy ([05_01](../05_training_on_the_odyssey/05_01_loss_and_epochs.ipynb)). Before, it asked "how much chance did you give the right next token, out of 50,257?". Now it asks "how much chance did you give the right label, out of 2?". Pure guessing gives `ln(2) = 0.693`.\n\n'
                "`train_classifier` wraps that loop: batches of 8 messages, shuffled each epoch, AdamW with a learning rate of 0.00005 (small, because we're adjusting a trained model, not building one). "
                "We run 3 epochs, which is 390 steps. It takes about 65 seconds on my laptop.\n\n"
                "👉 **Predict first:** the loss starts near 0.693 or above. Where will it be after 3 epochs?",
            ),
            code(
                r"""import time

from homer_ai.finetune import train_classifier

started = time.time()
history = train_classifier(
    model, train_ids, train_labels, epochs=3, lr=5e-5, batch_size=8, report_every=30
)
train_seconds = time.time() - started
print(f"training took {train_seconds:.0f} seconds")"""
            ),
            saw(
                "The loss starts above 1 (a random head can be confidently wrong, which costs more than guessing), drops under 0.693 within the first epoch, and ends near 0.15. On my laptop: about 65 seconds."
            ),
            q(
                "Step 8: did it learn? The three accuracies",
                "Now the exam. Training accuracy shows whether it learned the examples. Validation accuracy shows whether that carries over to new messages. And the test accuracy is the one we report, because we never looked at those 300 messages while working.",
            ),
            code(
                r"""results = {
    "training (first 300)": accuracy(model, train_ids[:300], train_labels[:300]),
    "validation": accuracy(model, val_ids, val_labels),
    "test": accuracy(model, test_ids, test_labels),
}
for name, value in results.items():
    print(f"{name:22} {value:.1%}")"""
            ),
            saw(
                "On my laptop: training 97.0%, validation 94.6%, test **93.7%**. From a coin flip to 281 of 300 right, in about a minute, with 94% of the model frozen.\n\n"
                "Training is a bit higher than the other two. That's the gap of [05_03](../05_training_on_the_odyssey/05_03_memorising_vs_understanding.ipynb): it knows its own examples a little better than new ones. It's small, so the model mostly learned the job and not the examples.\n\n"
                "For scale: Raschka's book trains 5 epochs on messages padded to 120 tokens and reports about 95.7% on its test set. We train less and cut at 64 tokens, and land two points lower."
            ),
            q(
                "Which mistakes does it make?",
                "One number hides what kind of mistakes they are, and for a spam filter that matters. A real message thrown into the spam folder is worse than one spam getting through.\n\n"
                "A **confusion table** counts both kinds. Each row is what the message really was, each column what the model said.",
            ),
            code(
                r"""from homer_ai.finetune import class_scores

with torch.no_grad():
    batches = [test_ids[i : i + 50] for i in range(0, 300, 50)]
    guesses = torch.cat([class_scores(model, b).argmax(dim=-1) for b in batches])

print(f"{'':14}{'said ham':>10}{'said spam':>11}")
for truth, name in enumerate(["really ham", "really spam"]):
    row = [((test_labels == truth) & (guesses == said)).sum().item() for said in (0, 1)]
    print(f"{name:14}{row[0]:>10}{row[1]:>11}")

print()
wrong = (guesses != test_labels).nonzero().flatten().tolist()
for i in wrong[:4]:
    label, text = test[i]
    print(f"really {label:4} | {text[:80]}")"""
            ),
            saw(
                "On my laptop the 19 mistakes split like this:\n\n"
                "```\n"
                "                said ham  said spam\n"
                "really ham           142          7\n"
                "really spam           12        139\n"
                "```\n\n"
                "It called a real message spam 7 times, and let spam through 12 times. Look at the wrong ones it printed. They're the odd cases: spam written in a chatty voice (`Did you hear about the new \"Divorce Barbie\"?`), and a real message that sounds like a chain letter (`Send this to ur friends and receive something…`). I'd have hesitated on some of those myself."
            ),
            q(
                "Can I try my own messages?",
                "Yes. Encode the text the same way (same length, same padding), and read the last token's scores. `softmax` turns the two scores into chances that add up to 1.",
            ),
            code(
                r"""my_messages = [
    "You are a winner! You have been specially selected to receive $1000 cash. Call!",
    "Hey, are we still on for dinner tonight? Let me know.",
    "Tell me, O Muse, of that ingenious hero who travelled far and wide.",
]
ids, _ = encode_messages([("ham", text) for text in my_messages], encoding, LENGTH)
with torch.no_grad():
    chances = torch.softmax(class_scores(model, ids), dim=-1)
for text, (_ham, spam) in zip(my_messages, chances.tolist(), strict=True):
    print(f"spam {spam:5.1%} | {text[:70]}")"""
            ),
            saw(
                "The prize message gets 64.9% spam and the dinner message 0.2%. The line from Butler's Odyssey gets 0.4%, so ham. Keep in mind what that means, though. The model never saw anything like Homer in training, and a classifier always answers, even about text far from its training data. Here the answer happens to be sensible."
            ),
            *questions(
                [
                    (
                        "Why read the last token and not the average of all of them?",
                        "Both can work. The last token is the simple choice for a GPT, because it's the only one that saw the whole text. In our padded rows the last token is usually a filler, and that's fine: the filler sits after the message, so it has seen all of it, and training teaches the last block to gather the answer there.",
                    ),
                    (
                        "Why is the learning rate so small?",
                        "0.00005, against 0.001 when we trained the Homer GPT from scratch. GPT-2's weights are already good. Big steps would knock them about and lose what they hold. Small steps adjust them.",
                    ),
                    (
                        "Could I get more than 93.7%?",
                        "Probably, by training longer. Change `epochs=3` to `epochs=5`, or set `LENGTH` to 87 so no message is cut, and run the notebook again. Each makes it slower. Judge by the validation accuracy while you try things, and look at the test accuracy once at the end.",
                    ),
                    (
                        "Is 93.7% good for a spam filter?",
                        "For a minute of training, yes. For a product, no: on this data set, much simpler methods that only count words are reported at about 97%. Fine-tuning a language model pays off on jobs where the meaning of a sentence matters more than which words are in it.",
                    ),
                    (
                        "Where is the trained classifier saved?",
                        "Nowhere. It lives in this notebook's memory, and training it again takes a minute. [08_04](08_04_lora.ipynb) saves a fine-tuned model, to the cache folder outside the project, and shows how small such a file can be.",
                    ),
                ]
            ),
            recap(
                [
                    "Classification sorts an input into one of a few classes. A GPT does it with a new output head (768 → 2), read at the last token.",
                    "Labelled data needs care first: balance the classes, split into training, validation and test, pad to one length.",
                    "Freezing locks most weights. We trained the last block, the final norm and the head: 5.7% of the model.",
                    "Three epochs (about a minute on a laptop CPU) took accuracy from a coin flip to 93.7% on unseen messages.",
                    "A confusion table shows which kind of mistake a classifier makes.",
                ],
                "[08_03: Instruction fine-tuning](08_03_instructions.ipynb).",
            ),
        ],
    )

    # ---------------------------------------------------------------- 08_03
    save(
        FOLDER + "08_03_instructions.ipynb",
        [
            title(
                "08_03 Instruction fine-tuning (small)",
                "chapter 7 (fine-tuning to follow instructions)",
            ),
            where(
                None,
                [
                    "data/homer/odyssey/english/",
                    "data/homer/iliad/english/",
                    "src/homer_ai/pretrained.py",
                ],
                "**after the main course**: teaching the adult to take requests",
            ),
            learn(
                [
                    "Why a plain GPT doesn't answer requests, and what instruction fine-tuning changes",
                    "What an instruction format is, and how to build a small instruction set from Butler's Homer",
                    "How to batch texts of different lengths so that padding doesn't count in the loss",
                    "How to score written answers, and what GPT-2 small can and can't learn in a minute",
                ]
            ),
            words(
                [
                    (
                        "instruction fine-tuning",
                        "fine-tuning on examples of a request followed by a good answer, so the model learns to answer and not just to continue",
                        '`What is the first word…` → `The first word is "Tell".`',
                    ),
                    (
                        "instruction format",
                        "one fixed layout for every example: where the request goes, where the input goes, where the answer starts",
                        "`### Instruction:` … `### Response:`",
                    ),
                    (
                        "ignore index",
                        "a special target value, -100, that tells the loss to skip that position",
                        "the targets of padding tokens",
                    ),
                    (
                        "exact match",
                        "a strict score: an answer counts only if it's letter for letter the expected one",
                        "27 of 30",
                    ),
                ]
            ),
            *background(),
            q(
                "GPT-2 writes English. Why can't I just ask it something?",
                "Because it was never taught that a question wants an answer. It learned one thing: what text usually comes next on a web page. Give it a question and it continues the page. Maybe with another question, maybe with something that only looks like an answer.\n\n"
                '**Instruction fine-tuning** fixes that with examples. Each one is a request and a good answer, always in the same layout. After enough of them, "what usually comes next" after a request *is* an answer.\n\n'
                "In the picture of [08_00](08_00_chapter_overview.ipynb): the adult speaks English, but nobody ever told them that at this desk people come with requests. We teach them to take requests politely. That's the same idea that turned plain language models into chat assistants, done here at toy size.",
            ),
            q(
                "What will our instructions be?",
                "Three small jobs on single sentences from Butler's English Homer in `data/homer/`:\n\n"
                "```\n"
                'first   What is the first word of this sentence?                      → The first word is "Telemachus".\n'
                'last    What is the last word of this sentence?                       → The last word is "did".\n'
                "poem    Which poem is this sentence from, the Iliad or the Odyssey?   → The Odyssey.\n"
                "```\n\n"
                "I chose them for one reason: a program can work out every right answer from our own files. The first and last word are in the sentence, and the poem is the folder the sentence came from. So nothing in the data is made up.\n\n"
                "They also differ in difficulty, which is what makes this interesting. For `first` the answer sits right at the start of the input. For `last` it has to be found at the end. And `poem` needs knowledge: who appears in which poem.\n\n"
                "👉 **Predict first:** which of the three will GPT-2 small learn in one minute of training? All, some, none?",
            ),
            step_intro(),
            q(
                "Step 1: where do the sentences come from?",
                "From the 48 English book files. We cut the text into sentences at `.`, `!` and `?`, and keep the ones that are 6 to 16 words long and have no quotation marks (so each is short and stands alone). Then we take 390 from each poem, in a random order fixed by a seed.",
            ),
            code(
                r"""import random
import re

from homer_ai.lessons import HOMER, load_book_files


def sentences(poem):
    found = []
    for book in load_book_files(HOMER / poem / "english"):
        for sentence in re.split(r"(?<=[.!?])\s+", book.replace("\n", " ")):
            n_words = len(sentence.split())
            plain = not set(sentence) & set('"“”')  # no quotation marks
            if 6 <= n_words <= 16 and sentence[0].isupper() and plain:
                found.append(sentence.strip())
    return found


rng = random.Random(123)
pairs = []  # (poem, sentence)
for poem in ("odyssey", "iliad"):
    found = sentences(poem)
    print(f"{poem:8} {len(found)} usable sentences, e.g. {found[0]!r}")
    rng.shuffle(found)
    pairs += [(poem, sentence) for sentence in found[:390]]
rng.shuffle(pairs)
print(len(pairs), "sentences in all, half from each poem")"""
            ),
            saw(
                "397 usable sentences in the Odyssey and 586 in the Iliad. We keep 390 of each, so 780, and the `poem` job is balanced: guessing gives 50%."
            ),
            q(
                "Step 2: how does a sentence become an instruction example?",
                "Each sentence gets one of the three jobs in turn, and a function writes the right answer. An example is a small dict: the instruction, the input (the sentence), and the output (the answer).",
            ),
            code(
                r"""TASKS = {
    "first": "What is the first word of this sentence?",
    "last": "What is the last word of this sentence?",
    "poem": "Which poem is this sentence from, the Iliad or the Odyssey?",
}


def right_answer(task, sentence, poem):
    found = re.findall(r"[A-Za-z'’-]+", sentence)  # the words, without punctuation
    if task == "first":
        return f'The first word is "{found[0]}".'
    if task == "last":
        return f'The last word is "{found[-1]}".'
    return f"The {poem.capitalize()}."


examples = []
for i, (poem, sentence) in enumerate(pairs):
    task = list(TASKS)[i % 3]
    examples.append(
        {
            "task": task,
            "instruction": TASKS[task],
            "input": sentence,
            "output": right_answer(task, sentence, poem),
        }
    )

test_examples, train_examples = examples[:90], examples[90:]
print(len(train_examples), "to train on,", len(test_examples), "kept for the test")
for example in test_examples[:3]:
    print()
    print("input :", example["input"])
    print("job   :", example["instruction"])
    print("answer:", example["output"])"""
            ),
            saw(
                "690 examples to train on and 90 for the test, 30 of each job. The three printed ones show one job each, with the answer the program worked out."
            ),
            q(
                "Step 3: what's the instruction format?",
                "The model reads plain text, so the three parts of an example must become one string, laid out the same way every time. We use the layout Raschka's book uses (it comes from Stanford's Alpaca project):\n\n"
                "```\n"
                "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n"
                "\n"
                "### Instruction:\n"
                "What is the first word of this sentence?\n"
                "\n"
                "### Input:\n"
                "Telemachus saw her long before any one else did.\n"
                "\n"
                "### Response:\n"
                'The first word is "Telemachus".<|endoftext|>\n'
                "```\n\n"
                "Everything down to `### Response:` is the **prompt**. For training we add the answer and `<|endoftext|>`. That last token teaches the model to stop: after an answer, the next thing is the end.\n\n"
                "When we ask a question later, we give only the prompt and let the model write on from `### Response:`.",
            ),
            code(
                r"""def prompt_of(example):
    return (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request."
        f"\n\n### Instruction:\n{example['instruction']}"
        f"\n\n### Input:\n{example['input']}"
        "\n\n### Response:\n"
    )


print(prompt_of(train_examples[0]) + train_examples[0]["output"])"""
            ),
            q(
                "Step 4: what does GPT-2 answer before any fine-tuning?",
                "Let's ask it the three test questions above. `reply` gives the model the prompt, lets it write up to 14 new tokens with greedy writing, and returns only the new part.",
            ),
            code(CH8_GPT2),
            code(
                r"""from homer_ai.pretrained import continue_text

EOT = 50256  # <|endoftext|>


def reply(model, example):
    prompt = prompt_of(example)
    text = continue_text(model, encoding, prompt, 14, stop_id=EOT)
    return text[len(prompt) :].strip()


for example in test_examples[:3]:
    print("expected:", example["output"])
    print("GPT-2:   ", repr(reply(model, example)))
    print()"""
            ),
            saw(
                "It doesn't answer. For the first one it talks *about* the question (`The first word of this sentence is the first word of the response`), and for the other two it just repeats pieces of the input. It read the request as the start of a page and continued the page."
            ),
            q(
                "Step 5: how do examples of different lengths go into one batch?",
                "Padding again, as in [08_02](08_02_spam_classifier.ipynb), but now the job is next-token guessing, so the targets are the inputs moved one token on ([02_04](../02_text_to_numbers/02_04_sliding_window.ipynb)).\n\n"
                "There's one catch. We don't want the model to be graded on guessing filler tokens. PyTorch's `cross_entropy` skips every position whose target is **-100** (its ignore index). So the targets get -100 wherever there's nothing real to guess:\n\n"
                "```\n"
                "a short text, 5 tokens, padded to 8:\n"
                "inputs :  t1   t2   t3   t4   t5   pad   pad   pad\n"
                "targets:  t2   t3   t4   t5   EOT  -100  -100  -100\n"
                "                              ↑ learn to stop    ↑ not graded\n"
                "```\n\n"
                "Read it column by column: after `t5` the right next token is `<|endoftext|>`, and after that nothing counts.",
            ),
            code(
                r"""encoded = [encoding.encode(prompt_of(e) + e["output"]) for e in train_examples]
sizes = [len(row) for row in encoded]
print("tokens per example: shortest", min(sizes), " longest", max(sizes))


def make_batch(rows):
    length = max(len(row) for row in rows) + 1  # + 1 for <|endoftext|>
    inputs = torch.full((len(rows), length), EOT)
    targets = torch.full((len(rows), length), -100)
    for r, row in enumerate(rows):
        row = row + [EOT]
        inputs[r, : len(row)] = torch.tensor(row)
        targets[r, : len(row) - 1] = torch.tensor(row[1:])
    return inputs, targets


inputs, targets = make_batch([[11, 12, 13, 14, 15], [21, 22, 23]])  # two toy texts
print("inputs :", inputs.tolist())
print("targets:", targets.tolist())"""
            ),
            saw(
                "Two toy texts of 5 and 3 tokens:\n\n"
                "```\n"
                "inputs : [11, 12, 13, 14, 15, 50256]   [21, 22, 23, 50256, 50256, 50256]\n"
                "targets: [12, 13, 14, 15, 50256, -100] [22, 23, 50256, -100, -100, -100]\n"
                "```\n\n"
                "Each targets row is its inputs row moved one on, it ends with 50256 (stop), and the rest is -100. The real examples are about 70 tokens long, mostly the fixed layout around a short sentence."
            ),
            q(
                "Step 6: how do we train?",
                "With the loop of [05_07](../05_training_on_the_odyssey/05_07_train_the_gpt.ipynb), unchanged: a batch of 8 random examples, the scores flattened, cross-entropy against the targets.\n\n"
                "One thing is different from 08_02: **nothing is frozen**. Writing an answer uses the whole model, bottom to top, so we let all 163 million numbers move. That makes each step slow, about 0.7 seconds on my laptop. We do 80 steps, so the model sees 640 examples, a little under one pass over the training set. It takes about a minute.\n\n"
                "👉 **Predict first:** the loss is the average over every token of the example, and most of those tokens are the fixed layout. Will the loss fall fast or slowly?",
            ),
            code(
                r"""import time

import torch.nn.functional as F

optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.1)
picker = torch.Generator().manual_seed(123)

model.train()
started = time.time()
for step in range(1, 81):
    chosen = torch.randint(0, len(encoded), (8,), generator=picker).tolist()
    inputs, targets = make_batch([encoded[i] for i in chosen])
    loss = F.cross_entropy(model(inputs).flatten(0, 1), targets.flatten())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if step % 10 == 0:
        print(f"step {step:2} | loss {loss.item():.3f} | {time.time() - started:4.0f}s")
model.eval()
print(f"training took {time.time() - started:.0f} seconds")"""
            ),
            saw(
                "Fast at first: about 1.24 at step 10 and 0.86 by step 30, then slowly on to about 0.63 at step 80. The quick drop is the model learning the layout, which is the same in every example and easy to predict. The slow part is the sentences themselves, which are new text every time."
            ),
            q(
                "Step 7: does it answer now?",
                "The same three test questions as in step 4.",
            ),
            code(
                r"""for example in test_examples[:3]:
    print("expected:", example["output"])
    print("model:   ", repr(reply(model, example)))
    print()"""
            ),
            saw(
                'Now it answers, in the right form, and it stops after the answer: `The first word is "As".`, `The last word is "Ulysses".`, `The Odyssey.` That part, the politeness, it learned in a minute.\n\n'
                "But only the first of the three is *right*. The last word was `rose`, and the sentence is from the Iliad. Three examples can't tell us much, so let's count."
            ),
            q(
                "Step 8: how often is it right?",
                "We score all 90 test examples with **exact match**: an answer counts only if it's letter for letter the expected one. That's strict, and for answers this short it's fair.",
            ),
            code(
                r"""right = {task: 0 for task in TASKS}
for example in test_examples:
    right[example["task"]] += reply(model, example) == example["output"]

for task, n in right.items():
    print(f"{task:6} {n:>2} of 30 exactly right")"""
            ),
            saw(
                "On my laptop:\n\n"
                "```\n"
                "first  28 of 30\n"
                "last   15 of 30\n"
                "poem   15 of 30\n"
                "```\n\n"
                "This is the honest picture of a small model.\n\n"
                "- **first** works. The answer is the first thing after `### Input:`, so the model only has to copy from a fixed place.\n"
                "- **last** works half the time. It has to find where the sentence ends and copy the word before the full stop. It learned the form of the answer, and often quotes a word from near the end, not the last one.\n"
                "- **poem** is at 15 of 30, and guessing gives 15. It learned nothing here. It says `The Iliad.` or `The Odyssey.` in perfect form, with no idea which. To get this right it would need to know that Ulysses and Telemachus belong to one poem and Achilles and Hector to the other, and 230 examples in one minute didn't teach it that.\n\n"
                "So instruction fine-tuning taught the *behaviour* quickly: take a request, answer in the expected form, stop. It didn't add knowledge or skill beyond what's easy. That's the limit of the adult picture I mentioned in 08_00: a person who understood the request wouldn't answer the poem question with a confident coin flip."
            ),
            q(
                "What does it do with a request it never saw?",
                "Our training set has exactly three instructions. Let's ask a fourth.",
            ),
            code(
                r"""new_request = {
    "instruction": "How many words are in this sentence?",
    "input": "Tell me, O Muse, of that ingenious hero.",
}
print(repr(reply(model, new_request)))"""
            ),
            saw(
                "`The hero.` Short, confident, in the style of the answers it learned, and not an answer to the question. It didn't learn to follow instructions in general. It learned three patterns, and here it falls somewhere between them. "
                "Models that handle new requests were trained on a great many different instructions (the set in Raschka's book has 1,100 examples of many kinds; real assistants use far more), and they're much bigger."
            ),
            *questions(
                [
                    (
                        "Why not use the instruction set from Raschka's book?",
                        "I looked. His repository (`rasbt/LLMs-from-scratch`) has `instruction-data.json` with 1,100 examples, and the repository's licence is Apache 2.0, with a line added that leaves out the book's own text and pictures. "
                        'That probably makes the data file fine to reuse, but "probably" isn\'t what I want for a course others will copy. So this notebook builds its own set from the Butler translations we already ship, where every answer is worked out by a few lines of code you can read above.',
                    ),
                    (
                        "Would more training fix the weak results?",
                        "Some of them. More steps would likely push `last` up, since the answer is in the input. `poem` needs the model to learn facts about names from 230 examples, which is slow going. "
                        "You can try: run the training cell again for 80 more steps (the optimizer keeps its state) and score again. The book uses a model three times bigger and a more varied instruction set; this notebook stays small so that it runs in a couple of minutes.",
                    ),
                    (
                        "Why did we unfreeze everything here, but not in 08_02?",
                        "A classifier only needs the last block to gather one answer. Writing an answer token by token uses every layer, and the new behaviour (answer, then stop) touches all of them. The price is speed: 0.7 seconds a step, against 0.17 in 08_02.",
                    ),
                    (
                        "Is exact match too strict?",
                        "For free-form answers it would be. `It is the Odyssey.` is right but not an exact match. Bigger projects score answers with a second model, or by hand. Our answers have one fixed form, and the model learned that form, so exact match is fair here.",
                    ),
                ]
            ),
            recap(
                [
                    "A plain GPT continues text. Instruction fine-tuning shows it requests followed by answers, in one fixed format, until answering is what comes next.",
                    "We built 780 examples from Butler's Homer with answers a program can work out, so nothing is invented.",
                    "Padding targets are set to -100 so the loss skips them; `<|endoftext|>` after each answer teaches the model to stop.",
                    "In one minute, GPT-2 small learned the form of the answers and the easy job well (28 of 30). The job that needs knowledge stayed at chance (15 of 30).",
                    "It learned three patterns, not instructions in general.",
                ],
                "[08_04: LoRA](08_04_lora.ipynb).",
            ),
        ],
    )

    # ---------------------------------------------------------------- 08_04
    save(
        FOLDER + "08_04_lora.ipynb",
        [
            title(
                "08_04 LoRA: fine-tuning with far fewer numbers",
                "appendix E (parameter-efficient fine-tuning with LoRA)",
            ),
            where(
                None,
                [
                    "src/homer_ai/lora.py",
                    "src/homer_ai/finetune.py",
                    "src/homer_ai/gpt.py",
                ],
                "**after the main course**: the adult takes notes on a small notepad",
            ),
            learn(
                [
                    "What a low-rank table is, with a 4 × 4 example you can check by hand",
                    "How to write a LoRA layer in about 20 lines",
                    "How many numbers LoRA trains, compared with 08_02",
                    "What LoRA gives (small files, an untouched model) and what it costs on a CPU (time)",
                ]
            ),
            words(
                [
                    (
                        "LoRA",
                        "low-rank adaptation: the model's weights stay frozen, and beside each chosen layer two thin tables are trained, whose product is added to the layer's result",
                        "`W` frozen, `B @ A` trained",
                    ),
                    (
                        "rank",
                        "here: the width of the two thin tables. A bigger rank can express a bigger change and costs more numbers",
                        "rank 8",
                    ),
                    (
                        "adapter",
                        "the small set of trained numbers that turns a frozen base model into a model for one job",
                        "a 5.4 MB file next to GPT-2's 548 MB",
                    ),
                    (
                        "merge",
                        "adding `B @ A` into `W` after training, so the layer is one plain table again",
                        "`W + scale * B @ A`",
                    ),
                ]
            ),
            *background(),
            q(
                "What problem does LoRA solve?",
                "In [08_02](08_02_spam_classifier.ipynb) we trained 7 million numbers, and the result was a changed last block: 28 MB that belong to the spam job only. In [08_03](08_03_instructions.ipynb) we trained all 163 million. "
                "Now think of a company with 50 jobs for one model. That's 50 changed copies. With today's models, each copy is many gigabytes.\n\n"
                "**LoRA** (low-rank adaptation) leaves every weight of the model exactly as it is. Beside a layer it puts two thin tables, and trains only those.\n\n"
                "In the picture of [08_00](08_00_chapter_overview.ipynb): the adult takes notes on a small notepad instead of rewiring their brain. One brain, and a notepad per job.",
            ),
            q(
                "How can two thin tables stand in for a big one?",
                "Fine-tuning a layer changes its weight table `W` into `W + ΔW`, where `ΔW` is the change, a table of the same shape. For a 768 × 768 layer, that's 589,824 numbers to learn.\n\n"
                "LoRA's bet is that the change is simple enough to be written as a product of two thin tables:\n\n"
                "```\n"
                "ΔW  =  B @ A          B: (768, r)     A: (r, 768)      r is the rank, a small number\n"
                "```\n\n"
                "Here it is small enough to do by hand, with 4 in place of 768 and rank 1:\n\n"
                "```\n"
                "B (4 × 1)      A (1 × 4)                 B @ A (4 × 4)\n"
                "  1                                        1  0  2  0\n"
                "  2        @   1  0  2  0       =          2  0  4  0\n"
                "  0                                        0  0  0  0\n"
                "  3                                        3  0  6  0\n"
                "```\n\n"
                "Eight numbers made a table of sixteen. Every row of the result is the row `A`, times one number from `B`. So it can't be *any* 4 × 4 table, only a simple, repetitive one. That's what **low rank** means.\n\n"
                "👉 **Predict first:** for 768 × 768 and rank 8, how many numbers are in `A` and `B` together?",
            ),
            step_intro(),
            q(
                "Step 1: is the hand example right, and what about 768?",
                "`@` is matrix multiplication ([A_03](../A_toolbox/A_03_dot_product.ipynb)).",
            ),
            code(
                r"""import torch

B = torch.tensor([[1.0], [2.0], [0.0], [3.0]])  # (4, 1)
A = torch.tensor([[1.0, 0.0, 2.0, 0.0]])  # (1, 4)
print(B @ A)
print("numbers in A and B:", A.numel() + B.numel())
print("numbers in the big table:", (B @ A).numel())
print()
for rank in (1, 8, 64):
    thin = 2 * rank * 768
    share = thin / 768**2
    print(f"768 x 768 at rank {rank:2}: {thin:>7,} numbers, {share:6.1%} of 589,824")"""
            ),
            saw(
                "The same 4 × 4 table as above, from 8 numbers. For a real 768 × 768 layer, rank 8 needs 12,288 numbers: 2.1% of the full table."
            ),
            md("The layer we're about to write, as a picture:"),
            code(DIAGRAM_LORA),
            q(
                "Step 2: what does a LoRA layer look like in code?",
                "It wraps an existing `nn.Linear`. The old layer is frozen. `A` and `B` are new, and their result is added to the old layer's result.\n\n"
                "```\n"
                "out = linear(x)  +  scale * (x @ A.T @ B.T)\n"
                "      └ frozen ┘    └──── trained ────────┘\n"
                "```\n\n"
                "Two details matter:\n\n"
                "- **`B` starts at zero.** Then `B @ A` is all zeros, and the wrapped layer gives exactly what the old layer gave. Fine-tuning starts from GPT-2 as it is, not from a disturbed copy. (`A` starts with small random numbers. If both were zero, neither could ever get a nudge.)\n"
                "- **`scale = alpha / rank`** sets how strongly the change counts. A common habit is `alpha = 2 × rank`, so scale 2.",
            ),
            code(
                r"""import math

from torch import nn


class LoRALinear(nn.Module):
    def __init__(self, linear, rank=8, alpha=16.0):
        super().__init__()
        self.linear = linear
        for parameter in linear.parameters():
            parameter.requires_grad = False  # frozen: the optimizer leaves it alone
        self.A = nn.Parameter(torch.empty(rank, linear.in_features))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))  # small random numbers
        self.B = nn.Parameter(torch.zeros(linear.out_features, rank))
        self.scale = alpha / rank

    def forward(self, x):
        return self.linear(x) + self.scale * (x @ self.A.T @ self.B.T)


torch.manual_seed(0)
plain = nn.Linear(768, 768)
wrapped = LoRALinear(plain, rank=8)
x = torch.randn(2, 768)
print("same answer as the plain layer at the start:", torch.equal(wrapped(x), plain(x)))
for name, parameter in wrapped.named_parameters():
    state = "trained" if parameter.requires_grad else "frozen"
    shape = str(tuple(parameter.shape))
    print(f"{name:14} {shape:12} {parameter.numel():>8,}  {state}")"""
            ),
            saw(
                "At the start the wrapped layer answers exactly like the plain one. Inside it, the 768 × 768 weight and its bias are frozen, and `A` (8 × 768) and `B` (768 × 8) are trained: 6,144 numbers each.\n\n"
                '`nn.Parameter` is how you tell PyTorch "this tensor is a weight of my layer", so that the optimizer finds it. The same class is in `lora.py`, with a `merged()` method we use at the end.'
            ),
            q(
                "Step 3: how do we put it into GPT-2?",
                "Same start as [08_02](08_02_spam_classifier.ipynb): the spam data, GPT-2 in our class, a new 2-score head. But this time **every** GPT-2 weight is frozen, the last block too. Only the new head is trained in full, because it starts random and is tiny.\n\n"
                "Then `apply_lora` walks through the model and wraps every `nn.Linear` it finds inside the blocks: query, key, value, the output projection and the two feed-forward layers. That's 6 layers in each of 12 blocks.\n\n"
                "👉 **Predict first:** 08_02 trained 7,090,946 numbers. Will this be more or fewer, with adapters in all 12 blocks?",
            ),
            code(CH8_SPAM),
            code(CH8_GPT2),
            code(
                r"""from homer_ai.finetune import accuracy, make_classifier, train_classifier
from homer_ai.lora import apply_lora, count_parameters

make_classifier(model)  # the new head, as in 08_02 ...
for parameter in model.parameters():
    parameter.requires_grad = False  # ... but now freeze ALL of GPT-2
for parameter in model.output_head.parameters():
    parameter.requires_grad = True  # the head is trained in full

wrapped_layers = apply_lora(model, rank=8, alpha=16, skip=("output_head",))
trainable, total = count_parameters(model)
print("layers wrapped:", wrapped_layers)
print(f"trained:        {trainable:>11,}")
print(f"all:            {total:>11,}")
print(f"share:          {trainable / total:.1%}")
print()
print(model.blocks[0].attention.query)"""
            ),
            saw(
                "72 layers wrapped, and 1,328,642 trained numbers: about 1.1% of the model, and 5 times fewer than in 08_02, even though now all 12 blocks can adapt and not only the last. "
                "(The total grew a little, to 125,768,450, because the thin tables were added.)\n\n"
                "The last lines show one wrapped layer: a `LoRALinear` with the old `Linear` inside."
            ),
            q(
                "Step 4: does it learn the spam job?",
                "Same data, same loop, same batch size as 08_02. Two things differ. The learning rate is higher, 0.0003 against 0.00005, because `B` starts at zero and has a long way to go. And we train 2 epochs, not 3, because each epoch is slower here (the next question says why).\n\n"
                "👉 **Predict first:** 08_02 reached 93.7% on the test set. Will 5 times fewer numbers do worse?",
            ),
            code(
                r"""import time

started = time.time()
train_classifier(model, train_ids, train_labels, epochs=2, lr=3e-4, report_every=30)
train_seconds = time.time() - started
print(f"training took {train_seconds:.0f} seconds")"""
            ),
            code(
                r"""results = {
    "training (first 300)": accuracy(model, train_ids[:300], train_labels[:300]),
    "validation": accuracy(model, val_ids, val_labels),
    "test": accuracy(model, test_ids, test_labels),
}
for name, value in results.items():
    print(f"{name:22} {value:.1%}")"""
            ),
            saw(
                "The loss falls as in 08_02, and ends near 0.2. On my laptop the two epochs took about 117 seconds, and the accuracies are: training 97.3%, validation 95.3%, test **94.3%**.\n\n"
                "So no, not worse. With a fifth of the trained numbers it lands where 08_02 did."
            ),
            q(
                "So which is better, 08_02 or LoRA?",
                "Here are both runs from my laptop, side by side:\n\n"
                "| | 08_02: last block trained | 08_04: LoRA, rank 8 |\n|---|---|---|\n"
                "| GPT-2's own weights | last block changed | all untouched |\n"
                "| Trained numbers | 7,090,946 | 1,328,642 |\n"
                "| Epochs | 3 | 2 |\n"
                "| Training time | about 65 s | about 117 s |\n"
                "| Time per step | 0.17 s | 0.45 s |\n"
                "| Validation accuracy | 94.6% | 95.3% |\n"
                "| Test accuracy | 93.7% | 94.3% |\n"
                "| Size of what was trained | 28 MB | 5.4 MB |\n\n"
                "The accuracy is about the same. LoRA trains 5 times fewer numbers, and its result is a much smaller file. But each step took more than twice as long, and I'd rather explain that than hide it.\n\n"
                "In 08_02 only the last block was trained, so working out the nudges could stop after one block. LoRA has adapters in all 12 blocks, so the nudges have to be worked out all the way down. Fewer numbers to store doesn't mean less work per step.\n\n"
                "So on a laptop, for one job, freezing all but the last block is the better deal. LoRA pays off in other ways:\n\n"
                "- **Memory.** Every trained weight needs room for its nudge and for the optimizer's notes about it. On big models that's what fills the graphics card. Fewer trained weights, less memory.\n"
                "- **Files.** One base model, and a small adapter per job.\n"
                "- **Safety of the base.** GPT-2's weights are untouched, so nothing it knew can be lost. Remove the adapter and you have the original back.",
            ),
            q(
                "Step 5: how small is the file?",
                "We save only what was trained: the `A` and `B` tables and the head. It goes to the cache folder outside the project, next to GPT-2's file. (Fine-tuned models don't belong in git. Even small ones add up.)",
            ),
            code(
                r"""from homer_ai.pretrained import CACHE, GPT2_FILE

adapter = {name: p.detach() for name, p in model.named_parameters() if p.requires_grad}
adapter_file = CACHE / "spam_lora_adapter.pt"
torch.save(adapter, adapter_file)

print(adapter_file)
print(f"adapter: {adapter_file.stat().st_size / 1e6:6.1f} MB  ({len(adapter)} tensors)")
print(f"GPT-2:   {GPT2_FILE.stat().st_size / 1e6:6.1f} MB")"""
            ),
            saw(
                "5.4 MB for the adapter, next to 548 MB for GPT-2: about 1%. To use it elsewhere you'd load GPT-2, apply LoRA the same way, and load these 146 tensors on top."
            ),
            q(
                "Step 6: does the notepad stay forever?",
                "It doesn't have to. Once training is done, `B @ A` is just a table of numbers of the same shape as `W`. Add it in, and the layer is one plain `nn.Linear` again, with no extra work at run time. That's called **merging**.\n\n"
                "```\n"
                "W_merged = W + scale * (B @ A)\n"
                "```\n\n"
                "`merged()` in `lora.py` does that for one layer. Let's check that the merged layer answers like the wrapped one.",
            ),
            code(
                r"""layer = model.blocks[11].attention.query  # a LoRALinear, after training
merged = layer.merged()  # a plain nn.Linear

x = torch.randn(4, 768)
with torch.no_grad():
    difference = (merged(x) - layer(x)).abs().max().item()
    change = layer.scale * (layer.B @ layer.A)
print("merged layer:", merged)
print(f"biggest difference in the answers: {difference:.6f}")
print(f"typical size of a weight in W:     {layer.linear.weight.abs().mean():.4f}")
print(f"typical size of the change B @ A:  {change.abs().mean():.4f}")"""
            ),
            saw(
                "The merged layer is a plain `Linear(768 → 768)` and its answers match the wrapped layer's to about five decimal places (the tiny rest is rounding). "
                "And look at the sizes: a typical weight in `W` is about 0.087, and a typical entry of the change is about 0.0005. The notepad holds a light correction, not a new brain."
            ),
            *questions(
                [
                    (
                        "How do I choose the rank?",
                        "Start small. Ranks of 4 to 16 are common. A higher rank can express a bigger change, costs more numbers, and on a small data set like ours rarely helps. Change `rank=8` above to 2 or 32 and compare the validation accuracy.",
                    ),
                    (
                        "Why wrap all six layers of a block?",
                        "That's what Raschka's book does, and it's the simple choice. The paper that introduced LoRA wrapped only query and value, which trains fewer numbers still. Which layers matter most depends on the job.",
                    ),
                    (
                        "Is the comparison with 08_02 fair?",
                        'Mostly. Same data, same split, same seed, same batch size, same test set. Two things differ besides the method: the learning rate (each method got one that suits it) and the number of epochs (3 against 2). So read the table as "both get to about 94%", not as proof that LoRA is better. On 300 test messages, the 0.6 points between them is two messages.',
                    ),
                    (
                        "Could I use LoRA for the instruction job of 08_03?",
                        "Yes, the same `apply_lora` works there (without the `skip`, since there's no new head). It would train about 1.3 million numbers and not 163 million. Don't expect it to fix the weak results of 08_03, though. Those come from the model's size and the short training, not from the method.",
                    ),
                ]
            ),
            recap(
                [
                    "LoRA freezes the model and trains two thin tables beside each layer. Their product, `B @ A`, is a low-rank change added to the layer's result.",
                    "`B` starts at zero, so fine-tuning starts from the model exactly as it was.",
                    "On the spam job LoRA trained 1.3 million numbers against 7.1 million in 08_02, and reached about the same accuracy.",
                    "On a CPU it was slower, because the nudges are worked out through all 12 blocks. Its gains are memory, small adapter files, and a base model that stays untouched.",
                    "After training, the change can be merged into the weights, so the layer is plain again.",
                ],
                "[09_00: Answers grounded in Homer's text](../09_advanced_grounded_answers/09_00_chapter_overview.ipynb). Or go [back to the course map](../00_start_here/00_00_course_map.ipynb).",
            ),
        ],
    )
