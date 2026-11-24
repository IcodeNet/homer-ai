# ruff: noqa: E501
"""Chapter 4, Building the model."""

from .cells import (
    background,
    code,
    exercise,
    learn,
    md,
    q,
    recap,
    save,
    saw,
    step_intro,
    title,
    where,
    words,
)
from .diagrams import DIAGRAM_BLOCK, DIAGRAM_GPT, DIAGRAM_GRADIENT
from .setup_code import CH4_LOAD, CH4_ROWS


def build():
    # =====================================================================
    # 04_building_the_model/04_00_chapter_overview.ipynb
    # =====================================================================
    save(
        "04_building_the_model/04_00_chapter_overview.ipynb",
        [
            title("04_00 Building the model (overview)", "chapter 4"),
            where("model reads", ["src/homer_ai/model.py", "src/homer_ai/gpt.py"]),
            learn(
                [
                    "Why attention alone isn't a full model",
                    "The picture of a full GPT: a stack of identical blocks",
                    "What our small `model.py` has, and what the full design adds",
                    "What a config is, and why one config beats numbers scattered around the code",
                    "Which notebook to open for each part",
                ]
            ),
            words(
                [
                    (
                        "GPT",
                        "a popular design for language models: embeddings, then a stack of blocks, then a final layer that scores every token",
                        "GPT-2 (the book builds a small one)",
                    ),
                    (
                        "block",
                        "one floor of the model: an attention step and a feed-forward step, each with a layer norm before it and a shortcut around it; a GPT stacks many identical blocks",
                        "our small model has none; GPT-2 small has 12",
                    ),
                    (
                        "config",
                        "short for configuration: one small table that holds every size of the model, so the numbers live in one place",
                        '`{"emb_dim": 32}`',
                    ),
                ]
            ),
            *background(),
            q(
                "Isn't attention already a whole model?",
                "Almost, for a toy. Here's the picture for this chapter. Our `model.py` is a hut with one room, and a GPT is a tall building where every floor has the same layout. "
                "On each floor the same two things happen: the letters look back at each other (that's attention), and then each one thinks alone about what it heard. "
                "The numbers climb from floor to floor, so the first floors can pick up simple things, such as which letters often go together, and the higher floors build on those to pick up bigger patterns. "
                "One floor is called a **block**. "
                "The picture breaks in one place: in a real building an architect gives each floor its job, and here all the floors are built the same and only training makes them differ.\n\n"
                "The hut, our `model.py`, is an embedding, **one** attention step, and a last layer that makes the 137 scores:\n\n"
                "```\nletters → embedding → attention → head → 137 scores\n```\n\n"
                "The building is drawn in [docs/neural-network-building.html](../../docs/neural-network-building.html), and in the next question. "
                "And a block is more than attention. It also has a few helper parts that keep the numbers well-behaved, so that a tall stack can still learn.",
            ),
            q(
                "What does a full GPT look like?",
                'Read it from the bottom to the top, for the five letters of `ἄνδρα` ("man"). Each box is built in a notebook of this chapter, '
                "and the small grey names, like `04_01`, tell you which one. Five words appear in the picture:\n\n"
                "- **layer norm**: keeps the numbers steady\n"
                "- **attention**: looks back at the earlier letters\n"
                "- **feed-forward**: lets each letter think on its own\n"
                "- **output head**: makes one score for every token\n"
                "- the `+`, a **shortcut**: the floor's input is added back to its output\n\n"
                "Every block takes numbers in one shape and gives them back in the same shape, so blocks stack like identical floors.",
            ),
            md(
                "**Look at:** the picture from the bottom (the letters) to the top (the scores). The orange floor is the one that repeats."
            ),
            code(DIAGRAM_GPT),
            q(
                "What does our `model.py` have, and what's missing?",
                "Let's count the real parts and weights of our model first.\n\n"
                "👉 **Predict first:** how many of the parts in the picture above does `HomerGreekLLM` have?",
            ),
            code(
                """from homer_ai.model import HomerGreekLLM

model = HomerGreekLLM(137, embed_dim=32)
for name, layer in model.named_children():
    count = sum(p.numel() for p in layer.parameters())
    print(f"{name:10} {count:>6,} weights")
print(f"{'total':10} {sum(p.numel() for p in model.parameters()):>6,} weights")"""
            ),
            saw(
                "Five layers: the embedding, query, key, value and `head`. That's 11,840 weights in all. "
                "The `forward` method also builds the causal mask, which has no weights.\n\n"
                "Three words for the numbers we just counted. A **weight** is one learned number in a layer's table. A bias is the extra learned number a layer can add to each output ([01_02](../01_foundations/01_02_neural_networks.ipynb)). "
                "And a parameter is PyTorch's word for either. `.parameters()` lists all of them, so summing it counts everything the model can learn. "
                "Our layers use `bias=False`, so here weights and parameters are the same 11,840 numbers.\n\n"
                "So how far is that from a full GPT? Here's the whole comparison:\n\n"
                "| Part | Our small `model.py` | The full design |\n|---|---|---|\n"
                "| Token embedding | yes (137 × 32) | yes |\n"
                "| Position embedding | **no** ([02_06](../02_text_to_numbers/02_06_positions.ipynb)) | yes (24 × 32) |\n"
                "| Attention | one head, mask | several heads ([03_04](../03_attention/03_04_many_heads.ipynb)), mask, and an output layer |\n"
                "| Feed-forward | **no** | yes, in every block |\n"
                "| Layer norm | **no** | before attention, before feed-forward, and once at the end |\n"
                "| Shortcuts | **no** | two in every block |\n"
                "| Dropout | **no** | optional |\n"
                "| Blocks | none: just one attention step | N identical blocks |\n"
                "| Output head | yes (32 × 137) | yes |\n"
                "| Weights | 11,840 | depends on the config (next question) |"
            ),
            q(
                "What's a config, and why do we need one?",
                "A full model has many sizes: how many tokens, how many numbers per token, how many heads, how many blocks. "
                "In the book they all live in one small table, a Python dictionary called a **config**. "
                "Here's ours, for our 137 tokens.\n\n"
                "👉 **Predict first:** one line sets how long a chunk of text the model reads at once. Which key is it?",
            ),
            code(
                """GPT_CONFIG = {
    "vocab_size": 137,      # how many different tokens: our 137 characters
    "context_length": 24,   # how many tokens the model reads at once
    "emb_dim": 32,          # how many numbers describe each token
    "n_heads": 2,           # how many attention heads per block
    "n_layers": 2,          # how many blocks are stacked
    "drop_rate": 0.1,       # dropout: how often to switch off a number while training
}
for key, value in GPT_CONFIG.items():
    print(f"{key:15} = {value}")"""
            ),
            saw(
                "Six lines, six sizes. Let's take them one at a time:\n\n"
                "- `vocab_size`: the number of tokens, 137 for our Book 1 ([02_01](../02_text_to_numbers/02_01_tokens_and_vocabulary.ipynb)). It sets the size of the first table and of the last layer.\n"
                "- `context_length`: how many letters the model reads at once, 24 ([02_04](../02_text_to_numbers/02_04_sliding_window.ipynb)). It sets the number of rows in the position table.\n"
                "- `emb_dim`: the numbers per letter, 32 ([02_05](../02_text_to_numbers/02_05_token_embeddings.ipynb)). Almost every layer in the model is this wide.\n"
                "- `n_heads`: attention heads per block ([03_04](../03_attention/03_04_many_heads.ipynb)). It must divide `emb_dim`: 32 numbers split between 2 heads of 16.\n"
                "- `n_layers`: how many blocks sit on top of each other. Our `model.py` has none, GPT-2 small has 12.\n"
                "- `drop_rate`: dropout, from [03_03](../03_attention/03_03_hiding_the_future.ipynb). 0.1 means 1 number in 10 is switched off while training."
            ),
            q(
                "Why not just write the numbers where we use them?",
                "Because then the same number sits in several files, and when you change it you always forget one. "
                "We can find the repeats in our real project right away.\n\n"
                "👉 **Predict first:** do you think the `32` and the `24` each appear once, or in several places?",
            ),
            code(
                """from pathlib import Path

import homer_ai
from homer_ai import train_homer

# we only READ these files as text
folder = Path(homer_ai.__file__).parent
for path in (Path(train_homer.__file__), folder / "model.py", folder / "lessons.py"):
    for number, line in enumerate(path.read_text().splitlines(), start=1):
        if "embed_dim=32" in line or "chunk_size=24" in line or "[-24:]" in line:
            print(f"{path.name}:{number}: {line.strip()}")"""
            ),
            saw(
                "So `embed_dim=32` is written in four places (`train_homer.py`, `model.py` as a default, and twice in `lessons.py`), and the window of 24 sits on its own as `chunk_size=24`. Nothing links them. "
                "Change one and forget the others, and the model loads with the wrong shapes, or quietly reads the wrong window. "
                "(The MCP server doesn't repeat them: a checkpoint carries its own config, [05_06](../05_training_on_the_odyssey/05_06_saving_and_loading.ipynb), and the server builds the model from that.) "
                'With a config, every file reads `GPT_CONFIG["emb_dim"]`. Change it once, and everything follows. '
                "(The GPT does it this way: `GPTConfig` in `src/homer_ai/gpt.py` is the one place. The notebooks here keep their own copy of a dictionary so that each one runs alone.)\n\n"
                "*In .NET we'd call this the options pattern: settings in one object, passed around, never copied.*"
            ),
            q(
                "Which notebook do I open for each part?",
                "| Notebook | Question it answers |\n|---|---|\n"
                "| [`04_01_layer_norm.ipynb`](04_01_layer_norm.ipynb) | How do we stop the numbers drifting too big or too small? |\n"
                "| [`04_02_feed_forward.ipynb`](04_02_feed_forward.ipynb) | What does each letter do with what attention gave it? |\n"
                "| [`04_03_shortcuts.ipynb`](04_03_shortcuts.ipynb) | How do learning signals reach the lower floors of a tall stack? |\n"
                "| [`04_04_one_block.ipynb`](04_04_one_block.ipynb) | How do the parts fit together into one block? |\n"
                "| [`04_05_the_full_model.ipynb`](04_05_the_full_model.ipynb) | How do we stack blocks into the whole model, and make it write? |",
            ),
            q(
                "Will we train this bigger model in this chapter?",
                "No. Chapter 4 only builds it, with random numbers, and checks that the shapes fit. "
                "Training it is [chapter 5](../05_training_on_the_odyssey/05_00_chapter_overview.ipynb)'s job. "
                "And we don't touch `model.py` here. The full model gets a file of its own, `src/homer_ai/gpt.py` ([04_05](04_05_the_full_model.ipynb)).",
            ),
            recap(
                [
                    "A full GPT is a stack of identical blocks between the embeddings and the output head. A block is attention plus feed-forward, each with a layer norm and a shortcut.",
                    "Our `model.py` has the embedding, one attention head and the head (11,840 weights). It has no positions, norms, feed-forward or shortcuts.",
                    "A config puts every size in one place. Without one, `32` is repeated across three files.",
                ],
                "[04_01: Layer norm](04_01_layer_norm.ipynb): keeping the numbers from drifting.",
            ),
        ],
    )

    # =====================================================================
    # 04_building_the_model/04_01_layer_norm.ipynb
    # =====================================================================
    save(
        "04_building_the_model/04_01_layer_norm.ipynb",
        [
            title("04_01 Layer normalization", "section 4.2"),
            where("model reads", ["src/homer_ai/gpt.py"]),
            learn(
                [
                    "Why the numbers flowing through a deep model can drift too big or too small",
                    "How layer norm fixes each row: mean 0 and variance 1, worked by hand on 4 numbers",
                    "How `nn.LayerNorm` does the same and learns a scale and a shift",
                    "That our small `model.py` has no layer norm, and why it gets away with it",
                ]
            ),
            words(
                [
                    (
                        "layer norm",
                        "short for layer normalization: a step that rescales each row of numbers to a fixed mean and spread, then lets the model learn a scale and a shift",
                        "`nn.LayerNorm(32)`",
                    ),
                    (
                        "mean",
                        "the average: add the numbers, divide by how many",
                        "the mean of 3, 5, 7, 9 is 6",
                    ),
                    (
                        "variance",
                        "how spread out the numbers are: the average of each number's squared distance from the mean",
                        "for 3, 5, 7, 9 it's 5",
                    ),
                    (
                        "standard deviation",
                        "the square root of the variance: a typical distance from the mean, in the same units as the numbers",
                        "√5 ≈ 2.24",
                    ),
                    (
                        "scale and shift",
                        "two learned numbers per position in the row: multiply by the scale, then add the shift",
                        "`scale = 1, shift = 0` at the start",
                    ),
                ]
            ),
            *background(),
            q(
                "What goes wrong without it?",
                "Each layer multiplies the numbers by its weights. Do that again and again, and the numbers can shrink towards 0 or grow huge. "
                "If they shrink, there's nothing left to learn from. If they grow, learning gets jumpy (like the learning-rate spike in [05_02](../05_training_on_the_odyssey/05_02_learning_rate.ipynb)).\n\n"
                "We can watch it happen. Take the real five rows of `ἄνδρα`, pass them through 8 plain layers (random numbers, no norm), "
                "and after each layer measure the spread (the **standard deviation**) of the numbers in a row.\n\n"
                "👉 **Predict first:** the 32 numbers of each row start with a spread of about 1. After 8 layers, will it be about 1, much smaller, or much bigger?",
            ),
            code(CH4_LOAD),
            md(
                "**A new dimension: the batch.** The next cell makes `ids` with shape `(1, 5)` and `x` with shape `(1, 5, 32)`. "
                "That extra first number is the batch: a group of texts that the model processes at the same time (real training feeds in many at once, because the hardware is faster that way). "
                "We have just one text, so the batch size is 1. Read the shape as `(texts, letters, numbers per letter)`. "
                "Our `model.py` takes one text with no batch dimension (shape `(5,)`). The full model of this chapter takes a batch."
            ),
            code(CH4_ROWS),
            code(
                """def spread(h):
    # the standard deviation of each row (each letter), averaged over the letters
    return h.std(dim=-1, unbiased=False).mean().item()


def run(scale, with_norm):
    torch.manual_seed(1)
    h, history = x.detach(), [spread(x)]
    for _ in range(8):
        layer = nn.Linear(32, 32, bias=False)
        with torch.no_grad():
            layer.weight *= scale  # scale 1 = PyTorch's normal start
        h = layer(h)
        if with_norm:
            h = nn.LayerNorm(32)(h)
        history.append(spread(h))
    return history, h


plain, drifted = run(1, with_norm=False)
big, _ = run(3, with_norm=False)
normed, _ = run(1, with_norm=True)
print("plain layers      :", f"1.00 → {plain[-1]:.3f}")
print("plain, weights ×3 :", f"1.00 → {big[-1]:.1f}")
print("with a norm each  :", f"1.00 → {normed[-1]:.3f}")"""
            ),
            saw(
                "With plain layers the spread shrinks from about 1 to about 0.01. Make the weights three times bigger and it grows to about 65 instead. "
                "Put a norm after every layer and it stays at 1. And a real model has dozens of these steps, so the effect is even stronger. "
                "(`unbiased=False` means: divide by the number of values, as the layer norm does, not by one less.)"
            ),
            q(
                "Can I see it?",
                "Same numbers, as lines. The vertical axis is a **log** scale (each gridline is 10 times the one below), because the lines are so far apart.",
            ),
            code(
                """import matplotlib.pyplot as plt

from homer_ai.studio_style import ACCENT, BAD, GOOD, INK, studio_axes, studio_figure

fig = studio_figure("Layer norm", "The spread of a row, layer after layer", (9, 6))
ax = studio_axes(fig)
steps = range(9)
style = {"linewidth": 3, "marker": "o"}
ax.plot(steps, big, color=BAD, label="plain, weights ×3: grows", **style)
ax.plot(steps, plain, color=ACCENT, label="plain: shrinks", **style)
ax.plot(steps, normed, color=GOOD, label="with layer norm: stays at 1", **style)
ax.set_yscale("log")
ax.set_xlabel("number of layers passed")
ax.set_ylabel("spread (standard deviation) of a row")
ax.legend(frameon=False, labelcolor=INK, loc="upper left")
plt.show()"""
            ),
            saw(
                "The two plain lines run away from 1 in opposite directions, and the green line stays flat. That flat line is the whole point of layer norm."
            ),
            step_intro(),
            q(
                "Step 1: what does layer norm do to one row, by hand?",
                "Two moves, for each row on its own:\n\n"
                "1. **Centre it:** subtract the row's mean, so the new mean is 0.\n"
                "2. **Rescale it:** divide by the row's standard deviation, so the new variance is 1.\n\n"
                "Let's pretend the row for `ἄ` had only 4 numbers: `3, 5, 7, 9`.\n\n"
                "```\nrow            3     5     7     9      mean = (3+5+7+9)/4 = 6\n- mean        -3    -1    +1    +3      \nsquared        9     1     1     9      variance = 20/4 = 5, std = √5 ≈ 2.24\n÷ std       -1.34 -0.45 +0.45 +1.34      the normalised row\n```\n\n"
                "👉 **Predict first:** what's the mean of the last row? And its variance?",
            ),
            code(
                """row = torch.tensor([[3.0, 5.0, 7.0, 9.0]])

mean = row.mean(dim=-1, keepdim=True)
variance = row.var(dim=-1, keepdim=True, unbiased=False)
eps = 1e-5  # a tiny number so we never divide by 0
by_hand = (row - mean) / torch.sqrt(variance + eps)

print("mean:", mean.item(), " variance:", variance.item())
print("by hand:", [round(v, 2) for v in by_hand[0].tolist()])
new_mean = by_hand.mean().item()
new_variance = by_hand.var(unbiased=False).item()
print("new mean:", round(new_mean, 6), " new variance:", round(new_variance, 4))"""
            ),
            saw(
                "Mean 6 and variance 5, as in the picture. The new row is `-1.34, -0.45, 0.45, 1.34`, with mean 0 and variance 1 (a tiny bit under, because of `eps`). "
                '`dim=-1` means "along the last direction", so the numbers of one row. And `keepdim=True` keeps the answer shaped as a column, so we can subtract it from the row.'
            ),
            q(
                "Step 2: what does `nn.LayerNorm` add?",
                "It does the same two moves, then multiplies by a learned **scale** and adds a learned shift, one pair for each number in the row. "
                "They start at scale 1 and shift 0, so at first the result is just our by-hand answer. Training can change them later, "
                "if the model finds that a different centre or spread suits that place better.\n\n"
                "👉 **Predict first:** will `nn.LayerNorm(4)` give the same row as our by-hand answer?",
            ),
            code(
                """norm = nn.LayerNorm(4)
out = norm(row)
print("same as by hand?", torch.allclose(out, by_hand, atol=1e-6))
print("scale:", norm.weight.tolist())
print("shift:", norm.bias.tolist())"""
            ),
            saw(
                "`True`, a scale of four ones and a shift of four zeros. "
                "(`weight` and `bias` are PyTorch's names for the scale and the shift.)"
            ),
            *exercise(
                "can you normalise another row by hand?",
                "Here's a row with much bigger numbers: `[10, 20, 30, 40]`. Do to it what step 1 did to `[3, 5, 7, 9]`.\n\n"
                "👉 **Predict first:** these numbers are about four times as big. Will the normalised row be bigger too?",
                starter="""
            other_row = torch.tensor([[10.0, 20.0, 30.0, 40.0]])

            # 👉 do: subtract the row's mean, then divide by the square root of (variance + eps)
            # 💡 hint: mean and var both take dim=-1, keepdim=True; var also takes unbiased=False
            other_normed = None
            """,
                solution="""
            other_row = torch.tensor([[10.0, 20.0, 30.0, 40.0]])

            other_mean = other_row.mean(dim=-1, keepdim=True)
            other_variance = other_row.var(dim=-1, keepdim=True, unbiased=False)
            other_normed = (other_row - other_mean) / torch.sqrt(other_variance + eps)
            """,
                check="""
            check(
                "the normalised row",
                other_normed,
                norm(other_row),
                coach="Numbers near -1.16 and 1.16 mean var() divided by n - 1: pass "
                "unbiased=False. Numbers like -0.12 mean you divided by the variance itself: "
                "take its square root first.",
            )
            """,
                hint="Step 1 has all three lines. Write `other_row` where it has `row`, and keep `eps`.",
                answer="`[-1.34, -0.45, 0.45, 1.34]`, exactly what `[3, 5, 7, 9]` gave. Both rows are four evenly spaced numbers, and once the mean and the spread are taken out, that pattern is all that's left. "
                "This is why layer norm keeps a deep model steady: however big or small a row has drifted, it comes out at the same size.",
            ),
            q(
                "Step 3: what does it do to each letter of `ἄνδρα`?",
                "Each letter's row is normalised on its own. Letters never mix here. "
                "So let's take the shrunken rows from our 8-layer test (`drifted`) and normalise them.\n\n"
                "```\nbefore:  ἄ → 32 tiny numbers      after:  ἄ → 32 numbers, mean 0, spread 1\n         ν → 32 tiny numbers              ν → 32 numbers, mean 0, spread 1   …\n```\n\n"
                "👉 **Predict first:** the spread of each row, before and after.",
            ),
            code(
                """after = nn.LayerNorm(32)(drifted)

print("letter   spread before   mean after   spread after")
for i, letter in enumerate("ἄνδρα"):
    before_spread = drifted[0, i].std(unbiased=False).item()
    mean_after = after[0, i].mean().item()
    spread_after = after[0, i].std(unbiased=False).item()
    row = f"  {letter}        {before_spread:.4f}"
    print(row, f"       {mean_after:+.4f}        {spread_after:.4f}")"""
            ),
            saw(
                "Every row had a spread of about 0.01. After the norm each one has a spread of 0.88 to 0.97, with a mean of 0. Why not exactly 1? Because the tiny `eps` we add starts to matter when the spread is this small (about 0.01, so a variance of about 0.0001, close to `eps`). "
                "Each letter's row keeps its own pattern. Only its centre and spread are reset."
            ),
            q(
                "What does layer norm cost?",
                "Two numbers (scale and shift) per number in the row. At our size that's 2 × 32 = 64 weights for each layer norm. "
                "A block has two of them and the model has one more at the end, so it's cheap.\n\n"
                "👉 **Predict first:** how many weights is `nn.LayerNorm(32)`?",
            ),
            code(
                """big_norm = nn.LayerNorm(32)
print("weights:", sum(p.numel() for p in big_norm.parameters()))"""
            ),
            saw("64: that's 32 scales and 32 shifts."),
            q(
                "Does our `model.py` have layer norm?",
                "No. It has only one attention step, so the numbers don't travel through many layers and nothing drifts much. "
                "A tall stack does need it, which is why the block in [04_04](04_04_one_block.ipynb) has two. "
                "We don't change `model.py` here.",
            ),
            recap(
                [
                    "Passing numbers through many layers makes a row's spread shrink or grow (we measured 1 → 0.01, and 1 → 65).",
                    "Layer norm gives each row mean 0 and variance 1, on its own, then applies a learned scale and shift (64 weights for a row of 32).",
                    "Our `model.py` has no layer norm. The block in 04_04 has two.",
                ],
                "[04_02: Feed-forward](04_02_feed_forward.ipynb): what each letter does with what attention gave it.",
                can=[
                    "normalise a row by hand: subtract the mean, divide by the spread",
                    "use `nn.LayerNorm` and say what its scale and shift are for",
                    "measure whether a row's spread shrinks or grows from layer to layer",
                ],
            ),
        ],
    )

    # =====================================================================
    # 04_building_the_model/04_02_feed_forward.ipynb
    # =====================================================================
    save(
        "04_building_the_model/04_02_feed_forward.ipynb",
        [
            title("04_02 Feed-forward network with GELU", "section 4.3"),
            where("model reads", ["src/homer_ai/gpt.py"]),
            learn(
                [
                    "What the feed-forward step is: a small 2-layer network, applied to each letter on its own",
                    "Why it expands to 4 times the width and shrinks back",
                    "What GELU is, and how it differs from ReLU",
                    "How many weights it has, compared with attention and our whole model",
                ]
            ),
            words(
                [
                    (
                        "feed-forward network",
                        "a small network of layers where numbers only move forward, one layer after the next; in a block it works on each letter separately",
                        "32 → 128 → 32 numbers",
                    ),
                    (
                        "activation function",
                        "a small curve applied to every number after a layer, so that stacked layers can do more than one straight layer could",
                        "ReLU, GELU",
                    ),
                    (
                        "ReLU",
                        "an activation: negative numbers become 0, positive numbers stay",
                        "ReLU(-2) = 0, ReLU(3) = 3",
                    ),
                    (
                        "GELU",
                        "an activation like ReLU, but smooth: small negative numbers are shrunk, not cut to 0",
                        "GELU(-1) ≈ -0.16",
                    ),
                ]
            ),
            *background(),
            q(
                "What's left to do once attention has mixed the letters?",
                "Attention (chapter 3) lets each letter **gather** information from the letters before it. "
                "The feed-forward step then lets each letter think about what it gathered. "
                "It's applied to every letter on its own, so there's no looking at neighbours here.\n\n"
                "```\nattention:     ἄ ν δ ρ α   →  every letter mixes in the earlier ones\nfeed-forward:  ἄ   ν   δ   ρ   α   →  each letter is processed alone, by the same small network\n```\n\n"
                "It's a small network (see [01_02](../01_foundations/01_02_neural_networks.ipynb)) of two layers with a curve between them. "
                "The first layer expands each letter's 32 numbers to 4 times as many, 128, which gives it more room to think. Then the curve is applied. "
                "Then the second layer shrinks it back to 32, so the shape still fits the next floor.\n\n"
                "```\n32 numbers → [layer 1] → 128 numbers → [GELU curve] → 128 numbers → [layer 2] → 32 numbers\n```",
            ),
            q(
                "Why is there a curve in the middle?",
                "Two layers one after the other, with nothing between them, are secretly just one layer. Multiplying by one table of numbers and then by another "
                "is the same as multiplying once by a combined table. The curve (an **activation function**) breaks that, so the two layers can do something one layer can't. "
                "The curve works on every number separately.\n\n"
                "The simple, classic curve is ReLU: negative numbers become 0 and positive numbers stay as they are. GPT models use a smoother one, GELU. "
                "Let's compute both for a few numbers.\n\n"
                "👉 **Predict first:** what will ReLU give for -2, -1, 0, 1, 2? What do you guess GELU gives for -1?",
            ),
            code(
                """import torch
from torch import nn

numbers = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
print("number:", numbers.tolist())
print("ReLU:  ", nn.ReLU()(numbers).tolist())
print("GELU:  ", [round(v, 3) for v in nn.GELU()(numbers).tolist()])"""
            ),
            saw(
                "ReLU gives `0, 0, 0, 1, 2`: every negative number is cut to 0. GELU gives `-0.046, -0.159, 0, 0.841, 1.954`. "
                "So the positives are almost unchanged, and the negatives aren't thrown away, only shrunk. "
                "(The book writes GELU with a longer formula that's very close. PyTorch's `nn.GELU()` is the exact one, and that's the one we use.)"
            ),
            q(
                "Can I see the two curves?",
                "The input is on the left-right axis, the output up-down. ReLU has a sharp corner at 0. GELU is smooth, and dips a little below 0 before it rises.",
            ),
            code(
                """import matplotlib.pyplot as plt

from homer_ai.studio_style import ACCENT, INK, MUTED, studio_axes, studio_figure

xs = torch.linspace(-4, 4, 400)
relu_curve, gelu_curve = nn.ReLU()(xs), nn.GELU()(xs)
low = gelu_curve.argmin()  # the lowest point of the dip

fig = studio_figure("Activation functions", "ReLU cuts, GELU curves", (9, 6))
ax = studio_axes(fig)
ax.plot(xs, relu_curve, color=MUTED, linewidth=3, linestyle="--", label="ReLU")
ax.plot(xs, gelu_curve, color=ACCENT, linewidth=3.5, label="GELU")
ax.axhline(0, color=MUTED, linewidth=0.8)
ax.axvline(0, color=MUTED, linewidth=0.8)
ax.annotate(
    f"lowest dip: GELU({xs[low]:.2f}) = {gelu_curve[low]:.2f}",
    (xs[low], gelu_curve[low]), (-3.9, 1.8), color=INK, fontsize=10,
    family="monospace", arrowprops={"arrowstyle": "->", "color": INK},
)
ax.set_xlabel("number in")
ax.set_ylabel("number out")
ax.set_ylim(-1, 4)
ax.legend(frameon=False, labelcolor=INK, loc="lower right")
plt.show()"""
            ),
            saw(
                "For big positive numbers the two lines are the same, and for big negative numbers both are near 0. "
                "In between, GELU is smooth where ReLU has a corner, and it dips to about -0.17. "
                "Why does smooth matter? A smooth curve gives gradients ([01_03](../01_foundations/01_03_how_a_program_learns.ipynb)) that change gently, and that suits learning."
            ),
            step_intro(),
            q(
                "Step 1: build the feed-forward step",
                "Three parts in a row: a layer from 32 to 128 numbers, the GELU curve, and a layer from 128 back to 32. "
                "`nn.Sequential` runs its parts one after another. We feed in the five rows of `ἄνδρα`.\n\n"
                "👉 **Predict first:** what shape comes out for 5 letters of 32 numbers each?",
            ),
            code(CH4_LOAD),
            code(CH4_ROWS),
            code(
                """feed_forward = nn.Sequential(
    nn.Linear(32, 4 * 32),  # expand: 32 → 128
    nn.GELU(),
    nn.Linear(4 * 32, 32),  # shrink: 128 → 32
)

inside = feed_forward[0](x)
print("in:        ", tuple(x.shape))
print("expanded:  ", tuple(inside.shape))
print("out:       ", tuple(feed_forward(x).shape))"""
            ),
            saw(
                "`(1, 5, 32)` in, `(1, 5, 128)` in the middle, `(1, 5, 32)` out. Same shape out as in, and that's what lets blocks stack."
            ),
            q(
                "Step 2: is each letter really processed on its own?",
                "Let's test it. Put in only the first letter, `ἄ`, and compare with the first row of the full run. If the letters really are separate, the two must be the same.\n\n"
                "👉 **Predict first:** same, or different?",
            ),
            code(
                """alone = feed_forward(x[:, :1])  # only the first letter
together = feed_forward(x)[:, :1]  # the first row of the full run
print("same?", torch.allclose(alone, together, atol=1e-6))"""
            ),
            saw(
                "`True`. The other four letters had no effect, so the letters don't talk to each other here. (Talking is attention's job.)"
            ),
            q(
                "Step 3: how many weights?",
                "Each `nn.Linear` has a table of weights and a list of biases (one added number per output):\n\n"
                "```\nlayer 1:  32 × 128 weights + 128 biases = 4,096 + 128 = 4,224\nlayer 2: 128 × 32  weights +  32 biases = 4,096 +  32 = 4,128\n```\n\n"
                "👉 **Predict first:** the total. Is it more or less than our whole model's 11,840?",
            ),
            code(
                """total = sum(p.numel() for p in feed_forward.parameters())
print("feed-forward weights:", f"{total:,}")
print("attention in model.py: 3,072 (query + key + value)")
print("whole model.py:        11,840")"""
            ),
            saw(
                "8,352 weights. That's more than two and a half times the attention (3,072), and about 70% of the size of our entire model. "
                "(The biases are why the total is a little more than 2 × 4,096.)"
            ),
            *exercise(
                "what does a feed-forward step cost for rows of 128 numbers?",
                "Bigger models use longer rows. Say each letter has 128 numbers, four times our 32. The step still expands to 4 times the width and shrinks back. Count its weights with plain arithmetic, biases included.\n\n"
                "👉 **Predict first:** four times as wide. Four times the weights?",
                starter="""
            width = 128

            # 👉 do: the weights of Linear(width, 4 * width) plus those of Linear(4 * width, width)
            # 💡 hint: a Linear layer has in × out weights, plus one bias for each output
            wide_weights = None
            """,
                solution="""
            width = 128

            expand = width * 4 * width + 4 * width
            shrink = 4 * width * width + width
            wide_weights = expand + shrink
            """,
                check="""
            wide = nn.Sequential(nn.Linear(128, 512), nn.GELU(), nn.Linear(512, 128))
            check(
                "feed-forward weights at width 128",
                wide_weights,
                sum(p.numel() for p in wide.parameters()),
                coach="Got 131,072? Those are the weights without the biases. Each Linear "
                "layer also has one bias per output: 512 for the first, 128 for the second.",
            )
            """,
                hint="For our width of 32 the sum is `(32 × 128 + 128) + (128 × 32 + 32) = 8,352`. Do the same with 128 and 512.",
                answer="131,712, nearly 16 times our 8,352. The row got 4 times as long and the middle layer too, and 4 × 4 = 16. "
                "So the cost grows with the **square** of the width, which is why wide models get big so fast.",
            ),
            q(
                "Does our `model.py` have a feed-forward step?",
                "No. Between attention and `head` it has nothing, and we don't change `model.py` here. "
                "The feed-forward step lives in the block: in [04_04](04_04_one_block.ipynb) it becomes half of it.",
            ),
            recap(
                [
                    "The feed-forward step is a 2-layer network applied to each letter alone: 32 → 128 → 32, with a GELU curve between.",
                    "GELU is a smooth ReLU: ReLU cuts negatives to 0, GELU shrinks them (GELU(-1) ≈ -0.16).",
                    "It has 8,352 weights, more than twice the attention in our model. Our `model.py` has no feed-forward step.",
                ],
                "[04_03: Shortcuts](04_03_shortcuts.ipynb): how learning reaches the lower floors.",
                can=[
                    "build the feed-forward step with `nn.Sequential`: expand, GELU, shrink",
                    "count the weights of a `Linear` layer: in × out, plus one bias per output",
                    "say how GELU differs from ReLU for negative numbers",
                ],
            ),
        ],
    )

    # =====================================================================
    # 04_building_the_model/04_03_shortcuts.ipynb
    # =====================================================================
    save(
        "04_building_the_model/04_03_shortcuts.ipynb",
        [
            title("04_03 Shortcut connections", "section 4.4"),
            where("model reads", ["src/homer_ai/gpt.py"]),
            learn(
                [
                    "Why a tall stack of layers is hard to train: the learning signal fades on its way down",
                    "What a shortcut is: add the input back to the output",
                    "A real experiment: the size of the gradient in each layer, with and without shortcuts",
                    "That our small `model.py` has no shortcuts, and doesn't need them",
                ]
            ),
            words(
                [
                    (
                        "shortcut connection",
                        "a path that skips a layer: the layer's input is added back to its output (also called a residual connection)",
                        "`x = x + layer(x)`",
                    ),
                    (
                        "vanishing gradient",
                        "the learning signal getting smaller and smaller on its way back through many layers, until the lowest layers barely learn",
                        "a gradient of 0.0002 in layer 1",
                    ),
                ]
            ),
            *background(),
            q(
                "What problem do shortcuts solve?",
                "Learning ([01_03](../01_foundations/01_03_how_a_program_learns.ipynb)) works backwards. The mistake at the output makes a **gradient** (a signal that says how to change each weight), "
                "and that signal travels back down through the layers. At each layer it's multiplied by more numbers. If those are smaller than 1, the signal gets smaller floor by floor, "
                "like a message whispered down a long line of people. The floors near the input get almost nothing and barely learn. That's the **vanishing gradient**.\n\n"
                "A picture with the real numbers comes after the experiment below.",
            ),
            q(
                "What's a shortcut?",
                "Don't replace the numbers with the layer's answer. **Add** the layer's answer to the numbers it was given:\n\n"
                "```\nwithout:   x → [layer] → out            out = layer(x)\nwith:      x → [layer] → + → out        out = x + layer(x)\n           └────────────┘ the shortcut\n```\n\n"
                "Two good things happen. Going forward, the layer only has to learn a small correction to `x`, which is easier than rebuilding it. "
                "Going backward, the gradient gets a second road that skips the layer, so it doesn't have to pass through the layer's multiplication. That road is clear all the way down to the lowest floor.\n\n"
                "Here's a tiny example with the first four numbers of the real row for `ἄ`, and one untrained layer.\n\n"
                "👉 **Predict first:** will `x + layer(x)` be close to `x`, or far from it?",
            ),
            code(
                """import torch
from torch import nn

from homer_ai.tokenizer import CharTokenizer
from homer_ai.train_homer import BOOKS_DIR, load_books

tokenizer = CharTokenizer(load_books(BOOKS_DIR)[0])
torch.manual_seed(0)
embedding = nn.Embedding(tokenizer.vocab_size, 32)
layer = nn.Linear(32, 32)

letter = torch.tensor(tokenizer.encode("ἄ"))
row = embedding(letter)[0]  # the 32 numbers for ἄ
answer = layer(row)

print("x (first 4)          :", [round(v, 2) for v in row[:4].tolist()])
print("layer(x)             :", [round(v, 2) for v in answer[:4].tolist()])
print("x + layer(x)         :", [round(v, 2) for v in (row + answer)[:4].tolist()])"""
            ),
            saw(
                "The layer's answer is a different set of numbers, but `x + layer(x)` still carries all of `x` inside it. "
                "So nothing from the input is lost, however badly the layer treats it."
            ),
            step_intro(),
            q(
                "Step 1: build a deep stack, with a switch for shortcuts",
                "In PyTorch a model is a class based on `nn.Module`. Its constructor (`__init__`) makes the layers, so PyTorch can find all their weights, and `forward` says what happens to the numbers that go through ([A_04](../A_toolbox/A_04_pytorch_in_this_repo.ipynb)). "
                "This is the book's experiment: five small layers, each followed by GELU. Four of them take 3 numbers in and give 3 out, and the last one goes from 3 numbers to 1. "
                "The switch `use_shortcut` decides whether a layer does `x + layer(x)` or just `layer(x)`. "
                "The last layer changes the shape (3 numbers to 1), so it can't have a shortcut.\n\n"
                "👉 **Predict first:** with the same starting weights, will the two stacks give the same output?",
            ),
            code(
                """class DeepStack(nn.Module):
    def __init__(self, sizes, use_shortcut):
        super().__init__()
        self.use_shortcut = use_shortcut
        self.layers = nn.ModuleList(
            nn.Sequential(nn.Linear(sizes[i], sizes[i + 1]), nn.GELU())
            for i in range(len(sizes) - 1)
        )

    def forward(self, x):
        for layer in self.layers:
            out = layer(x)
            if self.use_shortcut and x.shape == out.shape:
                x = x + out  # the shortcut
            else:
                x = out
        return x


sizes = [3, 3, 3, 3, 3, 1]
sample = torch.tensor([[1.0, 0.0, -1.0]])  # three invented numbers for one letter

torch.manual_seed(123)
plain = DeepStack(sizes, use_shortcut=False)
torch.manual_seed(123)  # the same starting weights
shortcut = DeepStack(sizes, use_shortcut=True)
print("without:", plain(sample).item())
print("with:   ", shortcut(sample).item())"""
            ),
            saw(
                "Different outputs from exactly the same weights, because the shortcuts changed what flows through. "
                "`nn.ModuleList` is a list that PyTorch knows holds layers, so it can find their weights. "
                "And `x.shape == out.shape` is the check that a shortcut is only added when the two have the same shape."
            ),
            q(
                "Step 2: how big is the gradient in each layer?",
                "We give each stack a target of 0, measure the mistake (mean squared error), run `backward()` ([01_03](../01_foundations/01_03_how_a_program_learns.ipynb)), "
                "and print the average size of the gradient of each layer's weights. Layer 1 is the one nearest the input.\n\n"
                "👉 **Predict first:** with shortcuts, will layer 1's gradient be larger or smaller than without? By how much?",
            ),
            code(
                """def gradient_sizes(model):
    loss = nn.MSELoss()(model(sample), torch.tensor([[0.0]]))
    loss.backward()
    return [layer[0].weight.grad.abs().mean().item() for layer in model.layers]


without = gradient_sizes(plain)
with_shortcuts = gradient_sizes(shortcut)
print("layer   without shortcuts   with shortcuts")
for i, (a, b) in enumerate(zip(without, with_shortcuts, strict=True), start=1):
    print(f"  {i}        {a:.5f}            {b:.5f}")"""
            ),
            saw(
                "Without shortcuts, the gradient in the layers near the input is tiny. Layer 1 gets about 0.0002, against about 0.005 in the last layer. That's 25 times less. "
                "It doesn't fall at every single step (layer 2 is a hair smaller than layer 1, because five small layers with random starting weights aren't perfectly even), but the trend is clear. "
                "With shortcuts, layer 1 gets about 0.22. That's roughly a thousand times bigger, and no layer is starved. The signal reaches the lowest floor. "
                "(Your numbers match ours because of `torch.manual_seed(123)`.)"
            ),
            q(
                "Can I see it?",
                "Same numbers as bars, one pair per layer. The vertical axis is a **log** scale, because the two sets differ by a factor of about a thousand.",
            ),
            code(
                """import matplotlib.pyplot as plt

from homer_ai.studio_style import BAD, GOOD, INK, MUTED, studio_axes, studio_figure

fig = studio_figure("Shortcuts", "The learning signal in each layer", (10, 6))
ax = studio_axes(fig)
positions = range(1, 6)
bar = {"width": 0.38}
left = [p - 0.2 for p in positions]
right = [p + 0.2 for p in positions]
label = {"ha": "center", "color": INK, "fontsize": 8, "family": "monospace"}
ax.bar(left, without, color=BAD, **bar, label="without shortcuts")
ax.bar(right, with_shortcuts, color=GOOD, **bar, label="with shortcuts")
for p, a, b in zip(positions, without, with_shortcuts, strict=True):
    ax.text(p - 0.2, a * 1.15, f"{a:.4f}", **label)
    ax.text(p + 0.2, b * 1.15, f"{b:.2f}", **label)
ax.set_yscale("log")
ax.set_ylim(top=3)
names = [f"layer {p}" for p in positions]
names[0] += " (input end)"
ax.set_xticks(list(positions), names, color=MUTED, fontsize=9)
ax.set_ylabel("average gradient size")
ax.legend(frameon=False, labelcolor=INK, loc="upper left")
plt.show()"""
            ),
            saw(
                "The red bars (no shortcuts) shrink towards the input end, and the green ones stay high everywhere. "
                "That's why a tall stack of blocks needs shortcuts. Without them the lower floors would learn almost nothing."
            ),
            md(
                "**Look at:** how thick each arrow is. The mistake travels down from the output; the thicker the arrow, the more of the signal reaches that layer. "
                "On the right, the green loops are the shortcuts."
            ),
            code(DIAGRAM_GRADIENT),
            q(
                "Does our `model.py` have shortcuts?",
                "No, and it doesn't need them: its layers aren't stacked, so nothing fades. The block in [04_04](04_04_one_block.ipynb) uses two shortcuts. "
                "We don't change `model.py` here.",
            ),
            recap(
                [
                    "In a deep stack the gradient fades on the way down, so the lowest layers barely learn (we measured about 0.0002 in layer 1).",
                    "A shortcut adds a layer's input back to its output (`x + layer(x)`). With it, layer 1's gradient was about 0.22.",
                    "Our `model.py` has no shortcuts. Every block has two.",
                ],
                "[04_04: One block](04_04_one_block.ipynb): putting the parts together.",
                can=[
                    "add a shortcut around a layer: `x + layer(x)`",
                    "measure the gradient that reaches the first layer, with and without shortcuts",
                ],
            ),
        ],
    )

    # =====================================================================
    # 04_building_the_model/04_04_one_block.ipynb
    # =====================================================================
    save(
        "04_building_the_model/04_04_one_block.ipynb",
        [
            title(
                "04_04 Connecting attention and feed-forward in a block", "section 4.5"
            ),
            where("model reads", ["src/homer_ai/gpt.py"]),
            learn(
                [
                    "How the four parts fit into one block: norm, attention, shortcut, norm, feed-forward, shortcut",
                    "How to write the block as a class, built here from small pieces",
                    "That the shape going in equals the shape coming out",
                    "How many weights one block has, part by part",
                ]
            ),
            words(
                [
                    (
                        "output projection",
                        "one extra layer after the heads are joined, so the heads can mix their results",
                        "32 × 32 weights, plus 32 biases",
                    ),
                    (
                        "buffer",
                        "a table a model keeps with it, saved and moved with it, but which isn't learned",
                        "the causal mask",
                    ),
                ]
            ),
            *background(),
            q(
                "In which order do the parts go?",
                "A block takes a row of numbers for each letter and gives back rows of the same shape. Inside there are two steps, and both follow the same recipe: "
                "**norm first, then the part, then add the input back** (a shortcut, [04_03](04_03_shortcuts.ipynb)).\n\n"
                "Putting the norm before each part (not after) is what GPT-2 and the book do. It's called pre-norm. "
                "Dropout (switching off some numbers at random while training, [03_03](../03_attention/03_03_hiding_the_future.ipynb)) is optional. With `drop_rate` 0 it does nothing.",
            ),
            md(
                "**Look at:** the grey road goes up through the parts. The green road is the shortcut that goes round them and is added back at each `+`."
            ),
            code(DIAGRAM_BLOCK),
            step_intro(),
            q(
                "First, a warm-up: how do `view` and `transpose` cut the numbers into heads?",
                "The class in step 1 does it in one line, on tensors with 4 dimensions. That's a big jump, so let's do it slowly on numbers we can read. "
                "Take `ἄνδρα` with only **4** numbers per letter, and number them 0 to 19 so we can follow each one. The shape is `(1, 5, 4)`: 1 text, 5 letters, 4 numbers per letter. "
                "We want 2 heads, so each head gets 2 of the 4 numbers:\n\n"
                "```\n"
                "            head 0    head 1\n"
                "letter ἄ:  [  0   1 |   2   3 ]\n"
                "letter ν:  [  4   5 |   6   7 ]\n"
                "letter δ:  [  8   9 |  10  11 ]\n"
                "letter ρ:  [ 12  13 |  14  15 ]\n"
                "letter α:  [ 16  17 |  18  19 ]\n"
                "```\n\n"
                "Two moves get us there:\n\n"
                "- `view(1, 5, 2, 2)` cuts each letter's 4 numbers into 2 pairs. The shape reads `(text, letters, heads, numbers per head)`. No number moves. Only the grouping changes.\n"
                "- `transpose(1, 2)` swaps the letters and the heads, so the shape reads `(text, heads, letters, numbers per head)`. With the heads in front, each head holds its own small `(5, 2)` table, and the attention maths of chapter 3 runs on each one.\n\n"
                "👉 **Predict first:** after the two moves, which numbers will head 0 hold for the letter `ν`?",
            ),
            code(
                """import torch

x = torch.arange(20).view(1, 5, 4)  # 1 text, 5 letters, 4 numbers per letter
print("start:          ", tuple(x.shape))

cut = x.view(1, 5, 2, 2)  # each letter's 4 numbers as 2 pairs
print("after view:     ", tuple(cut.shape), " (text, letters, heads, per head)")

heads = cut.transpose(1, 2)  # swap letters and heads
print("after transpose:", tuple(heads.shape), " (text, heads, letters, per head)")
print("head 0:", heads[0, 0].tolist())
print("head 1:", heads[0, 1].tolist())

back = heads.transpose(1, 2).reshape(1, 5, 4)  # swap back, glue the pairs together
print("joined again:   ", tuple(back.shape), " same as x?", torch.equal(back, x))"""
            ),
            saw(
                "`(1, 5, 4)` becomes `(1, 5, 2, 2)` and then `(1, 2, 5, 2)`. Head 0 holds the left half of every letter (`[4, 5]` for `ν`), and head 1 the right half (`[6, 7]`), just like the picture. "
                "Swapping back and gluing the pairs together (`reshape`) gives exactly the tensor we started with, so nothing was lost or mixed up.\n\n"
                "The class below does the same with `(1, 5, 32)`: 2 heads of 16 numbers. Its `split` is the `view` plus `transpose`, and its last line is the swap back plus `reshape`."
            ),
            *exercise(
                "can you cut 6 numbers into 3 heads?",
                "Same trick, other sizes. Each of the 5 letters now has 6 numbers, and we want 3 heads with 2 numbers each. The result should have the shape `(1, 3, 5, 2)`: text, heads, letters, numbers per head.\n\n"
                "👉 **Predict first:** head 0 should hold the first 2 numbers of every letter. Letter 0 has `0 1 2 3 4 5` and letter 1 has `6 7 8 9 10 11`. What are the first two rows of head 0?",
                starter="""
            six = torch.arange(30).view(1, 5, 6)  # 1 text, 5 letters, 6 numbers per letter

            # 👉 do: cut each letter's 6 numbers into 3 heads of 2, then put the heads
            #        before the letters
            three_heads = None
            """,
                solution="""
            six = torch.arange(30).view(1, 5, 6)  # 1 text, 5 letters, 6 numbers per letter

            three_heads = six.view(1, 5, 3, 2).transpose(1, 2)
            """,
                check="""
            check(
                "the shape of the 3 heads",
                three_heads,
                test=lambda heads: tuple(heads.shape) == (1, 3, 5, 2),
                expected="(1, 3, 5, 2)",
                show=lambda heads: tuple(heads.shape),
                coach="(1, 5, 3, 2) means the cut is done but the swap is missing: add "
                ".transpose(1, 2).",
            )
            check(
                "head 0",
                three_heads,
                six.view(1, 5, 3, 2).transpose(1, 2),
                show=lambda heads: heads[0, 0].tolist(),
                coach="Right shape, wrong numbers? Then you wrote view(1, 3, 5, 2) in one go. "
                "view only re-cuts the numbers in the order they are in. It can't swap two "
                "dimensions. Cut with view(1, 5, 3, 2) first, then swap with transpose(1, 2).",
            )
            """,
                hint="The warm-up did `x.view(1, 5, 2, 2)` and then `.transpose(1, 2)`. Only the two sizes in the `view` change.",
                answer="Head 0 is `[[0, 1], [6, 7], [12, 13], [18, 19], [24, 25]]`: the first two numbers of each letter. "
                "Try `six.view(1, 3, 5, 2)` to see the trap. The shape is right, but head 0 comes out as `[[0, 1], [2, 3], [4, 5], …]`, which is all of letter 0 and the start of letter 1, cut up. No error, just wrong numbers.",
            ),
            q(
                "Step 1: attention for several heads, in one class",
                "In [03_04](../03_attention/03_04_many_heads.ipynb) we built two heads by hand and joined them. Here every head is computed at once. "
                "The idea is the same:\n\n"
                "```\n(5 letters, 32 numbers) → cut the 32 into heads → (heads, 5, 16 numbers) → scores, mask, softmax, mix for each head → join → output projection\n```\n\n"
                "The cutting and joining are the moves of the warm-up: `view` re-cuts the same numbers into a new shape, `transpose` swaps two directions, and `reshape` re-cuts again after the swap back. "
                "The extra first dimension is the **batch** of [04_01](04_01_layer_norm.ipynb) (the texts processed together, here 1), so the tensors have 4 dimensions: `(batch, heads, letters, numbers per head)`.\n\n"
                "The causal mask is saved with `register_buffer`, so it travels with the model but has no weights. "
                "Here it's a table of `True`/`False`, and `masked_fill` writes `-inf` into the `True` places. You get the same result as adding the `-inf` table of [03_03](../03_attention/03_03_hiding_the_future.ipynb). "
                "The difference is that a `True`/`False` table can be cut down to the length of the text (`mask[:length, :length]`), and it's the form the book and GPT code use. "
                "We also add the output projection, one extra 32 × 32 layer that the book puts after the join so the heads can mix their results.\n\n"
                "👉 **Predict first:** with `n_heads=2`, how many numbers does one head use per letter?",
            ),
            code(
                """import torch
from torch import nn


class MultiHeadAttention(nn.Module):
    def __init__(self, emb_dim, n_heads, context_length, drop_rate):
        super().__init__()
        self.n_heads = n_heads
        self.head_size = emb_dim // n_heads  # 32 // 2 = 16
        self.query = nn.Linear(emb_dim, emb_dim, bias=False)
        self.key = nn.Linear(emb_dim, emb_dim, bias=False)
        self.value = nn.Linear(emb_dim, emb_dim, bias=False)
        self.out_proj = nn.Linear(emb_dim, emb_dim)  # the output projection
        self.dropout = nn.Dropout(drop_rate)
        mask = torch.triu(torch.ones(context_length, context_length), diagonal=1).bool()
        self.register_buffer("mask", mask)  # True = hidden

    def forward(self, x):
        batch, length, emb_dim = x.shape
        # cut the 32 numbers into heads: into (batch, heads, length, 16)
        def split(t):
            return t.view(batch, length, self.n_heads, self.head_size).transpose(1, 2)

        q, k, v = split(self.query(x)), split(self.key(x)), split(self.value(x))
        scores = q @ k.transpose(2, 3) / self.head_size**0.5
        scores = scores.masked_fill(self.mask[:length, :length], float("-inf"))
        weights = self.dropout(torch.softmax(scores, dim=-1))
        mixed = weights @ v  # (batch, heads, length, 16)
        context = mixed.transpose(1, 2).reshape(batch, length, emb_dim)  # join
        return self.out_proj(context)


attention = MultiHeadAttention(32, n_heads=2, context_length=24, drop_rate=0.0)
print("head size:", attention.head_size)"""
            ),
            saw(
                "16 numbers per head, and 2 heads × 16 = 32, as in [03_04](../03_attention/03_04_many_heads.ipynb). "
                "Everything else in `forward` is the scores, mask, softmax and mix of chapter 3, with one spare direction for the heads."
            ),
            q(
                "Is this really the same as the two heads of 03_04?",
                "The batched code is shorter but harder to read, so let's prove it. Below is the 03_04 way: a loop with one small attention per head "
                "(rows 0-15 of each weight table make head 1's numbers, rows 16-31 make head 2's), then `torch.cat` to join. "
                "We give it the **same weights** as the batched class, and compare after the output projection.\n\n"
                "👉 **Predict first:** will `torch.allclose` say the two agree?",
            ),
            code(CH4_LOAD.replace("import torch\nfrom torch import nn\n\n", "")),
            code(CH4_ROWS),
            code(
                """def by_loop(x, attention):
    # x has shape (letters, 32): one text, so no batch dimension, as in chapter 3
    size = attention.head_size
    contexts = []
    for h in range(attention.n_heads):
        rows = slice(h * size, (h + 1) * size)  # this head's 16 of the 32 numbers
        Q = x @ attention.query.weight[rows].T
        K = x @ attention.key.weight[rows].T
        V = x @ attention.value.weight[rows].T
        scores = (Q @ K.T) / size**0.5
        hidden = torch.triu(torch.ones(len(x), len(x)) * float("-inf"), diagonal=1)
        scores = scores + hidden
        contexts.append(torch.softmax(scores, dim=-1) @ V)
    return attention.out_proj(torch.cat(contexts, dim=-1))  # join, then project


attention.eval()  # no dropout
with torch.no_grad():
    batched = attention(x)[0]  # [0]: drop the batch dimension
    looped = by_loop(x[0], attention)
print("batched equals the loop of 03_04?", torch.allclose(batched, looped, atol=1e-6))"""
            ),
            saw(
                "`True`. The 4-dimensional code and the loop give the same numbers. The batched version just does both heads at once."
            ),
            q(
                "Step 2: the feed-forward step, and the block itself",
                "The feed-forward step is the same as in [04_02](04_02_feed_forward.ipynb). The block then follows the picture above, line by line: "
                "`x = x + dropout(attention(norm1(x)))`, then `x = x + dropout(feed_forward(norm2(x)))`.\n\n"
                "👉 **Predict first:** the block takes `(1, 5, 32)` for the five letters of `ἄνδρα`. What shape comes out?",
            ),
            code(
                """class FeedForward(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(emb_dim, 4 * emb_dim), nn.GELU(), nn.Linear(4 * emb_dim, emb_dim)
        )

    def forward(self, x):
        return self.layers(x)


class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.attention = MultiHeadAttention(
            cfg["emb_dim"], cfg["n_heads"], cfg["context_length"], cfg["drop_rate"]
        )
        self.feed_forward = FeedForward(cfg["emb_dim"])
        self.norm1 = nn.LayerNorm(cfg["emb_dim"])
        self.norm2 = nn.LayerNorm(cfg["emb_dim"])
        self.dropout = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        x = x + self.dropout(self.attention(self.norm1(x)))  # step 1 + shortcut
        x = x + self.dropout(self.feed_forward(self.norm2(x)))  # step 2 + shortcut
        return x


GPT_CONFIG = {
    "vocab_size": 137, "context_length": 24, "emb_dim": 32,
    "n_heads": 2, "n_layers": 2, "drop_rate": 0.1,
}
block = TransformerBlock(GPT_CONFIG)
print(block)"""
            ),
            saw(
                "The block lists its parts: the attention (with its three layers, the output projection and dropout), the feed-forward, two layer norms and a dropout. "
                "A `cfg` dictionary is all it needs. Every size comes from [the config](04_00_chapter_overview.ipynb) and nowhere else."
            ),
            q(
                "Step 3: does the shape stay the same?",
                "Take the five letters of `ἄνδρα` as untrained embedding rows (the `x` from above), and send them through the block.",
            ),
            code(
                """block.eval()  # switch dropout off, so the answers are the same every time
out = block(x)
print("in :", tuple(x.shape))
print("out:", tuple(out.shape))"""
            ),
            saw(
                "`(1, 5, 32)` in and `(1, 5, 32)` out. Same shape, so the output of one block can be the input of the next, again and again."
            ),
            q(
                "Step 4: does the block still hide the future?",
                "A block has to keep the rule from [03_03](../03_attention/03_03_hiding_the_future.ipynb). So let's test it: change only the **last** letter, "
                "and see whether the output for the first four letters changes.\n\n"
                "👉 **Predict first:** will the first four rows of the output change?",
            ),
            code(
                """changed = x.clone()
changed[0, 4] = torch.randn(32)  # a different last letter

same_before_last = torch.allclose(block(x)[0, :4], block(changed)[0, :4], atol=1e-6)
same_last = torch.allclose(block(x)[0, 4], block(changed)[0, 4], atol=1e-6)
print("first 4 letters unchanged?", same_before_last)
print("last letter unchanged?    ", same_last)"""
            ),
            saw(
                "The first four are unchanged (nothing leaks backwards) and the last one changed, as it should. The feed-forward and the norms work letter by letter, so only attention could leak, and the mask stops it."
            ),
            q(
                "Step 5: how many weights?",
                "Let's count the real ones, part by part. The attention has the three 32 × 32 layers (3,072, as in our model) plus the 32 × 32 output projection with its 32 biases.\n\n"
                "👉 **Predict first:** is the whole block bigger than our entire `model.py` (11,840)?",
            ),
            code(
                """def count(module):
    return sum(p.numel() for p in module.parameters())


for name in ["attention", "feed_forward", "norm1", "norm2"]:
    print(f"{name:13} {count(getattr(block, name)):>6,}")
print(f"{'one block':13} {count(block):>6,}")"""
            ),
            saw(
                "Attention 4,128, feed-forward 8,352, two norms at 64 each. That's 12,608 for one block, a bit more than our whole `model.py`. "
                "Feed-forward is the biggest part (about two thirds). And the mask is a buffer with no weights, so it isn't counted."
            ),
            recap(
                [
                    "A block is `x = x + attention(norm(x))`, then `x = x + feed_forward(norm(x))`. The shape in equals the shape out.",
                    "Our block with 32 numbers and 2 heads has 12,608 weights (attention 4,128, feed-forward 8,352, two norms 64 each) and doesn't leak the future.",
                    "Because the shape is unchanged, blocks can be stacked. That's the next notebook.",
                ],
                "[04_05: The full model](04_05_the_full_model.ipynb): stacking blocks and writing letters.",
                can=[
                    "cut a row into heads with `view`, and swap dimensions with `transpose`",
                    "write a block's `forward`: two shortcuts, each around norm and then a layer",
                    "check that a block keeps the shape and hides the future",
                ],
            ),
        ],
    )

    # =====================================================================
    # 04_building_the_model/04_05_the_full_model.ipynb
    # =====================================================================
    save(
        "04_building_the_model/04_05_the_full_model.ipynb",
        [
            title("04_05 The full model, and generating text", "sections 4.6 and 4.7"),
            where("model reads", ["src/homer_ai/gpt.py"]),
            learn(
                [
                    "How the full GPT is assembled: embeddings, N blocks, a final norm, the output head",
                    "How many weights a few configs have, against our 11,840 and GPT-2 small",
                    "How one text flows through it, with every shape printed",
                    "How a simple generate() writes one letter at a time (untrained, so gibberish)",
                    "That this model is `src/homer_ai/gpt.py`, checked against the one built here",
                ]
            ),
            words(
                [
                    (
                        "output head",
                        "the last layer of the model: it turns each letter's 32 numbers into 137 scores, one per token (not an attention head)",
                        "`nn.Linear(32, 137)`",
                    ),
                    (
                        "logits",
                        "the raw scores the model gives each possible next token, before softmax turns them into chances",
                        "137 logits for each letter",
                    ),
                ]
            ),
            *background(),
            q(
                "How do the pieces go together?",
                "It's the picture from [04_00](04_00_chapter_overview.ipynb), written as code. The model reads the letters, looks up each one's token row **and** position row and adds them ([02_06](../02_text_to_numbers/02_06_positions.ipynb)), "
                "sends them through the blocks, normalises once more, and then the output head makes the scores.\n\n"
                "```\nIDs (1, 5) → token + position (1, 5, 32) → N blocks (1, 5, 32) → final norm → output head → logits (1, 5, 137)\n```\n\n"
                "We build the block from 04_04 again here, compactly, so that this notebook runs on its own. "
                "The new class is `GPTModel` (the last class in the cell).\n\n"
                "👉 **Predict first:** the model gets IDs of shape `(1, 5)`. What's the shape of the answer?",
            ),
            code(
                """import torch
from torch import nn

from homer_ai.tokenizer import CharTokenizer
from homer_ai.train_homer import BOOKS_DIR, load_books

tokenizer = CharTokenizer(load_books(BOOKS_DIR)[0])


class MultiHeadAttention(nn.Module):  # from 04_04
    def __init__(self, emb_dim, n_heads, context_length, drop_rate):
        super().__init__()
        self.n_heads, self.head_size = n_heads, emb_dim // n_heads
        self.query = nn.Linear(emb_dim, emb_dim, bias=False)
        self.key = nn.Linear(emb_dim, emb_dim, bias=False)
        self.value = nn.Linear(emb_dim, emb_dim, bias=False)
        self.out_proj = nn.Linear(emb_dim, emb_dim)
        self.dropout = nn.Dropout(drop_rate)
        mask = torch.triu(torch.ones(context_length, context_length), diagonal=1).bool()
        self.register_buffer("mask", mask)

    def forward(self, x):
        batch, length, emb_dim = x.shape

        def split(t):
            return t.view(batch, length, self.n_heads, self.head_size).transpose(1, 2)

        q, k, v = split(self.query(x)), split(self.key(x)), split(self.value(x))
        scores = q @ k.transpose(2, 3) / self.head_size**0.5
        scores = scores.masked_fill(self.mask[:length, :length], float("-inf"))
        weights = self.dropout(torch.softmax(scores, dim=-1))
        mixed = weights @ v
        return self.out_proj(mixed.transpose(1, 2).reshape(batch, length, emb_dim))


class TransformerBlock(nn.Module):  # from 04_04
    def __init__(self, cfg):
        super().__init__()
        e = cfg["emb_dim"]
        heads, length, drop = cfg["n_heads"], cfg["context_length"], cfg["drop_rate"]
        self.attention = MultiHeadAttention(e, heads, length, drop)
        self.feed_forward = nn.Sequential(
            nn.Linear(e, 4 * e), nn.GELU(), nn.Linear(4 * e, e)
        )
        self.norm1, self.norm2 = nn.LayerNorm(e), nn.LayerNorm(e)
        self.dropout = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        x = x + self.dropout(self.attention(self.norm1(x)))
        return x + self.dropout(self.feed_forward(self.norm2(x)))


class GPTModel(nn.Module):  # new: the full model
    def __init__(self, cfg):
        super().__init__()
        self.token_embedding = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.position_embedding = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.dropout = nn.Dropout(cfg["drop_rate"])
        blocks = [TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        self.blocks = nn.Sequential(*blocks)
        self.final_norm = nn.LayerNorm(cfg["emb_dim"])
        self.output_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, ids):
        positions = torch.arange(ids.shape[1])  # 0, 1, 2, … for each place
        x = self.token_embedding(ids) + self.position_embedding(positions)
        x = self.dropout(x)
        x = self.blocks(x)
        x = self.final_norm(x)
        return self.output_head(x)  # the logits


GPT_CONFIG = {
    "vocab_size": tokenizer.vocab_size, "context_length": 24, "emb_dim": 32,
    "n_heads": 2, "n_layers": 2, "drop_rate": 0.1,
}
torch.manual_seed(0)
model = GPTModel(GPT_CONFIG)
ids = torch.tensor([tokenizer.encode("ἄνδρα")])
print("IDs in:    ", tuple(ids.shape))
print("logits out:", tuple(model(ids).shape))"""
            ),
            saw(
                '`(1, 5)` in, `(1, 5, 137)` out. That\'s 137 scores for each of the 5 letters, one score per token, the same output shape that `model.py` makes (apart from the leading 1 for "one text"). '
                "`torch.arange(ids.shape[1])` gives `0, 1, 2, 3, 4`, the places to look up in the position table. "
                "The names `token_embedding` and `output_head` are ours. `model.py` calls them `embedding` and `head`."
            ),
            q(
                "Can I follow the shapes step by step?",
                "Yes. We run the same steps one at a time and print the shape after each one. The model is in `eval()` mode, so dropout is off.\n\n"
                "👉 **Predict first:** which steps change the shape, and which keep it?",
            ),
            code(
                """model.eval()
with torch.no_grad():
    tokens = model.token_embedding(ids)
    places = model.position_embedding(torch.arange(5))
    x = tokens + places
    print("token rows         ", tuple(tokens.shape))
    print("position rows      ", tuple(places.shape), " (the same for every text)")
    print("token + position   ", tuple(x.shape))
    for i, block in enumerate(model.blocks, start=1):
        x = block(x)
        print(f"after block {i}      ", tuple(x.shape))
    x = model.final_norm(x)
    print("after final norm   ", tuple(x.shape))
    print("after output head  ", tuple(model.output_head(x).shape))"""
            ),
            saw(
                "Only the last step changes the shape (32 numbers become 137 scores). Every other step keeps `(1, 5, 32)`, and that's why we can add as many blocks as we like."
            ),
            step_intro(),
            q(
                "Step 1: how many weights for a few configs?",
                "The size of the model comes from the config. So let's build four, count the real weights, and put them next to our `model.py` and GPT-2 small. "
                "We don't build GPT-2 small here (it's huge). Its size is a number everyone quotes: about **124 million** (vocabulary 50,257, context 1,024, 768 numbers, 12 heads, 12 blocks).\n\n"
                "👉 **Predict first:** the first config below has 2 blocks. Will it have about 2 times, 3 times or 10 times the weights of `model.py`?",
            ),
            code(
                """def count(module):
    return sum(p.numel() for p in module.parameters())


configs = {
    "2 blocks, 32 wide": {**GPT_CONFIG},
    "4 blocks, 32 wide": {**GPT_CONFIG, "n_layers": 4},
    "4 blocks, 64 wide": {**GPT_CONFIG, "n_layers": 4, "emb_dim": 64, "n_heads": 4},
    "6 blocks, 128 wide": {**GPT_CONFIG, "n_layers": 6, "emb_dim": 128, "n_heads": 4},
}
print(f"{'our small model.py':22} {11_840:>12,}")
for name, cfg in configs.items():
    print(f"{name:22} {count(GPTModel(cfg)):>12,}")
print(f"{'GPT-2 small (known)':22} {'124 million':>12}")"""
            ),
            saw(
                "The 2-block model has 34,816 weights, about 3 times `model.py`. Four blocks, 32 wide: 60,032. Four blocks, 64 wide: 218,368. Six blocks, 128 wide: 1,225,728. "
                "And GPT-2 small is about 100 times bigger than even that. The weights grow with the number of blocks (a straight line) and with the width (faster). "
                "Doubling the width makes a block's own weights grow about 4 times, but the embeddings and the output head only 2 times, so the whole model grew about 3.6 times here (60,032 to 218,368). "
                "(The book counts about 163 million for its GPT-2 small, because it keeps the first table and the output head separate. Real GPT-2 shares one table for both, which gives 124 million. Ours keeps them separate, like the book.)"
            ),
            q(
                "Where do the weights sit?",
                "Here's the first config, part by part. The bars add up to the total.",
            ),
            code(
                """import matplotlib.pyplot as plt

from homer_ai.studio_style import ACCENT, INK, studio_axes, studio_figure

parts = {
    "token embedding": count(model.token_embedding),
    "position embedding": count(model.position_embedding),
    "attention (2 blocks)": sum(count(b.attention) for b in model.blocks),
    "feed-forward (2 blocks)": sum(count(b.feed_forward) for b in model.blocks),
    "layer norms (5)": sum(count(b.norm1) + count(b.norm2) for b in model.blocks)
    + count(model.final_norm),
    "output head": count(model.output_head),
}
print("sum of parts:", f"{sum(parts.values()):,}", " model:", f"{count(model):,}")

fig = studio_figure("Where the weights are", "The 2-block, 32-wide model", (10, 6))
ax = studio_axes(fig, (0.28, 0.1, 0.65, 0.7))
names = list(parts)[::-1]
values = [parts[n] for n in names]
ax.barh(names, values, color=ACCENT)
for i, v in enumerate(values):
    ax.text(v + 250, i, f"{v:,}", va="center", color=INK, family="monospace")
ax.set_xlim(0, 1.18 * max(values))
ax.grid(axis="y", visible=False)
ax.set_xlabel("weights")
plt.show()"""
            ),
            saw(
                "The parts add up exactly to the model's total. The two feed-forward steps are the biggest piece, and the position table and the norms are tiny."
            ),
            q(
                "Step 2: how does it write a letter?",
                "Same steps as in [01_04](../01_foundations/01_04_our_model_as_a_black_box.ipynb): feed in the letters so far, take the scores for the **last** letter, "
                "pick the best one (that's called greedy, because no chance is involved), add it to the text, and repeat. Two details. The model can only read `context_length` (24) letters, "
                'so we feed in only the last 24. And `dim=-1` in `argmax` means "best of the 137 scores".\n\n'
                "The model is untrained, so its weights are random. Expect gibberish. Writing real Greek is what chapter 5 is for.\n\n"
                "👉 **Predict first:** will it write Greek words?",
            ),
            code(
                """def generate(model, ids, n_new, context_length):
    model.eval()
    for _ in range(n_new):
        recent = ids[:, -context_length:]  # at most the last 24 letters
        with torch.no_grad():
            logits = model(recent)
        last = logits[:, -1, :]  # the scores for the last letter
        best = last.argmax(dim=-1, keepdim=True)  # greedy: the highest score
        ids = torch.cat([ids, best], dim=1)  # add the new letter
    return ids


start = torch.tensor([tokenizer.encode("ἄνδρα")])
written = generate(model, start, 12, GPT_CONFIG["context_length"])
print(repr(tokenizer.decode(written[0].tolist())))"""
            ),
            saw(
                "The first five letters are our own `ἄνδρα`, and then come 12 letters of untrained gibberish. "
                "For a model that has learned nothing, that's the right result: the code works, and the weights are random."
            ),
            q(
                "Is this model also in the project, so I can train and use it?",
                "Yes. The classes you just read are in `src/homer_ai/gpt.py`, as clean, documented code: `GPTModel`, `TransformerBlock`, `MultiHeadAttention`, `FeedForward`. "
                "There's one difference: the sizes live in a `GPTConfig` (a small frozen record), not in a dictionary. Everything else, including the names of the parts, is the same. "
                "But is the file's model really the one we built? Let's check: the same config, the same number of weights, the same names and shapes for every weight, and the same answer when it has the same weights.\n\n"
                "👉 **Predict first:** will the two have the same number of weights?",
            ),
            code(
                """from homer_ai.gpt import GPTConfig
from homer_ai.gpt import GPTModel as ProjectGPT

project = ProjectGPT(GPTConfig(**GPT_CONFIG))
mine = {name: tuple(w.shape) for name, w in model.state_dict().items()}
theirs = {name: tuple(w.shape) for name, w in project.state_dict().items()}
print("weights, this notebook:", f"{count(model):,}")
print("weights, gpt.py:       ", f"{count(project):,}")
print("same names and shapes? ", mine == theirs)

project.load_state_dict(model.state_dict())  # give it OUR weights
project.eval()
with torch.no_grad():
    print("same logits?           ", torch.equal(model(ids), project(ids)))"""
            ),
            saw(
                "The same 34,816 weights, the same names and shapes, and the same logits once the weights are copied over. "
                "So the model we built here **is** `src/homer_ai/gpt.py`. The rest of the course imports it from there, so the classes don't have to be pasted again."
            ),
            q(
                "Is this the size of the GPT that ships with the course?",
                'No, and it\'s worth seeing the two side by side, because both are called "the GPT". '
                "The config in this notebook is a **teaching size**: small enough to print every shape. "
                "The ready-made GPT in `models/` uses the settings in `GPT_SETTINGS`, at the top of `src/homer_ai/train_homer.py`. Same classes, bigger numbers.\n\n"
                "👉 **Predict first:** the shipped model is 4 times as wide and has twice the blocks. About how many times more weights?",
            ),
            code(
                """from homer_ai.train_homer import GPT_SETTINGS

shipped = GPTConfig(vocab_size=tokenizer.vocab_size, **GPT_SETTINGS)
print(f"{'setting':16}{'this notebook':>14}{'shipped':>10}")
for key in ("context_length", "emb_dim", "n_heads", "n_layers", "drop_rate"):
    print(f"{key:16}{GPT_CONFIG[key]:>14}{getattr(shipped, key):>10}")
shipped_weights = count(ProjectGPT(shipped))
print(f"{'weights':16}{count(model):>14,}{shipped_weights:>10,}")
print(f"the shipped model has {shipped_weights / count(model):.0f} times more weights")"""
            ),
            saw(
                "| Setting | This notebook | Shipped |\n|---|---|---|\n"
                "| `context_length` | 24 | 64 |\n| `emb_dim` | 32 | 128 |\n| `n_heads` | 2 | 4 |\n| `n_layers` | 2 | 4 |\n| `drop_rate` | 0.1 | 0.3 |\n| weights | 34,816 | 835,072 |\n\n"
                "The shipped model reads 64 letters at once, not 24, is 4 times as wide and has twice the blocks. That gives it 24 times more weights, which is still small enough to train on a laptop in about 3 minutes. "
                "Its dropout is higher (0.3) because a model that big can learn Book 1's 20,000 letters by heart. [05_07](../05_training_on_the_odyssey/05_07_train_the_gpt.ipynb) shows why that matters."
            ),
            q(
                "What's left to do?",
                "Training. That's chapter 5's job, with the loss and the learning rate from there, and [05_07](../05_training_on_the_odyssey/05_07_train_the_gpt.ipynb) walks through the GPT's own training loop. "
                "A ready-trained copy ships with the repo, and `uv run homer-train --model gpt` trains your own (about 3 minutes). "
                "`src/homer_ai/model.py` stays as it is: the small one-head model of chapter 1, your starting point.",
            ),
            recap(
                [
                    "The full model is: token + position embeddings, N identical blocks, a final layer norm, and an output head that makes 137 scores per letter.",
                    "Real counts: our `model.py` 11,840; 2 blocks 32 wide 34,816; 6 blocks 128 wide 1,225,728; GPT-2 small about 124 million (quoted).",
                    "An untrained model writes gibberish with `generate()`. Training it is chapter 5.",
                ],
                "[chapter 5: Training on the Odyssey](../05_training_on_the_odyssey/05_00_chapter_overview.ipynb): making the guesses good.",
                can=[
                    "name the parts of a GPT in order, from embeddings to output head",
                    "count a model's weights with `sum(p.numel() for p in model.parameters())`",
                ],
            ),
        ],
    )
