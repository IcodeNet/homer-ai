# ruff: noqa: E501
"""The pictures: drawing code for each diagram, and which notebook shows it.

This file is plain data on purpose. The notebooks put each `DIAGRAM_*` string in
a code cell (see the chapter modules); scripts/export_diagrams.py runs the same
strings to make the PNG files in docs/images/. `DIAGRAMS` lists them.
"""

# ---------------------------------------------------------------- diagrams
# Code of the picture cells (boxes and arrows, drawn with studio_style helpers).
# Each one is placed in its notebook below, after a one-line "look at" sentence.
DIAGRAM_JOURNEY = r"""import matplotlib.pyplot as plt

from homer_ai.studio_style import (
    ACCENT,
    EDGE,
    GOOD,
    INK,
    MUTED,
    arrow,
    chips,
    diagram_axes,
    studio_figure,
    tag,
)

fig = studio_figure("One word's journey", "What happens to ἄνδρα", (9, 7.5))
ax = diagram_axes(fig)


def row(y, labels, colors=None):
    left = 4.0 - (len(labels) * 0.9 - 0.1) / 2  # centre the row on x = 4
    chips(ax, left, y, labels, 0.8, 0.1, colors)


row(6.7, list("ἄνδρα"))
arrow(ax, (4, 6.15), (4, 5.65), "tokenizer (chapter 2)", label_offset=(1.15, 0))
row(5.2, ["58", "38", "29", "42", "26"])
arrow(
    ax,
    (4, 4.65),
    (4, 4.3),
    "model (chapters 3-4), trained in chapter 5",
    label_offset=(2.2, 0),
)

for i, (letter, chance) in enumerate([("␣", 0.26), ("ι", 0.15), ("τ", 0.07)]):
    bar_color = ACCENT if i == 0 else MUTED
    ax.bar(3 + i, chance * 3.5, width=0.6, bottom=2.95, color=bar_color)
    tag(ax, 3 + i, 3.12 + chance * 3.5, f"{chance:.2f}", INK, 9, mono=True)
    tag(ax, 3 + i, 2.7, letter, INK, 12, mono=True)

arrow(
    ax,
    (4, 2.35),
    (4, 1.75),
    "add the best guess, then guess again (chapter 6)",
    label_offset=(2.6, 0),
)
row(1.25, list("ἄνδρα␣"), [EDGE] * 5 + [GOOD])
tag(ax, 4, 0.5, "and so on: ἄνδρα τοι και …", INK, 10)

for y, text in [
    (6.7, "1. the text"),
    (5.2, "2. numbers"),
    (3.4, "3. the model's chances"),
    (1.25, "4. written text"),
]:
    tag(ax, 9.6, y, text, MUTED, 9, ha="right")
plt.show()"""

DIAGRAM_GPT = r"""import matplotlib.pyplot as plt

from homer_ai.studio_style import (
    ACCENT,
    INK,
    MUTED,
    arrow,
    box,
    chips,
    diagram_axes,
    plus,
    studio_figure,
    tag,
)

fig = studio_figure("A full GPT", "Floors, from letters up to scores", (9, 11.5))
ax = diagram_axes(fig, xlim=(-0.8, 9.2))
X = 4.2  # the middle of the building

# ground: the five letters of ἄνδρα as token IDs
chips(ax, X - 2.2, 0.85, ["58", "38", "29", "42", "26"], 0.8, 0.1)
for i, letter in enumerate("ἄνδρα"):
    tag(ax, X - 1.8 + i * 0.9, 0.3, letter, INK, 12, mono=True)
tag(ax, 7.0, 0.85, "token IDs", ha="left")

# floor 1: token embedding + position embedding
arrow(ax, (X, 1.4), (X, 1.85))
tag(ax, X + 0.2, 1.62, "(1, 5)", ha="left", mono=True)
box(ax, 2.6, 2.45, 2.7, 0.8, "token embedding", fontsize=10)
box(ax, 5.8, 2.45, 2.7, 0.8, "position embedding", fontsize=10)
plus(ax, 4.2, 2.45, 0.22)
tag(ax, 2.6, 1.95, "02_05")
tag(ax, 5.8, 1.95, "02_06")

# the block
arrow(ax, (X, 2.95), (X, 3.35))
tag(ax, X + 0.2, 3.15, "(1, 5, 32)", ha="left", mono=True)
box(ax, X, 5.15, 6.2, 3.65, color=ACCENT, lw=2.4, fill="none")
tag(ax, 1.4, 6.65, "BLOCK 1", ACCENT, 11, ha="left", weight="bold")
tag(ax, 7.1, 6.65, "04_04", ACCENT, 8, ha="right")
inner = [
    ("layer norm", "04_01"),
    ("attention", "03_04"),
    ("+", "04_03 shortcut"),
    ("layer norm", "04_01"),
    ("feed-forward", "04_02"),
    ("+", "04_03 shortcut"),
]
for i, (name, where) in enumerate(inner):
    y = 3.75 + i * 0.5
    if name == "+":
        plus(ax, 3.6, y, 0.2)
    else:
        box(ax, 3.6, y, 3.0, 0.38, name, fontsize=9.5)
    tag(ax, 5.4, y, where, MUTED, 8, ha="left")
    if i > 0:
        arrow(ax, (3.6, y - 0.3), (3.6, y - 0.21), head=8)

# more floors, the same again
arrow(ax, (X, 7.05), (X, 7.55))
tag(ax, X + 0.2, 7.3, "(1, 5, 32)", ha="left", mono=True)
text = "BLOCK 2 … BLOCK N: the same floor again"
box(ax, X, 7.95, 6.2, 0.7, text, MUTED, fontsize=9.5)

# the roof
arrow(ax, (X, 8.3), (X, 8.65))
box(ax, X, 8.95, 3.0, 0.5, "final layer norm", fontsize=10)
tag(ax, 6.1, 8.95, "04_01", ha="left")
arrow(ax, (X, 9.2), (X, 9.5))
box(ax, X, 9.8, 3.0, 0.5, "output head", fontsize=10)
tag(ax, 6.1, 9.8, "04_05", ha="left")
arrow(ax, (X, 10.05), (X, 10.4))
tag(ax, X + 0.2, 10.22, "(1, 5, 137)", ha="left", mono=True)
box(ax, X, 10.75, 4.6, 0.6, "137 scores for each letter", fontsize=10)
plt.show()"""

DIAGRAM_QKV = r"""import matplotlib.pyplot as plt

from homer_ai.studio_style import (
    ACCENT,
    GOOD,
    INK,
    MUTED,
    arrow,
    box,
    diagram_axes,
    studio_figure,
    tag,
)

# ν is the reader. The same tiny numbers as above: ν's query against the three keys.
scores = (tiny_Q @ tiny_K.T)[1]  # one score for each of ἄ, ν, δ
shares = torch.softmax(scores, dim=0)
mix = shares @ X  # the values are the rows as they are

fig = studio_figure("Query, key, value", "ν looks at ἄ, ν and δ")
ax = diagram_axes(fig)
for col, name, color in [
    (1.1, "query", ACCENT),
    (3.6, "key", GOOD),
    (5.0, "share", INK),
    (6.6, "value", MUTED),
    (9.0, "blend", INK),
]:
    tag(ax, col, 5.3, name, color, 10, weight="bold")

box(ax, 1.1, 2.9, 1.3, 0.8, "ν  Q", ACCENT, mono=True, lw=2.4)
tag(ax, 1.1, 2.15, "what ν asks for", MUTED, 8)
for i, letter in enumerate("ἄνδ"):
    y = 4.3 - 1.4 * i
    box(ax, 3.6, y, 1.2, 0.7, f"{letter}  K", GOOD, mono=True)
    arrow(
        ax, (1.75, 2.9), (3.0, y), f"{scores[i]:.0f}", ACCENT, 1.8,
        label_offset=(0, 0.22), fontsize=11,
    )
    arrow(ax, (4.25, y), (4.65, y), head=9)
    tag(ax, 5.0, y, f"{shares[i]:.2f}", INK, 11, mono=True)
    arrow(ax, (5.35, y), (5.95, y), head=9)
    box(ax, 6.6, y, 1.2, 0.7, f"{letter}  V", MUTED, mono=True)
    arrow(
        ax, (7.25, y), (8.3, 2.9), color=ACCENT if shares[i] > 0.4 else MUTED,
        width=1 + 9 * shares[i].item(), head=9,
    )
box(ax, 9.0, 2.9, 1.5, 0.9, f"[{mix[0]:.2f},\n {mix[1]:.2f}]", GOOD, mono=True, lw=2.4)
tag(ax, 3.1, 0.75, "score: query · key", MUTED, 9)
tag(ax, 6.1, 0.75, "share: softmax of the scores", MUTED, 9)
tag(ax, 8.9, 1.9, "mix of the values", MUTED, 8)
plt.show()"""

DIAGRAM_BLOCK = r"""import matplotlib.pyplot as plt

from homer_ai.studio_style import (
    ACCENT,
    GOOD,
    INK,
    MUTED,
    arrow,
    box,
    diagram_axes,
    plus,
    studio_figure,
    tag,
)

fig = studio_figure("One block", "Two steps, each with a shortcut", (8, 9.5))
ax = diagram_axes(fig)
X, SIDE, PITCH = 4.0, 7.0, 0.85  # the main road, the shortcut road, box spacing


def step(base, parts, notes, carried):
    # one step: the road splits at `base`; the parts go up the main road;
    # the shortcut goes round them and joins again at the + on top
    top = base + 0.8 + len(parts) * PITCH
    low = base
    for i, (name, note) in enumerate(zip(parts, notes, strict=True)):
        y = base + 0.8 + i * PITCH
        box(ax, X, y, 2.4, 0.55, name, fontsize=10)
        tag(ax, X - 1.4, y, note, MUTED, 8, ha="right")
        arrow(ax, (X, low), (X, y - 0.3), head=9)
        low = y + 0.3
    arrow(ax, (X, low), (X, top - 0.27), head=9)
    plus(ax, X, top, 0.25)
    ax.plot([X, SIDE, SIDE], [base, base, top], color=GOOD, linewidth=2.2)
    arrow(ax, (SIDE, top), (X + 0.27, top), color=GOOD, width=2.2)
    ax.scatter([X], [base], s=40, color=GOOD, zorder=4)
    note = f"shortcut: {carried} goes round\nand is added back"
    tag(ax, SIDE + 0.2, (base + top) / 2, note, GOOD, 9, ha="left")
    return top


tag(ax, X, 0.3, "x   (1, 5, 32)", INK, 11, mono=True, weight="bold")
arrow(ax, (X, 0.55), (X, 0.97))
first = ["layer norm", "attention", "dropout"]
mid = step(1.0, first, ["04_01", "03_04", "03_03"], "x")
tag(ax, X + 0.5, mid + 0.4, "x′", INK, 11)
arrow(ax, (X, mid + 0.27), (X, mid + 0.7))
parts = ["layer norm", "feed-forward", "dropout"]
top = step(mid + 0.7, parts, ["04_01", "04_02", "03_03"], "x′")
arrow(ax, (X, top + 0.27), (X, top + 0.75), color=ACCENT, width=2.2)
text = "output   (1, 5, 32): the same shape as x"
tag(ax, X, top + 1.05, text, ACCENT, 11, weight="bold")
plt.show()"""

DIAGRAM_LOOP = r"""import matplotlib.pyplot as plt

from homer_ai.studio_style import (
    ACCENT,
    BAD,
    EDGE,
    GOOD,
    INK,
    MUTED,
    arrow,
    box,
    bracket,
    chips,
    diagram_axes,
    studio_figure,
    tag,
)

fig = studio_figure("The writing loop", "One new letter on every round", (10, 6.8))
ax = diagram_axes(fig)

# the text so far: 30 letters, and the new letter in green
text = "ἄνδρα μοι ἔννεπε, μοῦσα, πολύτ"
letters = [c if c != " " else "␣" for c in text]
colors = [BAD] * 6 + [EDGE] * 24 + [GOOD]
chips(ax, 0.4, 4.7, [*letters, "␣"], 0.28, 0.02, colors, fontsize=9)
bracket(ax, 0.4, 0.4 + 6 * 0.3 - 0.02, 5.1, "out of view", BAD, fontsize=9)
bracket(ax, 0.4 + 6 * 0.3, 0.4 + 30 * 0.3 - 0.02, 5.1, "the last 24 letters", ACCENT)
tag(ax, 9.83, 4.15, "new", GOOD, 9)

# the loop: four steps, then back to the text
steps = [
    (1.2, "1. read the\nlast 24 letters"),
    (3.7, "2. model:\n137 scores"),
    (6.2, "3. softmax:\nchances"),
    (8.7, "4. pick the best,\nadd it to the end"),
]
for x, label in steps:
    box(ax, x, 3.0, 1.9, 0.9, label, fontsize=9.5)
for x in (1.2, 3.7, 6.2):
    arrow(ax, (x + 0.97, 3.0), (x + 1.53, 3.0))
arrow(ax, (2.7, 4.45), (1.8, 3.5), color=ACCENT, width=2, curved=0.2)
arrow(ax, (8.7, 3.5), (9.83, 4.5), color=GOOD, width=2.2, curved=-0.25)

# the chances after ἄνδρα
arrow(ax, (6.2, 2.5), (6.2, 2.05), head=9)
for i, (letter, chance) in enumerate([("␣", 0.26), ("ι", 0.15), ("τ", 0.07)]):
    x = 5.6 + 0.6 * i
    ax.bar(x, chance * 3, width=0.45, bottom=0.9, color=ACCENT if i == 0 else MUTED)
    tag(ax, x, 0.65, letter, INK, 11, mono=True)
    tag(ax, x, 0.9 + chance * 3 + 0.15, f"{chance:.2f}", MUTED, 8, mono=True)
tag(ax, 8.7, 1.9, "greedy: the biggest\nchance wins", ACCENT, 9)
tag(ax, 1.2, 1.9, "older letters fall out\nof view, like a short\nmemory", BAD, 9)
plt.show()"""

DIAGRAM_TRAINING = r"""import matplotlib.pyplot as plt

from homer_ai.studio_style import (
    ACCENT,
    EDGE,
    INK,
    MUTED,
    box,
    diagram_axes,
    ring,
    studio_figure,
    tag,
)

fig = studio_figure("The training step", "Four things, again and again", (9, 6.5))
ax = diagram_axes(fig)
CX, CY = 5.0, 3.5

boxes = [
    ("1. guess", "logits = model(inputs)"),
    ("2. measure the loss", "F.cross_entropy(logits, targets)"),
    ("3. work out the nudges", "loss.backward()"),
    ("4. nudge the weights", "optimizer.step()"),
]
centres = ring(
    ax, CX, CY, 3.0, 1.9, [""] * 4, [EDGE, ACCENT, EDGE, EDGE], box_w=3.4, box_h=1.1
)
for (x, y), (name, code_line) in zip(centres, boxes, strict=True):
    tag(ax, x, y + 0.2, name, INK, 11, weight="bold")
    tag(ax, x, y - 0.22, code_line, MUTED, 8, mono=True)

tag(ax, CX, CY + 0.35, "one round per chunk", INK, 10)
tag(ax, CX, CY - 0.05, "all the chunks = 1 epoch", ACCENT, 10)
tag(ax, CX, CY - 0.45, "ἄνδρ → α is one flashcard", MUTED, 9)
box(ax, CX, 0.4, 5.0, 0.5, "", MUTED, dashed=True)
tag(ax, CX, 0.4, "before step 3: optimizer.zero_grad() wipes old nudges", MUTED, 8.5)
plt.show()"""

DIAGRAM_LOOKUP = r"""import matplotlib.pyplot as plt

from homer_ai.studio_style import (
    ACCENT,
    EDGE,
    INK,
    MUTED,
    arrow,
    box,
    diagram_axes,
    studio_figure,
    tag,
)

word = "ἄνδρα"
word_ids = tokenizer.encode(word)
fig = studio_figure("Embedding lookup", "Five IDs pull out five rows", (9, 6.5))
ax = diagram_axes(fig)

tag(ax, 2.0, 5.95, "the embedding table (137 × 32)", INK, 10, weight="bold")
tag(ax, 8.2, 5.95, "what the model reads (5 × 32)", INK, 10, weight="bold")
table_y = {}
for k, token_id in enumerate(sorted(word_ids)):
    y = 5.2 - 0.95 * k
    table_y[token_id] = y
    char = tokenizer.id_to_char[token_id]
    color = ACCENT if token_id == word_ids[0] else EDGE
    box(ax, 2.0, y, 3.0, 0.55, f"{token_id:>3} {char}  [32 numbers]", color, mono=True)
    if k < 4:
        tag(ax, 2.0, y - 0.47, "⋮", MUTED, 10)
for k, token_id in enumerate(word_ids):
    y = 4.9 - 0.8 * k
    color = ACCENT if k == 0 else EDGE
    label = f"{word[k]}  [32 numbers]"
    box(ax, 8.2, y, 3.0, 0.55, label, color, mono=True)
    arrow(ax, (3.55, table_y[token_id]), (6.65, y), color=ACCENT if k == 0 else MUTED)
tag(ax, 5.1, 1.45, "look up each ID: take its row", MUTED, 9)
plt.show()"""

DIAGRAM_BLEND = r"""import matplotlib.pyplot as plt

from homer_ai.studio_style import (
    ACCENT,
    GOOD,
    MUTED,
    arrow,
    box,
    diagram_axes,
    studio_figure,
    tag,
)

share = weights.tolist()
fig = studio_figure("A blend with shares", "δ gets the widest arrow")
ax = diagram_axes(fig)
tag(ax, 5, 5.55, "the letters it looks at", MUTED, 9)
for i, letter in enumerate("ἄνδ"):
    cx = 2.2 + 2.8 * i
    color = ACCENT if i == 2 else MUTED
    box(ax, cx, 4.7, 1.7, 0.9, f"{letter}\n{X[i].int().tolist()}", color, mono=True)
    arrow(
        ax,
        (cx, 3.65),
        (3.9 + 1.1 * i, 1.85),
        color=color,
        width=1 + 10 * share[i],
        head=12,
    )
    tag(ax, cx, 4.0, f"share {share[i]:.2f}", color, 10, mono=True)
shown = ", ".join(f"{v:.3f}" for v in context.tolist())
box(ax, 5, 1.2, 4.6, 1.1, f"context vector\n[{shown}]", GOOD, mono=True, lw=2.4)
tag(ax, 5, 0.25, "a wider arrow means a bigger share", MUTED, 9)
plt.show()"""

DIAGRAM_MCP = r"""import matplotlib.pyplot as plt

from homer_ai.studio_style import (
    ACCENT,
    GOOD,
    INK,
    MUTED,
    arrow,
    box,
    diagram_axes,
    studio_figure,
    tag,
)

fig = studio_figure("Claude Code and our server", "Who says what, in order", (9, 6.5))
ax = diagram_axes(fig)
who = [(1.5, "you + Claude Code"), (5.0, "our MCP server"), (8.5, "our model")]
for x, name in who:
    box(ax, x, 5.7, 2.7, 0.6, name, fontsize=10)
    ax.plot([x, x], [5.35, 0.5], color=MUTED, linewidth=1, linestyle="--", zorder=0)


def say(y, a, b, text, color, step=None):
    arrow(ax, (a, y), (b, y), color=color, width=2)
    tag(ax, (a + b) / 2, y + 0.22, text, color, 9)
    if step:
        tag(ax, a + (0.2 if b > a else -0.2), y - 0.2, step, INK, 10, weight="bold")


say(4.7, 1.5, 5.0, "what tools do you have?", MUTED, "1")
say(3.9, 5.0, 1.5, "consult_homer_llm: description + arguments", ACCENT, "2")
say(3.0, 1.5, 5.0, 'consult_homer_llm("ἄνδρα μοι")', MUTED, "3")
say(2.4, 5.0, 8.5, "write letters", MUTED)
say(1.8, 8.5, 5.0, "Greek letters", MUTED)
say(1.0, 5.0, 1.5, "the text, plus real lines", GOOD, "4")
tag(ax, 5.0, 0.2, "messages are JSON text, sent through stdio", MUTED, 9)
plt.show()"""

DIAGRAM_GRADIENT = r"""import matplotlib.pyplot as plt

from homer_ai.studio_style import (
    ACCENT,
    BAD,
    GOOD,
    INK,
    MUTED,
    arrow,
    diagram_axes,
    stack_of_floors,
    studio_figure,
    tag,
)

fig = studio_figure("Shortcuts", "The learning signal, layer by layer", (11, 6.5))
ax = diagram_axes(fig, xlim=(0, 12))
top_size = max(*without, *with_shortcuts)
panels = [
    (3.0, "without shortcuts", BAD, without),
    (9.0, "with shortcuts", GOOD, with_shortcuts),
]
for x, name, color, sizes in panels:
    tag(ax, x, 6.0, name, color, 12, weight="bold")
    floors = [(f"layer {k}", "", MUTED) for k in range(1, 6)]
    ys = stack_of_floors(ax, x, 0.25, 2.0, floors, floor_h=0.5, gap=0.55)
    arrow(ax, (x, 5.7), (x, ys[4] + 0.3), "mistake", INK, 1.5, label_offset=(0.7, 0))
    for k in range(5):
        ratio = sizes[k] / top_size  # the width of the arrow
        local = sizes[k] / max(sizes)  # the colour: how this layer compares
        shade = ACCENT if local > 0.5 else (MUTED if local > 0.1 else BAD)
        if color == GOOD:
            shade = GOOD
        if k < 4:
            start, end = (x, ys[k + 1] - 0.3), (x, ys[k] + 0.3)
            arrow(ax, start, end, color=shade, width=1.5 + 7 * ratio**0.5, head=9)
        tag(ax, x + 1.2, ys[k], f"{sizes[k]:.4f}", shade, 10, ha="left", mono=True)
for k in (1, 2, 3, 4):  # a shortcut skips layers 1 to 4 (layer 5 is the last)
    y = 0.5 + (k - 1) * 1.05  # the middle of layer k
    arrow(ax, (8.3, y + 0.6), (8.3, y - 0.6), color=GOOD, width=2, curved=0.9, head=9)
tag(ax, 6.0, 3.0, "gradient size\nis written beside\neach layer", MUTED, 9)
plt.show()"""

DIAGRAM_WINDOW = r"""import matplotlib.pyplot as plt

from homer_ai.studio_style import (
    ACCENT,
    GOOD,
    INK,
    MUTED,
    bracket,
    chips,
    diagram_axes,
    studio_figure,
    tag,
)

fig = studio_figure("The sliding window", "Inputs, and the next letter as the answer")
ax = diagram_axes(fig)
phrase = "ἄνδρα μοι"
letters = [c if c != " " else "␣" for c in phrase]
LEFT, STEP = 1.3, 0.8  # where the first chip starts, and the distance between chips

tag(ax, 0.1, 4.6, "the text", MUTED, 9, ha="left")
chips(ax, LEFT, 4.6, letters, 0.7, 0.1)
bracket(ax, LEFT, LEFT + 4 * STEP - 0.1, 4.05, "inputs: a chunk of 4", ACCENT, True)
bracket(ax, LEFT + STEP, LEFT + 5 * STEP - 0.1, 5.15, "targets: one later", GOOD)

tag(ax, 0.1, 3.0, "flashcards", MUTED, 9, ha="left")
for k in range(1, 5):
    y = 3.0 - 0.8 * (k - 1)
    colors = [ACCENT] * k + [GOOD]
    chips(ax, LEFT, y, letters[: k + 1], 0.7, 0.1, colors)
    label = f"{phrase[:k]} → {phrase[k]}"
    tag(ax, LEFT + (k + 1) * STEP + 0.3, y, label, INK, 10, ha="left")
plt.show()"""

DIAGRAM_DOTS = r"""import matplotlib.pyplot as plt

from homer_ai.studio_style import (
    ACCENT,
    GOOD,
    INK,
    MUTED,
    diagram_axes,
    dot,
    studio_figure,
    tag,
)

with torch.no_grad():
    outputs = layer(x).tolist()
fig = studio_figure("One layer", "3 numbers in, 2 numbers out", (8, 6))
ax = diagram_axes(fig)
inputs = [(2.0, 4.6), (2.0, 3.2), (2.0, 1.8)]
outs = [(7.0, 4.0), (7.0, 2.4)]
for (x0, y0) in inputs:
    for (x1, y1) in outs:
        ax.plot([x0, x1], [y0, y1], color=MUTED, linewidth=1.4, zorder=1)
for (px, py), value in zip(inputs, [1, 2, 3], strict=True):
    dot(ax, px, py, str(value), color=MUTED)
for (px, py), value in zip(outs, outputs, strict=True):
    dot(ax, px, py, f"{value:.2f}", r=0.5, color=ACCENT, fontsize=10)
    ax.plot([px + 0.5, px + 1.2], [py, py], color=GOOD, linewidth=1.4)
    dot(ax, px + 1.5, py, "b", r=0.3, color=GOOD, fontsize=10)
tag(ax, 2.0, 5.6, "3 numbers in", INK, 11, weight="bold")
tag(ax, 7.0, 5.1, "2 numbers out", INK, 11, weight="bold")
tag(ax, 4.5, 0.9, "6 lines = 6 weights", MUTED, 10)
tag(ax, 8.2, 3.2, "2 bias dots (b)\n= 2 biases", GOOD, 9)
tag(ax, 4.5, 0.4, "6 weights + 2 biases = 8", ACCENT, 11, weight="bold")
plt.show()"""


DIAGRAM_RAG = r"""import matplotlib.pyplot as plt

from homer_ai.studio_style import (
    ACCENT,
    GOOD,
    INK,
    MUTED,
    arrow,
    box,
    diagram_axes,
    studio_figure,
    tag,
)

fig = studio_figure("Grounded answers", "Fetch the page first, then answer", (10, 6.6))
ax = diagram_axes(fig)

box(ax, 1.6, 4.9, 2.8, 0.9, "a question\nin English", INK, fontsize=10)
box(ax, 5.0, 4.9, 2.6, 0.9, "search\n(BM25)", ACCENT, fontsize=11, lw=2.2)
box(ax, 8.4, 4.9, 2.8, 0.9, "the 5 best\npassages", GOOD, fontsize=10)
arrow(ax, (3.0, 4.9), (3.7, 4.9), color=MUTED, width=2)
arrow(ax, (6.3, 4.9), (7.0, 4.9), color=GOOD, width=2)
tag(ax, 5.0, 5.65, "1. retrieve", ACCENT, 10, weight="bold")

box(ax, 5.0, 3.2, 4.4, 0.75, "both poems, cut into 2,140 passages", MUTED,
    fontsize=9.5, dashed=True)
arrow(ax, (5.0, 3.6), (5.0, 4.4), color=MUTED)
tag(ax, 5.0, 2.62, "each one remembers its place: odyssey 9:24", MUTED, 9, mono=True)

box(ax, 8.4, 1.3, 2.8, 1.0, "Claude reads them\nand answers", INK, fontsize=10)
arrow(ax, (8.4, 4.4), (8.4, 1.85), color=GOOD, width=2)
arrow(ax, (1.6, 4.4), (1.6, 1.3), color=MUTED, dashed=True)
arrow(ax, (1.6, 1.3), (2.75, 1.3), color=MUTED, dashed=True)
tag(ax, 0.85, 2.9, "the question\ngoes along", MUTED, 8.5)
tag(ax, 8.4, 0.45, "2. answer", ACCENT, 10, weight="bold")
box(ax, 4.6, 1.3, 3.6, 1.0, '"Noman."\n[odyssey 9:24]', GOOD, mono=True, fontsize=10.5,
    lw=2.2)
arrow(ax, (7.0, 1.3), (6.45, 1.3), color=GOOD, width=2)
tag(ax, 4.6, 0.45, "an answer you can check", GOOD, 9)
plt.show()"""

DIAGRAM_BM25 = r"""import matplotlib.pyplot as plt

from homer_ai.studio_style import (
    ACCENT,
    BAD,
    GOOD,
    INK,
    MUTED,
    studio_axes,
    studio_figure,
)

K1, B = 1.5, 0.75


def part(tf, length, average=100):
    return tf * (K1 + 1) / (tf + K1 * (1 - B + B * length / average))


counts = range(0, 11)
fig = studio_figure("BM25", "The fifth Noman counts for less than the first")
ax = studio_axes(fig)
ax.plot(counts, list(counts), color=MUTED, linewidth=2, linestyle="--")
ax.plot(counts, [part(tf, 50) for tf in counts], "o-", color=GOOD, linewidth=2.5)
ax.plot(counts, [part(tf, 100) for tf in counts], "o-", color=ACCENT, linewidth=2.5)
ax.plot(counts, [part(tf, 300) for tf in counts], "o-", color=BAD, linewidth=2.5)
ax.axhline(K1 + 1, color=MUTED, linewidth=1, linestyle=":")
ax.text(10, 2.62, "the ceiling: k1 + 1 = 2.5", color=MUTED, fontsize=9, ha="right")
ax.text(3.2, 4.0, "plain counting", color=MUTED, fontsize=10)
labels = [
    (part(10, 50) + 0.06, "short passage", GOOD),
    (part(10, 100) - 0.08, "average length", ACCENT),
    (part(10, 300), "3 times longer", BAD),
]
for y, text, color in labels:
    ax.text(10.2, y, text, color=color, fontsize=9.5, va="center")
ax.set_xlim(-0.3, 12.6)
ax.set_ylim(0, 4.5)
ax.set_xlabel("how often the word is in the passage (tf)")
ax.set_ylabel("what it adds to the score (before idf)")
ax.tick_params(colors=INK)
plt.show()"""


# ---------------------------------------------------------------- which picture goes where
# Notebook that teaches each picture, relative to the project folder.
NB = "notebooks/"

# Inputs a diagram needs, computed the way the notebook computes them.
PRELUDE_TORCH = "import torch\nfrom torch import nn\n"
PRELUDE_TINY = (
    PRELUDE_TORCH
    + """
X = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])  # one invented row per letter
swap = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
tiny_Q = X @ swap
tiny_K = X
"""
)
PRELUDE_BLEND = (
    PRELUDE_TORCH
    + """
letters = ["ἄ", "ν", "δ"]
X = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
scores = X @ X[2]
weights = torch.softmax(scores, dim=0)
context = weights @ X
"""
)
PRELUDE_LOOKUP = """from homer_ai.tokenizer import CharTokenizer
from homer_ai.train_homer import BOOKS_DIR, load_books

tokenizer = CharTokenizer(load_books(BOOKS_DIR)[0])  # the vocabulary of Book 1
"""
PRELUDE_DOTS = (
    PRELUDE_TORCH
    + """
torch.manual_seed(42)
layer = nn.Linear(3, 2)
x = torch.tensor([1.0, 2.0, 3.0])
"""
)
PRELUDE_GRADIENT = (
    PRELUDE_TORCH
    + """
class DeepStack(nn.Module):
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
                x = x + out
            else:
                x = out
        return x


sizes = [3, 3, 3, 3, 3, 1]
sample = torch.tensor([[1.0, 0.0, -1.0]])
torch.manual_seed(123)
plain = DeepStack(sizes, use_shortcut=False)
torch.manual_seed(123)
shortcut = DeepStack(sizes, use_shortcut=True)


def gradient_sizes(model):
    loss = nn.MSELoss()(model(sample), torch.tensor([[0.0]]))
    loss.backward()
    return [layer[0].weight.grad.abs().mean().item() for layer in model.layers]


without = gradient_sizes(plain)
with_shortcuts = gradient_sizes(shortcut)
"""
)

DIAGRAM_PRETRAINED = r"""import matplotlib.pyplot as plt

from homer_ai.studio_style import (
    ACCENT,
    EDGE,
    GOOD,
    INK,
    MUTED,
    arrow,
    box,
    diagram_axes,
    studio_figure,
    tag,
)

fig = studio_figure("GPT-2's weights", "The same numbers, under our names", (10, 8.6))
ax = diagram_axes(fig)
tag(ax, 2.3, 7.1, "in GPT-2's file", INK, 11, weight="bold")
tag(ax, 7.9, 7.1, "in our GPTModel (gpt.py)", INK, 11, weight="bold")

# (GPT-2's name, its height, [(our name, its height)], colour)
rows = [
    ("wte", 6.5, [("token_embedding", 6.5), ("output_head  (a copy)", 5.85)], GOOD),
    ("wpe", 5.2, [("position_embedding", 5.2)], EDGE),
    ("h.N.ln_1", 4.55, [("blocks.N.norm1", 4.55)], EDGE),
    (
        "h.N.attn.c_attn",
        3.25,
        [
            ("…attention.query", 3.9),
            ("…attention.key", 3.25),
            ("…attention.value", 2.6),
        ],
        ACCENT,
    ),
    ("h.N.attn.c_proj", 1.95, [("…attention.out_proj", 1.95)], EDGE),
    ("h.N.ln_2", 1.3, [("blocks.N.norm2", 1.3)], EDGE),
    ("h.N.mlp.c_fc, c_proj", 0.65, [("…feed_forward.0, .2", 0.65)], EDGE),
    ("ln_f", 0.0, [("final_norm", 0.0)], EDGE),
]
for gpt2_name, y, our_names, color in rows:
    box(ax, 2.3, y, 3.3, 0.48, gpt2_name, color, mono=True, fontsize=10)
    for our_name, our_y in our_names:
        box(ax, 7.9, our_y, 3.5, 0.48, our_name, color, mono=True, fontsize=10)
        line = MUTED if color == EDGE else color
        arrow(ax, (3.95, y), (6.15, our_y), color=line)
tag(ax, 5.05, 3.75, "one table,", ACCENT, 9)
tag(ax, 5.05, 2.72, "cut in three", ACCENT, 9)
ax.plot([0.35, 0.35], [0.45, 4.75], color=MUTED, linewidth=2, zorder=3)
ax.text(0.12, 2.6, "N = 0 … 11: the 12 blocks", color=MUTED, fontsize=9,
        rotation=90, ha="center", va="center")
ax.set_ylim(-0.45, 7.45)
plt.show()"""

DIAGRAM_LORA = r"""import matplotlib.pyplot as plt

from homer_ai.studio_style import (
    ACCENT,
    GOOD,
    INK,
    MUTED,
    arrow,
    box,
    diagram_axes,
    plus,
    studio_figure,
    tag,
)

fig = studio_figure("LoRA", "A frozen table, and two thin ones beside it", (10, 6.4))
ax = diagram_axes(fig)

box(ax, 0.9, 2.9, 1.2, 0.9, "x\n768", INK, mono=True)
box(ax, 5.0, 4.3, 3.0, 1.5, "W\n768 × 768", MUTED, mono=True, fontsize=12, dashed=True)
tag(ax, 5.0, 5.35, "frozen: 589,824 numbers, never nudged", MUTED, 9.5)
box(ax, 3.9, 1.5, 1.9, 0.5, "A  8 × 768", GOOD, mono=True, lw=2.2)
box(ax, 6.3, 1.5, 0.6, 1.5, "B", GOOD, mono=True, lw=2.2)
tag(ax, 6.3, 0.52, "768 × 8", GOOD, 9.5, mono=True)
tag(ax, 3.9, 0.95, "starts random", GOOD, 9)
tag(ax, 6.3, 0.2, "starts at zero", GOOD, 9)
tag(ax, 5.0, 2.6, "trained: 2 × 8 × 768 = 12,288 numbers", GOOD, 9.5)
arrow(ax, (1.5, 3.1), (3.5, 4.2), color=MUTED)
arrow(ax, (1.5, 2.7), (2.95, 1.6), color=GOOD)
arrow(ax, (4.85, 1.5), (6.0, 1.5), "8", color=GOOD, mono=True)
plus(ax, 8.1, 2.9)
arrow(ax, (6.5, 4.2), (7.9, 3.1), color=MUTED)
arrow(ax, (6.6, 1.6), (7.9, 2.7), color=GOOD)
arrow(ax, (8.35, 2.9), (8.9, 2.9), color=ACCENT)
box(ax, 9.45, 2.9, 1.0, 0.9, "out\n768", ACCENT, mono=True)
plt.show()"""

# (file name without .png, DIAGRAM_ constant, inputs, notebook, alt text)
DIAGRAM_LADDER = r"""import matplotlib.pyplot as plt

from homer_ai.studio_style import (
    ACCENT,
    EDGE,
    GOOD,
    INK,
    MUTED,
    arrow,
    diagram_axes,
    stack_of_floors,
    studio_figure,
    tag,
)

fig = studio_figure(
    "Your domain", "Four rungs: climb only as far as you need", (10, 6.8)
)
ax = diagram_axes(fig)

# (title, what it is, colour), from the bottom rung up
rungs = [
    ("1. Prompt a big model", "ask well; nothing of yours is inside it", EDGE),
    (
        "2. RAG + tools over MCP",
        "your documents and systems, looked up for each answer",
        GOOD,
    ),
    (
        "3. Fine-tune a small open model",
        "your examples change the weights; it runs on your machines",
        ACCENT,
    ),
    ("4. Train from scratch", "a model of your own, from nothing", EDGE),
]
# why a firm takes each step
reasons = [
    ("start here", MUTED),
    ("exact answers, each with its source\n10_01 and 10_03", GOOD),
    (
        "data that can't leave, cost at volume,\na frozen model for audits: 10_02",
        ACCENT,
    ),
    ("rare: it needs more text and money\nthan most firms have", MUTED),
]
middles = stack_of_floors(ax, 3.1, 0.35, 5.4, rungs, floor_h=1.05, gap=0.22)
tag(ax, 8.05, 5.6, "you climb for…", INK, 10.5, weight="bold")
for y, (text, color) in zip(middles, reasons, strict=True):
    arrow(ax, (5.95, y), (6.45, y), color=color)
    tag(ax, 8.05, y, text, color, 9)
arrow(ax, (0.2, 0.5), (0.2, 5.2), color=MUTED, width=2)
ax.text(
    0.02,
    2.85,
    "more cost, more work",
    color=MUTED,
    fontsize=9,
    rotation=90,
    ha="center",
    va="center",
)
plt.show()"""

DIAGRAM_SUPPORT = r"""import matplotlib.pyplot as plt

from homer_ai.studio_style import (
    ACCENT,
    BAD,
    GOOD,
    INK,
    MUTED,
    arrow,
    box,
    diagram_axes,
    studio_figure,
    tag,
)

fig = studio_figure("Support desk", "Look it up, or hand it over", (10, 6.8))
ax = diagram_axes(fig)

box(ax, 1.5, 5.15, 2.5, 0.9, "a customer's\nmessage", INK, fontsize=10)
box(
    ax,
    5.0,
    5.15,
    3.0,
    0.9,
    "the assistant\n(a big model)",
    ACCENT,
    fontsize=10.5,
    lw=2.2,
)
arrow(ax, (2.75, 5.15), (3.5, 5.15), color=MUTED, width=2)
tag(ax, 8.3, 5.15, "three tools,\nover MCP", MUTED, 9)

box(
    ax,
    1.9,
    3.45,
    2.9,
    0.95,
    "search_terms\nthe product sheets",
    GOOD,
    mono=True,
    fontsize=9.5,
)
box(
    ax, 5.0, 3.45, 2.9, 0.95, "get_rate\nthe rates table", GOOD, mono=True, fontsize=9.5
)
box(
    ax,
    8.1,
    3.45,
    2.9,
    0.95,
    "escalate\ncomplaint, vulnerable,\nadvice, bereavement",
    BAD,
    mono=True,
    fontsize=8.5,
)
arrow(ax, (4.3, 4.7), (2.4, 3.95), color=GOOD)
arrow(ax, (5.0, 4.7), (5.0, 3.95), color=GOOD)
arrow(ax, (5.7, 4.7), (7.6, 3.95), color=BAD)

box(
    ax,
    3.45,
    1.7,
    5.4,
    0.95,
    'an answer with its source\n"2.85%"  [rates#easy-access/2026-09-01]',
    GOOD,
    mono=True,
    fontsize=9.5,
    lw=2.2,
)
box(
    ax, 8.1, 1.7, 2.9, 0.95, "no answer:\na person takes over", BAD, fontsize=10, lw=2.2
)
arrow(ax, (1.9, 2.95), (2.6, 2.2), color=GOOD, width=2)
arrow(ax, (5.0, 2.95), (4.3, 2.2), color=GOOD, width=2)
arrow(ax, (8.1, 2.95), (8.1, 2.2), color=BAD, width=2)

box(
    ax,
    5.0,
    0.4,
    9.2,
    0.55,
    "every tool call adds one line to the audit trail",
    MUTED,
    fontsize=9.5,
    dashed=True,
)
plt.show()"""

DIAGRAMS = [
    (
        "word-journey",
        "DIAGRAM_JOURNEY",
        "",
        NB + "00_start_here/00_00_course_map.ipynb",
        "The Greek word ἄνδρα on its way through the model: characters, IDs, model scores, and the best guess added as a new letter",
    ),
    (
        "neuron-layer",
        "DIAGRAM_DOTS",
        PRELUDE_DOTS,
        NB + "01_foundations/01_02_neural_networks.ipynb",
        "One neural-network layer: 3 numbers in, 6 weight lines, 2 biases, 2 numbers out",
    ),
    (
        "sliding-window",
        "DIAGRAM_WINDOW",
        "",
        NB + "02_text_to_numbers/02_04_sliding_window.ipynb",
        "A window sliding over the token IDs of a line of Homer, giving inputs and targets shifted by one",
    ),
    (
        "embedding-lookup",
        "DIAGRAM_LOOKUP",
        PRELUDE_LOOKUP,
        NB + "02_text_to_numbers/02_05_token_embeddings.ipynb",
        "Each ID of ἄνδρα picks one row of the embedding table and stacks it into a 5 by 32 table",
    ),
    (
        "attention-blend",
        "DIAGRAM_BLEND",
        PRELUDE_BLEND,
        NB + "03_attention/03_01_simple_attention.ipynb",
        "Attention shares as arrow widths, blending three letters into one context vector",
    ),
    (
        "qkv",
        "DIAGRAM_QKV",
        PRELUDE_TINY,
        NB + "03_attention/03_02_trainable_attention.ipynb",
        "Query against keys gives scores, scores become shares, and the shares mix the values",
    ),
    (
        "gpt-floors",
        "DIAGRAM_GPT",
        "",
        NB + "04_building_the_model/04_00_chapter_overview.ipynb",
        "The full GPT as a building: embeddings at the bottom, repeated blocks of attention and feed-forward, scores at the top",
    ),
    (
        "shortcut-gradients",
        "DIAGRAM_GRADIENT",
        PRELUDE_GRADIENT,
        NB + "04_building_the_model/04_03_shortcuts.ipynb",
        "The learning signal reaching each layer, with and without shortcut connections",
    ),
    (
        "one-block",
        "DIAGRAM_BLOCK",
        "",
        NB + "04_building_the_model/04_04_one_block.ipynb",
        "One transformer block step by step: layer norm, attention, shortcut, layer norm, feed-forward, shortcut",
    ),
    (
        "training-ring",
        "DIAGRAM_TRAINING",
        "",
        NB + "05_training_on_the_odyssey/05_00_chapter_overview.ipynb",
        "The training ring: guess, measure the loss, work out the nudges, nudge, and repeat",
    ),
    (
        "generation-loop",
        "DIAGRAM_LOOP",
        "",
        NB + "06_making_it_talk/06_01_writing_greek.ipynb",
        "The writing loop: score the next letter, pick one, add it to the text, and repeat",
    ),
    (
        "mcp-flow",
        "DIAGRAM_MCP",
        "",
        NB + "06_making_it_talk/06_03_the_mcp_server.ipynb",
        "Claude asks, the harness calls the MCP server, our model writes Greek, and the answer goes back to Claude",
    ),
    (
        "gpt2-names",
        "DIAGRAM_PRETRAINED",
        "",
        NB + "08_advanced_fine_tuning/08_01_loading_gpt2.ipynb",
        "GPT-2's tensor names on the left, the names of our GPTModel on the right; the fused query, key and value table is cut in three",
    ),
    (
        "lora",
        "DIAGRAM_LORA",
        "",
        NB + "08_advanced_fine_tuning/08_04_lora.ipynb",
        "LoRA: the input goes through the frozen table W and through two thin trained tables A and B, and the two results are added",
    ),
    (
        "grounded-answers",
        "DIAGRAM_RAG",
        "",
        NB + "09_advanced_grounded_answers/09_00_chapter_overview.ipynb",
        "A question goes to a search over both poems, the five best passages come back with their book and line, and Claude answers from them with a citation",
    ),
    (
        "bm25-saturation",
        "DIAGRAM_BM25",
        "",
        NB + "09_advanced_grounded_answers/09_01_chunking_and_bm25.ipynb",
        "What a word adds to a BM25 score as it appears more often: it flattens towards a ceiling, sooner in a long passage, while plain counting keeps rising",
    ),
    (
        "domain-ladder",
        "DIAGRAM_LADDER",
        "",
        NB + "10_advanced_your_domain/10_00_chapter_overview.ipynb",
        "Four rungs from the bottom up: prompt a big model, RAG and tools over MCP, fine-tune a small open model, train from scratch, each with the reason a firm climbs to it",
    ),
    (
        "support-flow",
        "DIAGRAM_SUPPORT",
        "",
        NB + "10_advanced_your_domain/10_01_support_with_citations.ipynb",
        "A customer's message reaches the assistant, which searches the product sheets, reads the rates table, or escalates; the result is an answer with its source or a handover to a person, and every tool call is logged",
    ),
]
