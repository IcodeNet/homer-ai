# ruff: noqa: E501
"""Chapter 1, Foundations: the ideas every stage uses."""

from .cells import (
    CH1_HERE,
    code,
    exercise,
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
from .diagrams import DIAGRAM_DOTS
from .setup_code import BLACK_BOX_SETUP, GENERATE


def build():
    # =====================================================================
    # 01_foundations/01_00_chapter_overview.ipynb
    # =====================================================================
    save(
        "01_foundations/01_00_chapter_overview.ipynb",
        [
            title("01_00 Foundations (overview)", "chapter 1 and appendix A"),
            where(None, ["src/homer_ai/model.py"], here=CH1_HERE),
            learn(
                [
                    "What a language model is, and what it isn't",
                    "Why guessing the next token is the whole idea",
                    'What makes a model "large", and how small ours is',
                    "Which notebook to open for each idea",
                ]
            ),
            words(
                [
                    (
                        "language model",
                        "a program that, given some text, guesses what comes next",
                        "after `ἄνδρα μο` it guesses `ι`",
                    ),
                    (
                        "weight",
                        "one adjustable number inside a model; learning means changing the weights",
                        "our model has 11,840",
                    ),
                    (
                        "training",
                        "showing the model text again and again, and adjusting its weights so its guesses get better",
                        "800 rounds on Book 1",
                    ),
                ]
            ),
            md("## 3. The idea"),
            q(
                "What is a language model?",
                'Think of the children\'s game "hot and cold": something is hidden, you take a step, and a friend says "warmer" or "colder". '
                'You don\'t need to understand anything about the room to win. You just keep moving the way that gets you "warmer". '
                "A language model plays that game with text: it guesses what comes next, gets told how far off it was, and shifts its numbers a little so the next guess is warmer. "
                "Do that thousands of times and the guesses get good. "
                "The game makes one thing look easier than it is: you only move across a floor, and our model has 11,840 numbers to move at once, "
                "so it needs some maths to tell which way is warmer for each of them ([01_03](01_03_how_a_program_learns.ipynb)).\n\n"
                "So, a language model is a program that reads some text and **guesses what comes next**. Give it `ἄνδρα μο` and it guesses the next character, `ι`, "
                'because `ἄνδρα μοι` ("tell me of the man") is how the Odyssey begins.\n\n'
                "Ours guesses one character at a time. The big ones guess a **token**: a small piece of text, often part of a word. "
                "(Chapter 2 is all about tokens.)",
            ),
            q(
                "How can guessing the next character write a whole poem?",
                "By doing it over and over. Guess one character, stick it on the end, then guess again with the longer text:\n\n"
                "```\nἄνδρα μο → ι → ἄνδρα μοι → (space) → ἄνδρα μοι  → ἔ → …\n```\n\n"
                "Every chatbot works like this underneath. It never plans a whole sentence. It picks the next piece, then the next, and so on.",
            ),
            q(
                "What isn't a language model?",
                "It isn't a database of facts, and it doesn't look anything up. It doesn't understand Greek the way you do, either. "
                "What it has is **patterns**: which characters tend to follow which, learned from the text it trained on. "
                "So ours knows Book 1 of the Odyssey and nothing else.",
            ),
            q(
                'What makes a model "large"?',
                "The number of **weights**, the adjustable numbers inside it. More weights means room for more patterns.\n\n"
                "| Model | Weights |\n|---|---|\n| ours, `HomerGreekLLM` | 11,840 |\n| GPT-2 small (2019) | 124 million |\n| today's big chat models | hundreds of billions |\n\n"
                'Ours is tiny on purpose. It\'s the same idea at toy size, so it trains on a laptop in minutes. "Large" really just means "lots of weights".',
            ),
            q(
                "Where does this chapter fit, and what comes next?",
                "This chapter gives you the four basic ideas that every later chapter leans on:\n\n"
                "| Notebook | Question it answers |\n|---|---|\n"
                "| [01_01](01_01_a_language_model_without_a_brain.ipynb) | Can a language model be just counting? |\n"
                "| [01_02](01_02_neural_networks.ipynb) | What is a neural network? |\n"
                "| [01_03](01_03_how_a_program_learns.ipynb) | How does a program learn? |\n"
                "| [01_04](01_04_our_model_as_a_black_box.ipynb) | What does our model do, seen from outside? |\n\n"
                "New to tensors (PyTorch's tables of numbers)? See [A_02](../A_toolbox/A_02_pytorch_tensors.ipynb).",
            ),
            *questions(
                [
                    (
                        "Is mine really a 'large' language model, then?",
                        "No, it's tiny. But that's fine: every idea in this course is the one the big models use too. Only the size differs.",
                    ),
                ]
            ),
            recap(
                [
                    "A language model guesses what comes next; repeating that writes text.",
                    "It learns patterns from its training text, stored in weights.",
                    '"Large" means lots of weights: ours has 11,840, GPT-2 small 124 million.',
                ],
                "[01_01: A language model without a brain](01_01_a_language_model_without_a_brain.ipynb).",
            ),
        ],
    )

    # =====================================================================
    # 01_foundations/01_01_a_language_model_without_a_brain.ipynb
    # =====================================================================
    save(
        "01_foundations/01_01_a_language_model_without_a_brain.ipynb",
        [
            title("01_01 A language model without a brain", "chapter 1"),
            where(None, ["data/homer/odyssey/greek/book_01.txt"], here=CH1_HERE),
            learn(
                [
                    "How to build the simplest language model: by counting",
                    "How it writes Greek, one character at a time",
                    "Why it gets stuck or writes nonsense",
                    "What a better model would need",
                ]
            ),
            words(
                [
                    (
                        "bigram",
                        "a pair of neighbours: one character and the one right after it",
                        "`ν` then a space",
                    ),
                    (
                        "probability",
                        "how likely something is, from 0 (never) to 1 (always); often called the chance",
                        "0.5 = half the time",
                    ),
                    (
                        "greedy",
                        "always picking the single most likely next character",
                        "after `ἄ` always `λ`",
                    ),
                    (
                        "sampling",
                        "picking the next character at random, but with likely ones picked more often",
                        "like drawing from a bag",
                    ),
                ]
            ),
            md("## 3. The idea"),
            q(
                "What is the simplest way to guess the next character?",
                "Look at what usually came next in the text itself. Say that in Book 1 the character `ν` is followed by a space 529 times. "
                "Then after a `ν`, a space is a pretty good guess.\n\n"
                "Counting pairs like this (one character and the one right after it) is called a **bigram** count. *Bi* means two.",
            ),
            q(
                "Is that really a language model?",
                "Yes, it is. It reads text and guesses what comes next. Nothing in it learns anything, though. It's just a table of counts, "
                "which is why we call it a language model **without a brain**. It's the baseline the real model has to beat.",
            ),
            step_intro(),
            q(
                "Which pairs are in a tiny piece of text?",
                "Let's start with the first two words, `ἄνδρα μοι`. Here's each character with the one right after it:\n\n"
                "```\n'ἄ'→'ν'  'ν'→'δ'  'δ'→'ρ'  'ρ'→'α'  'α'→' '  ' '→'μ'  'μ'→'ο'  'ο'→'ι'\n```\n\n"
                'Each pair is one small fact, like "after `ν` came `δ`, once". And yes, the space counts as a character too.\n\n'
                "👉 **Predict first:** 9 characters. How many pairs?",
            ),
            code(
                """from itertools import pairwise

for current, nxt in pairwise("ἄνδρα μοι"):
    print(f"{current!r}→{nxt!r}", end="  ")"""
            ),
            saw(
                "So, 8 pairs from 9 characters, because the last character has nothing after it. "
                "`pairwise` hands them over one at a time. Next we do the same for all of Book 1, and count."
            ),
            q(
                "How do I count which character follows which?",
                "Walk along the text in pairs, each character with the one after it. "
                "For every pair we add 1 to a counter, and what we end up with is a table of counts.\n\n"
                "👉 **Predict first:** after `ν`, what do you think comes most often?",
            ),
            code(
                """from collections import Counter, defaultdict
from itertools import pairwise

from homer_ai.train_homer import BOOKS_DIR, load_books

text = load_books(BOOKS_DIR)[0]  # Book 1 of the Odyssey

follows = defaultdict(Counter)  # for each character: a counter of what came next
for current, nxt in pairwise(text):
    follows[current][nxt] += 1

print(follows["ν"].most_common(5))"""
            ),
            saw(
                "There it is: `(' ', 529)` comes first. That's `ν` followed by a space, because so many Greek words end in **-ν**."
            ),
            q(
                "Can I see those counts?",
                "Sure. A picture shows how lopsided they are. These are the 8 most common characters after `ν`.\n\n"
                "👉 **Predict first:** will the space bar be a little taller than the rest, or much taller?",
            ),
            code(
                """import matplotlib.pyplot as plt

from homer_ai.studio_style import ACCENT, INK, MUTED, studio_axes, studio_figure

top = follows["ν"].most_common(8)
total = sum(follows["ν"].values())

fig = studio_figure(
    "What follows one character, in Book 1",
    f"After ν, a space comes {top[0][1]} times out of {total:,}",
)
ax = studio_axes(fig)
ax.set_axisbelow(True)  # grid lines behind the bars
bars = ax.bar([repr(c) for c, _ in top], [n for _, n in top], color=MUTED)
bars[0].set_color(ACCENT)
for bar, (_, n) in zip(bars, top, strict=True):
    ax.text(
        bar.get_x() + bar.get_width() / 2, n + 6, str(n),
        ha="center", color=INK, family="monospace", fontsize=11,
    )
ax.tick_params(axis="x", labelsize=14, labelcolor=INK)
ax.grid(axis="x", visible=False)
ax.set_ylabel("times it came next")
plt.show()"""
            ),
            saw(
                "The space bar is far taller than the others, so a space is the best guess. "
                "But look at the rest. Plenty of other characters follow `ν` too, and the next steps show what that does to us."
            ),
            q(
                "What if it always picks the most likely next character?",
                "That's called **greedy**. We start with `ἄ` (the Odyssey's first character) and do it 40 times.\n\n"
                "👉 **Predict first:** after `ἄ`, the most common next character is `λ`. What happens next?",
            ),
            code(
                """line = "ἄ"
for _ in range(40):
    last = line[-1]
    best = follows[last].most_common(1)  # e.g. [('λ', 20)]: one (character, count) pair
    line += best[0][0]  # [0] = that pair, then [0] again = its character

print(line)"""
            ),
            saw(
                "`ἄλλλλλλλ…`. After `λ`, the most common next character is `λ` again (think of `πολλὰ`), "
                "so greedy picks it forever. It's stuck in a loop. Our real model loops too, by the way: you'll see `ἴδεν ἴδεν ἴδεν` and `και και και` later on."
            ),
            *exercise(
                "which character most often follows `κ`?",
                "The greedy loop asked `follows` for the most common next character, over and over. Now you ask it once, for `κ`. "
                "We want the character alone, not the pair with its count.\n\n"
                "👉 **Predict first:** think of a short Greek word that starts with `κ` and turns up everywhere.",
                starter="""
            # 👉 do: the one character that most often comes after "κ" in Book 1
            # 💡 hint: most_common(1) gives a list with one (character, count) pair
            after_kappa = None
            """,
                solution="""
            after_kappa = follows["κ"].most_common(1)[0][0]
            """,
                check="""
            check(
                "the character after κ",
                after_kappa,
                max(follows["κ"], key=follows["κ"].get),
                coach="most_common(1) gives a list like [('α', 140)]. Take [0] for the pair, "
                "then [0] again for its character.",
            )
            """,
                hint='Look at the two lines inside the greedy loop: `best = follows[last].most_common(1)` and then `best[0][0]`. Here `last` is `"κ"`.',
                answer="It's `α`, 140 times out of 432. That's mostly `καὶ` (and), one of the most common words in the book.",
            ),
            q(
                "How can it pick differently each time?",
                "With **sampling**: pick at random, but in proportion to the counts. If `λ` came after `ἄ` 20 times and `γ` 10 times, "
                "then `λ` gets picked twice as often as `γ`, but `γ` still gets its chance. `random.choices` does exactly this.\n\n"
                '`random.seed(42)` makes the "random" picks the same on every run, so your output matches this page.\n\n'
                "👉 **Predict first:** will it write real Greek words?",
            ),
            code(
                """import random

random.seed(42)
line = "ἄ"
for _ in range(60):
    options = follows[line[-1]]
    line += random.choices(list(options), weights=list(options.values()))[0]

print(line)"""
            ),
            saw(
                "`ἄγχαὶ τʼ ἀν καὶ ν γῆιδαίρτεροσθοἴ ἕσιχοσισαὶ ἔπε ἥρ ἐπὶ θεσε`. "
                "It does look Greek. Real little words turn up (`καὶ`, `ἐπὶ`) and the accents land in sensible places. "
                "Most of it is nonsense, though, because it only ever looks **one** character back."
            ),
            md(
                "## 5. Full example\n\n### ❓ How sure is it about each guess?\n\n"
                "Turn the counts after one character into **probabilities**: divide each count by the total, "
                "and they always add up to 1. (In the code, `{character!r}` prints the character in quotes, so a space shows up as `' '`.)\n\n"
                "👉 **Predict first:** after a space, how likely is the most common next character?"
            ),
            code(
                """after_space = follows[" "]
total = sum(after_space.values())

for character, count in after_space.most_common(5):
    chance = count / total
    print(f"{character!r}: {count} times, probability {chance:.3f}")"""
            ),
            saw(
                "So after a space, `δ` is the most common next character, but its probability is only about 0.09. That's fewer than 1 time in 10. "
                "Lots of characters can start a word, so one character of context just isn't enough to be sure."
            ),
            *questions(
                [
                    (
                        "How could it do better?",
                        "By looking further back. After `ἄνδρα μο` the next character is almost certainly `ι`. After just `ο` it could be lots of things. "
                        "The trouble is that counting every possible 8-character history would need an enormous table, and most histories never show up in Book 1 anyway. "
                        "So we build a model that **learns** patterns and can look many characters back. That's the rest of the course.",
                    ),
                ]
            ),
            recap(
                [
                    "Counting which character follows which (a bigram count) is already a language model.",
                    "Greedy picking gets stuck in loops; sampling looks Greek but is mostly nonsense.",
                    "It only looks one character back: a real model must look further, and learn.",
                ],
                "[01_02: Neural networks](01_02_neural_networks.ipynb).",
                can=[
                    "count which character follows which, with `pairwise` and a `Counter`",
                    "write text from the counts, greedy or by sampling",
                    "turn counts into probabilities that add up to 1",
                ],
            ),
        ],
    )

    # =====================================================================
    # 01_foundations/01_02_neural_networks.ipynb
    # =====================================================================
    save(
        "01_foundations/01_02_neural_networks.ipynb",
        [
            title("01_02 Neural networks", "appendix A"),
            where(None, ["src/homer_ai/model.py"], here=CH1_HERE),
            learn(
                [
                    "What a neural network is: layers of weights, numbers in → numbers out",
                    "How one layer turns 3 numbers into 2",
                    "How to count a network's weights",
                    "How our model's layers stack like floors",
                ]
            ),
            words(
                [
                    (
                        "neural network",
                        "layers of weights that turn numbers in into numbers out",
                        "our `HomerGreekLLM`",
                    ),
                    (
                        "layer",
                        "one step of the calculation: numbers go in, are mixed using weights, and new numbers come out",
                        "one floor of a building",
                    ),
                    (
                        "linear layer",
                        "the simplest layer: each output is the inputs multiplied by weights and added up, plus one extra number",
                        "`nn.Linear(3, 2)`",
                    ),
                    (
                        "bias",
                        "the extra number a linear layer adds to each output",
                        "one per output",
                    ),
                ]
            ),
            md("## 3. The idea"),
            q(
                "What is a neural network?",
                "A calculation made of **layers**. Numbers go in at the bottom, each layer mixes them using its weights (its adjustable numbers), "
                "and numbers come out at the top. That's all it is, really.\n\n"
                "The name comes from a loose comparison with brain cells (neurons), but you don't need any biology to use one.",
            ),
            q(
                "Is there a picture?",
                "Yes. Think of an office building. The numbers go up the stairs floor by floor, and on every floor the workers adjust knobs (the weights). "
                "There's a drawing of it, [the learning building](../../docs/neural-network-building.html), in this repo's `docs/` folder.",
            ),
            q(
                "Where do the weights come from?",
                "They start out **random**, so a brand-new network gives random answers. "
                "Training (notebook 01_03) is what adjusts the weights until the answers are good.",
            ),
            step_intro(),
            q(
                "How do I make one layer?",
                "PyTorch has one ready-made, the **linear layer**. `nn.Linear(3, 2)` takes 3 numbers in and gives 2 numbers out. "
                "Each output is the 3 inputs multiplied by weights and added up, plus one extra number, the bias. "
                "`torch` is PyTorch, and `torch.tensor` holds the numbers (see [A_02](../A_toolbox/A_02_pytorch_tensors.ipynb)).\n\n"
                "```\n[1, 2, 3]   →   layer: 6 weights + 2 biases   →   [?, ?]\n 3 numbers in                                  2 numbers out\n```\n\n"
                "👉 **Predict first:** we feed in `[1, 2, 3]`. Will the 2 outputs mean anything?",
            ),
            code(
                """import torch
from torch import nn

torch.manual_seed(42)  # the same "random" weights every run
layer = nn.Linear(3, 2)

x = torch.tensor([1.0, 2.0, 3.0])
print(layer(x))"""
            ),
            saw(
                "Two numbers, `0.7130` and `0.9659`, and they mean nothing yet, because the weights are random. "
                "(You can ignore `grad_fn=…`. That's PyTorch noting how the numbers were made, which it needs for learning later.)"
            ),
            q(
                "How many weights does that layer have?",
                "Each of the 2 outputs has one weight per input (so 3), plus its bias. That's 2 × 3 + 2 = 8. "
                "`.parameters()` lists all the adjustable numbers, and `.numel()` counts the numbers in each.\n\n"
                "👉 **Predict first:** 8?",
            ),
            code(
                """print(layer.weight.shape, layer.bias.shape)
print(sum(p.numel() for p in layer.parameters()))"""
            ),
            saw(
                "A 2 × 3 table of weights, 2 biases, and 8 adjustable numbers in total. "
                '(PyTorch calls every adjustable number a **parameter**. In this course we just say "the weights" for weights and biases together.)'
            ),
            md(
                "**Look at:** the 3 dots on the left are the numbers going in, and the 2 on the right are the numbers coming out. Each of the 6 lines is one weight, and each green dot is a bias."
            ),
            code(DIAGRAM_DOTS),
            q(
                "What happens if I stack two layers?",
                "The output of the first floor becomes the input of the second. `nn.Sequential` stacks layers in order.\n\n"
                "👉 **Predict first:** a 3 → 4 layer, then a 4 → 2 layer. How many weights in total?",
            ),
            code(
                """torch.manual_seed(42)
net = nn.Sequential(nn.Linear(3, 4), nn.Linear(4, 2))

print(net(x))
print(sum(p.numel() for p in net.parameters()))"""
            ),
            saw(
                "Still 2 numbers out, but now **26** weights: (4 × 3 + 4) + (2 × 4 + 2) = 16 + 10."
            ),
            md(
                "## 5. Full example\n\n### ❓ What are the floors of our own model?\n\n"
                "`HomerGreekLLM` in `src/homer_ai/model.py` is a neural network too. Here are its floors, bottom to top:\n\n"
                "```\nhead              → a score for each of the 137 possible next characters\nquery, key, value → the attention floor: which earlier characters matter (chapter 3)\nembedding         → turns each character's ID into 32 numbers (chapter 2)\n```\n\n"
                "We open up each floor in the chapters that follow. Here, let's just count the weights.\n\n"
                "👉 **Predict first:** more or fewer than 10,000?"
            ),
            code(
                """from homer_ai.model import HomerGreekLLM

model = HomerGreekLLM(vocab_size=137, embed_dim=32)

for name, p in model.named_parameters():
    print(f"{name:20} {str(tuple(p.shape)):10} {p.numel():6}")
print("total:", sum(p.numel() for p in model.parameters()))"""
            ),
            saw(
                "So that's 4,384 in the embedding (137 × 32), 1,024 each in query, key and value (32 × 32), and another 4,384 in the head (137 × 32). "
                "**11,840** in total."
            ),
            *questions(
                [
                    (
                        "Why are the starting weights random, and not zero?",
                        "If every weight started out the same, every worker on a floor would do the same job, and training could never make them different. "
                        "A random start gives each one its own direction to grow in.",
                    ),
                ]
            ),
            recap(
                [
                    "A neural network is layers of weights: numbers in, numbers out.",
                    "A linear layer multiplies inputs by weights, adds them up, and adds a bias.",
                    "Our model stacks embedding → attention → head: 11,840 weights.",
                ],
                "[01_03: How a program learns](01_03_how_a_program_learns.ipynb).",
                can=[
                    "make a layer with `nn.Linear` and say how many weights it has",
                    "stack layers, and count a whole model's weights",
                ],
            ),
        ],
    )

    # =====================================================================
    # 01_foundations/01_03_how_a_program_learns.ipynb
    # =====================================================================
    save(
        "01_foundations/01_03_how_a_program_learns.ipynb",
        [
            title("01_03 How a program learns", "appendix A"),
            where(None, ["src/homer_ai/train_homer.py"], here=CH1_HERE),
            learn(
                [
                    "The learning loop: guess → measure the mistake → adjust → repeat",
                    "How to learn one weight by hand",
                    "How PyTorch finds the direction to adjust: gradients and backpropagation",
                    "Where this loop sits in our training code",
                ]
            ),
            words(
                [
                    (
                        "loss",
                        "a mistake score: one number for how wrong the guesses were; lower is better",
                        "`(guess - answer)²`",
                    ),
                    (
                        "gradient",
                        "for each weight: which way, and how steeply, the loss goes up if that weight grows; we step the other way",
                        "`w.grad` = `-27.0`",
                    ),
                    (
                        "backpropagation",
                        "the method PyTorch uses to work out every weight's gradient, going backwards from the loss",
                        "`loss.backward()`",
                    ),
                    (
                        "optimizer",
                        "the PyTorch tool that adjusts the weights using their gradients",
                        "`torch.optim.SGD`",
                    ),
                ]
            ),
            md("## 3. The idea"),
            q(
                "What does it mean for a program to learn?",
                "Its weights change so that its guesses get better. It happens in a loop:\n\n"
                "```\nguess → measure the mistake → adjust the weights a little → repeat\n```\n\n"
                "Nobody types in the right weights by hand. The program finds them itself, one small adjustment at a time.",
            ),
            q(
                "How do we measure the mistake?",
                "With a **loss**: one number that's big when the guesses are bad and small when they're good. "
                "A simple one is the mistake multiplied by itself, `(guess - answer)²`. Squaring makes it positive either way, so too high and too low both count as wrong.",
            ),
            q(
                "How does it know which way to adjust?",
                "Picture yourself on a hillside in fog, where your height is the loss. You can't see the bottom, but you can feel which way the ground slopes. "
                "So you take a small step downhill, and then another. That slope, for each weight, is its **gradient**.",
            ),
            step_intro(),
            q(
                "Can I learn one weight by hand?",
                "Let's try. The task is to find `w` so that `w × 3` gives `6`. (It's 2, obviously, but the program doesn't know that.) "
                "We start at `w = 0.5`. Each round we guess, see how far off we are, and move `w` a little in the right direction:\n\n"
                "```\n"
                "w       guess = w × 3    mistake = guess − 6    nudge = −0.1 × mistake\n"
                "0.500   1.500            −4.500                 +0.450\n"
                "0.950   2.850            −3.150                 +0.315\n"
                "1.265   …\n"
                "```\n\n"
                "The guess is too small, so the mistake is negative, so the nudge pushes `w` up.\n\n"
                "👉 **Predict first:** will `w` get to 2 in one step, or creep towards it?",
            ),
            code(
                """w = 0.5
for step in range(8):
    guess = w * 3
    mistake = guess - 6  # negative = guess too small
    print(f"step {step}: w = {w:.3f}, guess = {guess:.3f}, mistake = {mistake:.3f}")
    w = w - 0.1 * mistake  # nudge w against the mistake"""
            ),
            saw(
                "`w` creeps up from 0.5 (0.950, 1.265, 1.486, …) and the mistake shrinks every time. "
                "That `0.1` is the size of the nudge. Too big and you overshoot, too small and it takes forever. "
                "Its proper name is the **learning rate**, and [05_02](../05_training_on_the_odyssey/05_02_learning_rate.ipynb) is all about it."
            ),
            *exercise(
                "can you do one nudge yourself?",
                "Same task (`w × 3` should give `6`), but this time we start at `w = 1.0`. The guess and the mistake are worked out for you. You write the nudge.\n\n"
                "👉 **Predict first:** the guess is 3, which is too small. Should the new `w` be above or below 1.0?",
                starter="""
            start_w = 1.0
            guess = start_w * 3
            mistake = guess - 6  # negative = guess too small

            # 👉 do: nudge start_w against the mistake, with a nudge size of 0.1
            new_w = None
            """,
                solution="""
            start_w = 1.0
            guess = start_w * 3
            mistake = guess - 6  # negative = guess too small

            new_w = start_w - 0.1 * mistake
            """,
                check="""
            check(
                "w after one nudge",
                new_w,
                1.3,
                coach="Got 0.7? Then you added the mistake. The mistake is negative here "
                "(the guess is too small), so we subtract it, and that pushes w up.",
            )
            """,
                hint="It's the last line of the loop above, `w = w - 0.1 * mistake`, with `start_w` in place of `w`.",
                answer="The mistake is `3 − 6 = −3`, so the nudge is `−0.1 × −3 = +0.3` and `w` goes from 1.0 to 1.3. Closer to 2, but not there yet.",
            ),
            q(
                "Can I see it learn?",
                "Sure. We run the same loop again, but this time we write down `w` and the loss at every step, then draw them. "
                "The dashed line is the answer, `2`.\n\n"
                "👉 **Predict first:** will `w` bend smoothly onto the dashed line, or zigzag across it?",
            ),
            code(
                """import matplotlib.pyplot as plt

from homer_ai.studio_style import ACCENT, GOOD, INK, MUTED, studio_axes, studio_figure

w = 0.5
ws, losses = [], []
for _step in range(15):
    ws.append(w)
    losses.append((w * 3 - 6) ** 2)  # the loss: (guess - answer) squared
    w = w - 0.1 * (w * 3 - 6)

fig = studio_figure("One weight, learning", f"w creeps from {ws[0]} to {ws[-1]:.2f}")
left = studio_axes(fig, (0.09, 0.12, 0.38, 0.68))
left.plot(ws, color=ACCENT, linewidth=3, marker="o", markersize=5)
left.axhline(2, color=GOOD, linestyle="--", linewidth=1.5)
left.text(14, 2.06, "target 2", color=GOOD, ha="right", family="monospace")
left.set_xlabel("step")
left.set_ylabel("w")

right = studio_axes(fig, (0.57, 0.12, 0.38, 0.68))
right.plot(losses, color=INK, linewidth=3, marker="o", markersize=5)
right.set_xlabel("step")
right.set_ylabel("loss")
right.text(0.5, losses[0], f"{losses[0]:.1f}", color=MUTED, family="monospace")
plt.show()"""
            ),
            saw(
                "`w` bends smoothly onto the dashed line, fast at first, then slower as the mistake shrinks. "
                "The loss falls the same way: steep at the start, flat at the end. That flat end is what a program that has learned looks like."
            ),
            q(
                "How does PyTorch do the same thing?",
                "With three calls that you'll see in every training loop:\n\n"
                "- `loss.backward()` works out the gradient of every weight. The method is called **backpropagation**, because it goes backwards from the loss through every calculation.\n"
                '- `optimizer.step()` nudges every weight downhill, using its gradient. **SGD** (stochastic gradient descent, or "step down the slope") is the simplest optimizer: weight = weight − nudge size × gradient.\n'
                "- `optimizer.zero_grad()` wipes the last round's gradients first, like clearing a calculator. Otherwise they'd pile up.\n\n"
                "`requires_grad=True` tells PyTorch to keep track of `w` so it can find its gradient.\n\n"
                "👉 **Predict first:** at `w = 0.5`, the guess is too small. Will the gradient be positive or negative?",
            ),
            code(
                """import torch

w = torch.tensor(0.5, requires_grad=True)
optimizer = torch.optim.SGD([w], lr=0.02)

for step in range(8):
    loss = (w * 3 - 6) ** 2
    optimizer.zero_grad()
    loss.backward()
    grad = w.grad.item()
    w_now, loss_now = w.item(), loss.item()
    print(f"step {step}: w = {w_now:.4f}, loss = {loss_now:.4f}, gradient = {grad:.3f}")
    optimizer.step()

print(f"final w = {w.item():.4f}")"""
            ),
            saw(
                'The gradient starts at **−27.0**. Negative means "growing `w` makes the loss go down", '
                "so the step makes `w` bigger. The loss falls 20.25 → 8.29 → 3.40 → … → 0.04, and `w` ends up at about 1.958, close to 2.\n\n"
                "Where does −27 come from? The by-hand loop moved `w` by `0.1 × mistake`. The gradient of this loss is bigger than the mistake: it's `2 × 3 × mistake` (the 2 comes from the square, the 3 from `w × 3`). "
                "At `w = 0.5` the mistake is −4.5, so the gradient is `6 × −4.5 = −27`. SGD moves `w` by `lr × gradient`, which is `lr × 6 × mistake`. "
                "So `lr=0.02` here is a nudge of `0.12 × mistake`, about the same size as the `0.1 × mistake` we used by hand. The two 0.1s you'll see are not the same thing."
            ),
            md(
                "## 5. Full example\n\n### ❓ Where is this loop in our real training code?\n\n"
                "`train()` in `src/homer_ai/train_homer.py` does exactly the same four things, only with 11,840 weights instead of one:\n\n"
                "```python\nlogits = model(inputs)                    # 1. guess\nloss = F.cross_entropy(logits, targets)   # 2. measure the mistake\noptimizer.zero_grad()                     #    clear the last gradients\nloss.backward()                           # 3. find every weight's gradient\noptimizer.step()                          # 4. nudge every weight downhill\n```\n\n"
                "`logits` is just the usual name for the model's raw scores. `cross_entropy` is a loss made for guessing between many options (chapter 5). And our optimizer is `AdamW`, a smarter one that adapts the step size for each weight.\n\n"
                "One last experiment with our single weight. We raise the learning rate from 0.02 to `lr=0.1`. From the sum above, that's a nudge of `0.1 × 6 × mistake = 0.6 × mistake`, six times the by-hand one.\n\n"
                "👉 **Predict first:** with this bigger nudge on the same task, do you get closer to 2 in 8 steps, or overshoot?"
            ),
            code(
                """w = torch.tensor(0.5, requires_grad=True)
optimizer = torch.optim.SGD([w], lr=0.1)

for step in range(8):
    loss = (w * 3 - 6) ** 2
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(f"step {step}: w = {w.item():.2f}")"""
            ),
            saw(
                "`w` jumps right past 2, to 3.20, then swings back and forth around it (1.04, 2.77, 1.39, …). After 8 steps it's at 1.75, which is further from 2 than the smaller nudge got (1.958). Every step overshoots. "
                "(By hand, `0.1` crept up smoothly. Here `lr=0.1` overshoots, because SGD multiplies it by the gradient, which is 6 times the mistake.) "
                "So picking the nudge size matters, and chapter 5 shows how to do it."
            ),
            recap(
                [
                    "Learning is a loop: guess → loss → adjust → repeat.",
                    "The gradient says which way is downhill; backpropagation (`loss.backward()`) finds it for every weight.",
                    "`optimizer.step()` takes the step; the nudge size is the learning rate.",
                ],
                "[01_04: Our model as a black box](01_04_our_model_as_a_black_box.ipynb).",
                can=[
                    "nudge one weight against its mistake, by hand",
                    "write the PyTorch training loop: `zero_grad`, `backward`, `step`",
                    "tell when a learning rate is too big: the weight jumps past the answer",
                ],
            ),
        ],
    )

    save(
        "01_foundations/01_04_our_model_as_a_black_box.ipynb",
        [
            title("01_04 Our model as a black box", None),
            where(
                None,
                [
                    "src/homer_ai/model.py",
                    "src/homer_ai/homer_model.pt",
                    "src/homer_ai/homer_mcp_server.py",
                ],
                here=CH1_HERE,
            ),
            learn(
                [
                    "What our model does, seen from outside: text in, scores for the next character out",
                    "What an untrained model writes, and what a trained one writes",
                    "Which parts of the box the next chapters open",
                ]
            ),
            words(
                [
                    (
                        "black box",
                        "something you use by its inputs and outputs, without looking inside",
                        "a TV remote",
                    ),
                    (
                        "score",
                        "the number the model gives each possible next character; higher means more likely",
                        "137 scores per position",
                    ),
                    (
                        "checkpoint",
                        "a saved copy of a trained model's weights, in a file",
                        "`homer_model.pt`",
                    ),
                ]
            ),
            md("## 3. The idea"),
            q(
                "Why use the model without understanding it first?",
                "Because then you know what all the parts are for. It's like driving a car before you open up the engine: "
                "first you see what it does, and the later chapters show how. We open the box in chapters 2-4.",
            ),
            q(
                "What goes in, and what comes out?",
                "**In:** a line of text, turned into numbers by the tokenizer (chapter 2).\n\n"
                "**Out:** for every position in the line, 137 scores, one for each character the model knows (its **vocabulary**). "
                "The highest score is the model's guess for the next character.",
            ),
            q(
                "Where does a trained model come from?",
                "Training (chapter 5) adjusts the weights and saves them in a **checkpoint** file, `src/homer_ai/homer_model.pt`. "
                "Your own file isn't stored in git, so a fresh copy of the project doesn't have one. What it does have is a ready-made one, "
                "[models/odyssey_book1.pt](../../models/README.md), and `load_homer_model` uses that and tells you so. "
                "To make your own, run `uv run homer-train` (about 8 to 10 minutes). After that, yours gets used instead. "
                "(The course has a second, bigger model too, the GPT of chapter 4. Its command is `uv run homer-train --model gpt`, about 3 minutes, and [05_07](../05_training_on_the_odyssey/05_07_train_the_gpt.ipynb) walks through it.)",
            ),
            step_intro(),
            q(
                "How do I load an untrained and a trained model?",
                "The untrained one is just a new `HomerGreekLLM`, with random weights. The trained one loads its weights from the checkpoint "
                "and builds its tokenizer from what's saved in the same file, exactly as `homer_mcp_server.py` does. That matters: the model learned with one numbering of the characters (`π` = 41, say), and a tokenizer built from another text would number them differently, so the trained weights would read nonsense. "
                "The helper `load_homer_model` (in `src/homer_ai/lessons.py`) does all this without writing anything. [05_06](../05_training_on_the_odyssey/05_06_saving_and_loading.ipynb) builds the steps by hand.\n\n"
                "👉 **Predict first:** the cell prints one line saying which model it loaded. On your computer, will it be your own or the ready-made one?",
            ),
            code(BLACK_BOX_SETUP),
            saw(
                "One line telling you which model was loaded: `Using the ready-made model from models/ (137 tokens)…` on a fresh copy of the project, or `Using your own model…` once you've trained one. Either way the tokenizer has 137 tokens, taken from the file."
            ),
            q(
                "What comes out of the box?",
                "Give the model `ἄνδρα μο` (8 characters) and it gives back a table: 8 rows (one per position) × 137 scores. "
                "`softmax` then turns the last row of scores into **probabilities** that add up to 1.\n\n"
                "```\n"
                "ἄ  ν  δ  ρ  α  ' '  μ  ο     →   model   →   8 rows × 137 scores\n"
                "                                             last row → chances for the character after `ἄνδρα μο`\n"
                "```\n\n"
                "👉 **Predict first:** for the trained model, which character after `ἄνδρα μο`?",
            ),
            code(
                """ids = torch.tensor(tokenizer.encode("ἄνδρα μο"), dtype=torch.long)

for name, model in [("untrained", untrained), ("trained", trained)]:
    if model is None:
        continue
    with torch.no_grad():
        scores = model(ids)
    chances = torch.softmax(scores[-1], dim=-1)
    top = torch.topk(chances, 5)
    best = [
        (tokenizer.decode([i.item()]), round(c.item(), 3))
        for c, i in zip(top.values, top.indices, strict=True)
    ]
    print(name, tuple(scores.shape), best)"""
            ),
            saw(
                "The table is `(8, 137)`. The untrained model gives every character about the same tiny chance "
                "(about 0.009 each, close to 1 in 137, so a pure guess). The trained one puts `ι` first, at about 0.21. It has learned that `ἄνδρα μοι` comes next."
            ),
            q(
                "Can I see the trained model's chances?",
                'Yes. Here\'s a bar chart of its 8 best guesses for the character after `ἄνδρ` (the start of `ἄνδρα`, "man"). '
                "Each bar is a probability, and all 137 of them add up to 1.\n\n"
                "👉 **Predict first:** will one bar tower over the rest, or will several be close?",
            ),
            code(
                """import matplotlib.pyplot as plt

from homer_ai.studio_style import ACCENT, INK, MUTED, studio_axes, studio_figure

if trained is None:
    print("no trained model yet: make one first, then run this cell again")
else:
    ids = torch.tensor(tokenizer.encode("ἄνδρ"), dtype=torch.long)
    with torch.no_grad():
        chances = torch.softmax(trained(ids)[-1], dim=-1)
    top = torch.topk(chances, 8)
    letters = [repr(tokenizer.decode([i.item()])) for i in top.indices]
    values = top.values.tolist()

    fig = studio_figure(
        "The trained model's chances",
        f"After ἄνδρ, its best guess is {letters[0]}: {values[0]:.0%}",
    )
    ax = studio_axes(fig)
    ax.set_axisbelow(True)  # grid lines behind the bars
    bars = ax.bar(letters, values, color=MUTED)
    bars[0].set_color(ACCENT)
    for bar, value in zip(bars, values, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2, value + 0.01, f"{value:.2f}",
            ha="center", color=INK, family="monospace", fontsize=11,
        )
    ax.tick_params(axis="x", labelsize=14, labelcolor=INK)
    ax.grid(axis="x", visible=False)
    ax.set_ylabel("probability")
    plt.show()"""
            ),
            saw(
                "No bar towers over the rest. The best guess is only about 0.22, and `ε` is close behind. "
                "So the model hands back a table of chances, not one fixed answer. It really is guessing, and Greek really can go on in several ways."
            ),
            md(
                "## 5. Full example\n\n### ❓ What does each model write?\n\n"
                "We start with `ἄνδρα` and do this 40 times: guess the most likely next character, add it, guess again.\n\n"
                "👉 **Predict first:** will the trained model write real Homer?"
            ),
            code(GENERATE),
            saw(
                "With the model trained in this project (yours may differ if you retrained), you get:\n\n"
                "```\nuntrained: ἄνδραἂ<|unk|><|unk|><|unk|>ὮὮὮὮὮὮὮὮὮὮὮὮὮὮὮὮὮὮῶὧῶὄχῶὄὧὧῶὄῶἣὄῶἣἣἣ\ntrained:   ἄνδρα τοι και και και κα κα κατον κατοι και κ\n```\n\n"
                "The untrained model writes random characters, even `<|unk|>` (the stand-in for a character it doesn't know). The trained one writes Greek-looking words with spaces, "
                "but then gets stuck repeating `και`, just like the counting model in 01_01. Fixing that is what chapters 2-6 are for."
            ),
            *questions(
                [
                    (
                        "Is this the same model that Claude talks to?",
                        "It can be, but it isn't the first choice. `homer_mcp_server.py` looks for the GPT of chapter 4 first (yours, then the ready-made one), and only uses this small model if there's no GPT file, or if you point `HOMER_MODEL` at it ([06_03](../06_making_it_talk/06_03_the_mcp_server.ipynb)). "
                        "Whichever it loads, it runs this same guessing loop whenever Claude calls the tool.",
                    ),
                ]
            ),
            recap(
                [
                    "From outside, the model is text in → 137 scores per position out; the highest score is its guess.",
                    "Untrained, it guesses at random; trained on Book 1, it writes Greek-looking words but repeats itself.",
                    "Chapters 2-4 open the box: tokens and embeddings, attention, the full model.",
                ],
                "Chapter 2: [Text to numbers](../02_text_to_numbers/02_00_chapter_overview.ipynb).",
                can=[
                    "load the model and give it a text",
                    "read what comes out: one row of 137 scores per position",
                    "tell an untrained model's writing from a trained one's",
                ],
            ),
        ],
    )
