# ruff: noqa: E501
"""Chapter 3, Attention."""

from .cells import (
    background,
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
from .diagrams import DIAGRAM_BLEND, DIAGRAM_QKV
from .setup_code import CH3_HEATMAP, CH3_LOAD


def build():
    # =====================================================================
    # 03_attention/03_00_chapter_overview.ipynb
    # =====================================================================
    save(
        "03_attention/03_00_chapter_overview.ipynb",
        [
            title("03_00 Attention (overview)", "chapter 3"),
            where("model reads", ["src/homer_ai/model.py"]),
            learn(
                [
                    "Why guessing the next letter needs more than the last letter",
                    "What attention is: deciding which earlier letters matter, and by how much",
                    "The three ideas of the chapter: scores and mixing, no peeking, many heads",
                    "Which part of our `model.py` is attention",
                    "Which notebook to open for each idea",
                ]
            ),
            words(
                [
                    (
                        "attention",
                        "the model's way of giving each earlier letter a share of its focus, then mixing them by those shares",
                        "while guessing after `μοῦσ`, `σ` gets the biggest share",
                    ),
                    (
                        "context",
                        "the earlier letters the model can look at when it makes one guess",
                        "for the guess after `μοῦσ`, the context is `μ ο ῦ σ`",
                    ),
                ]
            ),
            *background(),
            q(
                "Why isn't the last letter enough?",
                "Picture the letters of a line as people who walk into a room one by one and sit down round a table. "
                "Each of them has one job, to guess who walks in next, and knowing only itself it would be guessing blind. "
                "So it looks round the table and decides whom to listen to: a lot to the ones beside it, a little to someone further away, and nothing to anyone who hasn't arrived yet. "
                "What it takes away is one mix of their voices, and it guesses from that. "
                "The picture is too kind in one way: people choose whom to listen to by thinking, and a letter does it with a multiplication that only gets good through training.\n\n"
                'Now the same thing on real text. Read the first line of the *Odyssey*: `ἄνδρα μοι ἔννεπε, μοῦσα` ("tell me, Muse, of the man"). '
                "Say you've read `μοῦσ` and have to guess the next letter. If all you saw was the last letter, `σ`, you could be at the end of hundreds of different words. "
                "The letters just before it (`ο ῦ`) narrow it down a lot.\n\n"
                "```\nμ  ο  ῦ  σ  →  ?\n      └──┴──┘ the letters just before matter most\n```\n\n"
                "We can actually count this in the real Book 1. Below, we look at every place where the text has the letters `before`, and see which letter comes next.\n\n"
                "👉 **Predict first:** with more letters in front, will the most common next letter be more certain, or less?",
            ),
            code(
                """from collections import Counter

from homer_ai.train_homer import BOOKS_DIR, load_books

text = load_books(BOOKS_DIR)[0]

for before in ["σ", "οῦσ", "μοῦσ"]:
    nexts = Counter()
    for i in range(len(text) - len(before)):
        # does the text, from place i, begin with `before`?
        if text.startswith(before, i):
            nexts[text[i + len(before)]] += 1
    total = sum(nexts.values())
    best, n = nexts.most_common(1)[0]
    print(f"after {before!r:8} seen {total:>3} times;", end=" ")
    print(f"most common next {best!r}: {n} of {total} = {n / total:.0%}")"""
            ),
            saw(
                "After `σ` alone, the best guess is right only about a quarter of the time. After `οῦσ` it's about half. After `μοῦσ` it's always `α` "
                "(though that only happened once, so don't read much into that one; it's the direction that counts). "
                'By the way, `Counter` is a ready-made Python tool that counts things, and `text.startswith(before, i)` asks "from place `i`, does the text begin with `before`?". '
                "So more earlier letters make the guess more certain. The model has to **look back**."
            ),
            q(
                "Do all the earlier letters matter equally?",
                "No. The letters right before matter most, and a letter far back matters less. But a few far-back ones can still matter a lot "
                "(the first letter of a word can tell you the ending, for example). "
                "So the model also has to decide which earlier letters to listen to, and how much.\n\n"
                "That's **attention**. Think of 100 points of focus, shared out between the earlier letters. Here's an invented picture of the idea (the real numbers come in [03_01](03_01_simple_attention.ipynb)):\n\n"
                "```\nguessing after μοῦσ:   μ ▏        a little\n                       ο ▏▏▏▏\n                       ῦ ▏▏▏▏▏▏\n                       σ ▏▏▏▏▏▏▏▏▏▏  the most\n```\n\n"
                "Then the model mixes the earlier letters, each one counting as much as its share. What comes out is one list of numbers that sums up the **context**: "
                "what the model needs from everything it has read so far.",
            ),
            q(
                "Where is attention in our `model.py`?",
                "Right in the middle of `forward`, between the embedding (chapter 2) and the final `head` that makes the 137 scores. "
                "Let's print the real lines of `forward`, so you can see where it sits.\n\n"
                "👉 **Predict first:** which lines do you think are attention?",
            ),
            code(
                """import inspect

from homer_ai.model import HomerGreekLLM

print(inspect.getsource(HomerGreekLLM.forward))"""
            ),
            saw(
                "`x = self.embedding(idx)` is the numbers from chapter 2. Everything from `Q, K, V = …` down to `context = weights @ V` is **attention**. "
                "Then `self.head(context)` turns the mixed result into 137 scores, one per token. "
                "(Chapter 4 builds the rest of the model around this.)"
            ),
            q(
                "What are the three ideas in this chapter?",
                "1. **Score and mix.** How much does each earlier letter matter (a score), and how do we mix them by those scores?\n"
                "2. **No peeking.** While it guesses a letter, the model mustn't see the letters that come after it, or it would just copy the answer.\n"
                '3. **Many heads.** One way of looking may not be enough, so we let several small "lookers" each search for something different.',
            ),
            q(
                "Which notebook do I open for each idea?",
                "| Notebook | Question it answers |\n|---|---|\n"
                "| [`03_01_simple_attention.ipynb`](03_01_simple_attention.ipynb) | How do we score the earlier letters and mix them, with nothing to learn yet? |\n"
                "| [`03_02_trainable_attention.ipynb`](03_02_trainable_attention.ipynb) | How does the model learn what to look for? (query, key, value) |\n"
                "| [`03_03_hiding_the_future.ipynb`](03_03_hiding_the_future.ipynb) | How do we stop the model peeking at later letters? |\n"
                "| [`03_04_many_heads.ipynb`](03_04_many_heads.ipynb) | What if several heads look at once? |",
            ),
            q(
                "Is everything in this chapter in our `model.py`?",
                "Most of it. Our model has **one** attention head and the mask that stops peeking (ideas 1 and 2). It has no second head (idea 3) and no positions ([02_06](../02_text_to_numbers/02_06_positions.ipynb)). "
                "The GPT of chapter 4 (`src/homer_ai/gpt.py`) has both: several heads ([04_04](../04_building_the_model/04_04_one_block.ipynb)) and positions ([04_05](../04_building_the_model/04_05_the_full_model.ipynb)).",
            ),
            recap(
                [
                    "To guess the next letter well, the model must look back: more earlier letters make the guess more certain.",
                    "Attention gives each earlier letter a share of focus (the shares add up to 1), then mixes the letters by those shares.",
                    "Idea 1 is scores and mixing, idea 2 is not peeking ahead, idea 3 is several heads.",
                ],
                "[03_01: Simple attention](03_01_simple_attention.ipynb): scoring and mixing with nothing to learn.",
            ),
        ],
    )

    # =====================================================================
    # 03_attention/03_01_simple_attention.ipynb
    # =====================================================================
    save(
        "03_attention/03_01_simple_attention.ipynb",
        [
            title("03_01 Simple attention", "section 3.3"),
            where("model reads", ["src/homer_ai/model.py"]),
            learn(
                [
                    "How to score how well two letters match, using the dot product",
                    "How softmax turns scores into attention weights that add up to 1",
                    "How the weights mix the letters into one context vector",
                    "Why attention with nothing to learn always prefers each letter itself",
                ]
            ),
            words(
                [
                    (
                        "attention score",
                        "a number for how well two letters match: the dot product of their rows",
                        "`δ` against `δ` scores 2",
                    ),
                    (
                        "attention weights",
                        "the scores after softmax: one share per earlier letter, and the shares add up to 1",
                        "`[0.21, 0.21, 0.58]`",
                    ),
                    (
                        "context vector",
                        "the mix of all the letters' rows, each counted by its attention weight",
                        "`[0.79, 0.79]` in our tiny example",
                    ),
                    (
                        "softmax",
                        "turns any list of scores into shares between 0 and 1 that add up to 1 (see [A_04](../A_toolbox/A_04_pytorch_in_this_repo.ipynb))",
                        "`[1, 1, 2]` → `[0.21, 0.21, 0.58]`",
                    ),
                ]
            ),
            *background(),
            q(
                "What does attention do, in one picture?",
                "Take the start of `ἄνδρα`, the three letters `ἄ ν δ`. The model has read them and is about to guess the next one. "
                "It looks back at each letter and decides how much to listen to it. Each letter gets a **share**, and the shares add up to 1 (all of its focus). "
                "Then it blends the three letters, each counted by its share.\n\n"
                "```\nlook at:     ἄ        ν        δ\nshare:       ?        ?        ?         (the three add up to 1)\nblend:   share×ἄ  +  share×ν  +  share×δ   =  the context vector\n```\n\n"
                "So three steps: score each letter, turn the scores into shares, blend. "
                "Here the letters have nothing to learn. We use their embedding rows from [02_05](../02_text_to_numbers/02_05_token_embeddings.ipynb) just as they are.",
            ),
            step_intro(),
            q(
                "Step 1: how do we score the letters?",
                "The score is a **dot product**: multiply two rows number by number and add up the results ([A_03](../A_toolbox/A_03_dot_product.ipynb) explains it with films; if you skipped it, 10 minutes there will make this whole chapter easier). "
                "A high score means the two rows point the same way.\n\n"
                "To keep the sums small, we give each letter an invented row of just 2 numbers. The letter doing the looking is `δ`:\n\n"
                "| Letter | Row | The maths (`δ` = `[1, 1]`) | Score |\n|---|---|---|---|\n"
                "| `ἄ` | `[1, 0]` | (1 × 1) + (1 × 0) | **1** |\n"
                "| `ν` | `[0, 1]` | (1 × 0) + (1 × 1) | **1** |\n"
                "| `δ` | `[1, 1]` | (1 × 1) + (1 × 1) | **2** |\n\n"
                "The code scores all three at once with `X @ X[2]`. The `@` is **matrix multiplication**, and for a table and one row it means: take the dot product of every row of the table with that row. "
                "Here `X` is the table of three rows, `X[2]` is the row of `δ`, and the result is three scores, one per row of `X`.\n\n"
                "👉 **Predict first:** will it print `[1, 1, 2]`?",
            ),
            code(
                """import torch

letters = ["ἄ", "ν", "δ"]
X = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])  # one invented row per letter

scores = X @ X[2]  # every row dotted with the row of δ (the 3rd letter)
print(scores)"""
            ),
            saw(
                "`tensor([1., 1., 2.])`, the same as the table. `X[2]` is the 3rd row (counting starts at 0), and `X @ X[2]` dots every row of `X` with it."
            ),
            q(
                "Step 2: how do the scores become shares?",
                "Scores like `[1, 1, 2]` aren't shares yet, because they don't add up to 1. **Softmax** fixes that ([A_04](../A_toolbox/A_04_pytorch_in_this_repo.ipynb)). "
                "It makes every score positive, keeps the order (bigger score, bigger share), and divides so the total is 1. "
                "What comes out are the attention weights.\n\n"
                "👉 **Predict first:** `δ` has the biggest score, 2. Will its share be about half, or more? And what will the three shares add up to?",
            ),
            code(
                """weights = torch.softmax(scores, dim=0)
for letter, w in zip(letters, weights, strict=True):
    print(f"{letter}  share {w:.3f}")
print("they add up to", weights.sum().item())"""
            ),
            saw(
                "`0.212`, `0.212` and `0.576`. So `δ` gets a bit more than half, and the three add up to 1 (`1.0`, apart from tiny rounding in the last digit). "
                "Notice that a score of 2 against 1 gave a share 2.7 times bigger, not 2 times. Softmax boosts the biggest."
            ),
            q(
                "Step 3: how do we blend the letters?",
                "Multiply each letter's row by its share, then add them up. The blended row is the **context vector**:\n\n"
                "```\n0.212 × [1, 0]  +  0.212 × [0, 1]  +  0.576 × [1, 1]\n=  [0.212, 0]  +  [0, 0.212]  +  [0.576, 0.576]\n=  [0.788, 0.788]\n```\n\n"
                "In PyTorch this whole blend is one `@`: the row of 3 shares dotted with each of the 2 columns of `X`.\n\n"
                "👉 **Predict first:** will `weights @ X` print `[0.788, 0.788]`?",
            ),
            code(
                """context = weights @ X
print(context)"""
            ),
            saw(
                "`tensor([0.7881, 0.7881])`. It leans towards `δ`'s row `[1, 1]`, because `δ` got the biggest share."
            ),
            md(
                "**Look at:** the width of each arrow. A wider arrow is a bigger share, and the three arrows meet in the context vector."
            ),
            code(DIAGRAM_BLEND),
            *exercise(
                "what is the context vector of `ν`?",
                "We did the three steps for `δ`, the third letter. Do them for `ν`, the second one: score, softmax, blend.\n\n"
                "👉 **Predict first:** `ν` is the row `[0, 1]`. Which of the three letters will it give the smallest share?",
                starter="""
            # 👉 do: the context vector of ν (row 1 of X): scores, then softmax, then the blend
            # 💡 hint: for δ the lines were X @ X[2], torch.softmax(scores, dim=0), weights @ X
            nu_context = None
            """,
                solution="""
            nu_scores = X @ X[1]
            nu_weights = torch.softmax(nu_scores, dim=0)
            nu_context = nu_weights @ X
            """,
                check="""
            check(
                "the context vector of ν",
                nu_context,
                torch.softmax(X @ X[1], dim=0) @ X,
                coach="Three things to look at. ν is X[1], not X[2] (we count from 0). "
                "Softmax goes between the scores and the blend. And the blend is weights @ X, "
                "which gives 2 numbers, like one row of X.",
            )
            """,
                hint="Copy the three lines of steps 1 to 3 and change one thing: the row we score against is `X[1]`.",
                answer="About `[0.58, 0.84]`. The scores are `[0, 1, 1]`: `ν` has nothing in common with `ἄ`, so `ἄ` gets the smallest share (0.16), and `ν` and `δ` get 0.42 each. "
                "The blend is mostly `ν` and `δ`, and both of those have a 1 in the second place, so that's the bigger number.",
            ),
            q(
                "Can every letter do this at once?",
                "Yes. Every letter takes a turn as the one doing the looking, and row `i` of the table is letter `i` looking at all the letters. "
                "That's just `X @ X.T` (every row against every row, as in [A_03](../A_toolbox/A_03_dot_product.ipynb)), softmax along each row, then `weights @ X`.\n\n"
                "One thing isn't right yet, though: `ἄ` is also looking at `ν` and `δ`, which come **after** it. "
                "When the model is guessing, those letters are in the future. [03_03](03_03_hiding_the_future.ipynb) hides them. Here we let every letter see everything, to keep it simple.\n\n"
                "👉 **Predict first:** the last row should be the one we just did. Will it match `[0.212, 0.212, 0.576]`?",
            ),
            code(
                """all_scores = X @ X.T
all_weights = torch.softmax(all_scores, dim=-1)  # dim=-1: along each row
all_context = all_weights @ X

print("scores")
print(all_scores)
print("weights (each row adds up to 1)")
print(all_weights)
print("context vectors, one row per letter")
print(all_context)"""
            ),
            saw(
                "The last row of the weights is `[0.2119, 0.2119, 0.5761]`, the same as before, and every row adds up to 1. "
                '`dim=-1` means "along the last direction", which here is along each row.'
            ),
            md(
                "## 5. Full example: the real embeddings of `ἄνδρα`\n\n"
                'Now the real thing: the five letters of `ἄνδρα` ("man"), each with its real 32-number embedding row from our trained model '
                "(still no learned attention, we only use `model.embedding`). The picture is a **heatmap**, a table drawn as coloured boxes. "
                "Row = the letter that's looking, column = the letter it looks at, and the number is its attention weight. The code has exactly the shape of what we did by hand."
            ),
            code(CH3_LOAD),
            code(
                """ids = torch.tensor(tokenizer.encode("ἄνδρα"))
with torch.no_grad():
    X = model.embedding(ids)  # shape (5, 32): the real rows, not invented ones
    scores = X @ X.T
    weights = torch.softmax(scores, dim=-1)
    context = weights @ X

print("X", tuple(X.shape), " scores", tuple(scores.shape))
print("context", tuple(context.shape))
print("each row adds up to:", weights.sum(dim=-1))
print("weight each letter gives itself:", weights.diagonal())"""
            ),
            saw(
                "`X` is `(5, 32)`, the scores are `(5, 5)`, and the context is `(5, 32)`: one blended row per letter, as long as the embedding. Each row of weights adds up to 1."
            ),
            code(
                CH3_HEATMAP.replace(
                    "# the heatmap function, drawn in 03_01",
                    "# our helper that draws one heatmap",
                )
            ),
            code(
                """fig = studio_figure("Simple attention", "Every letter looks mostly at itself", (8, 6.5))
letters = list("ἄνδρα")
draw_weights(
    fig, (0.14, 0.08, 0.66, 0.62), weights.numpy(), letters,
    "attention weights: each row adds up to 1",
)
plt.show()
biggest = [letters[j] for j in weights.argmax(dim=-1)]
print("each letter's biggest weight is on:", biggest)"""
            ),
            saw(
                "The biggest weight in every row sits on the diagonal, so each letter mostly looks at **itself** (the printed weights on the diagonal are all above 0.9 for the trained model). "
                "Why? A row dotted with itself is its squared length, and here that tends to be bigger than its dot product with any other row."
            ),
            q(
                "One quick catch: is that useful?",
                'Not very. If every letter mostly listens to itself, the "mix" is nearly just the letter again, and nothing from the earlier letters comes through. '
                "We want to listen to the letters that **help**, not the ones that merely look the same.\n\n"
                "The way out is to let the model learn what to look for, and that's [03_02](03_02_trainable_attention.ipynb).",
            ),
            *questions(
                [
                    (
                        "Why softmax? Why not just divide each score by the total?",
                        "Scores can be negative (see the red boxes in [A_03](../A_toolbox/A_03_dot_product.ipynb)), and then dividing by the total makes no sense. "
                        "Softmax makes every score positive first, so the shares always work out.",
                    ),
                    (
                        "Is the context vector the same size as the embedding?",
                        "Yes. 32 numbers per letter in, 32 numbers per letter out. That's why the output can go straight into the next layer.",
                    ),
                ]
            ),
            recap(
                [
                    "Simple attention: score every letter with a dot product, softmax the scores into weights that add up to 1, then mix the rows by those weights.",
                    "The result is one context vector per letter, as long as the embedding (32 numbers for us).",
                    "With nothing to learn, every letter mostly looks at itself, so the model needs to learn what to look for.",
                ],
                "[03_02: Trainable attention](03_02_trainable_attention.ipynb): query, key and value.",
                can=[
                    "score letters against each other with a dot product",
                    "turn scores into shares with softmax, and blend the rows into a context vector",
                    "do it for every letter at once with `X @ X.T`",
                ],
            ),
        ],
    )

    # =====================================================================
    # 03_attention/03_02_trainable_attention.ipynb
    # =====================================================================
    save(
        "03_attention/03_02_trainable_attention.ipynb",
        [
            title("03_02 Trainable attention", "section 3.4"),
            where("model reads", ["src/homer_ai/model.py"]),
            learn(
                [
                    "Why simple attention isn't enough, and what query, key and value fix",
                    "The library picture: what I look for, the label on each book, what's inside",
                    "Why the scores are divided by the square root of 32",
                    "How to rebuild the attention step of `model.py` and check it against the real model",
                ]
            ),
            words(
                [
                    (
                        "query",
                        "a list of numbers a letter makes to say what it is looking for",
                        "`Q = model.query(x)`",
                    ),
                    (
                        "key",
                        "a list of numbers a letter makes to say what it offers, like the label on a book",
                        "`K = model.key(x)`",
                    ),
                    (
                        "value",
                        "a list of numbers a letter makes to hand over when others pay attention to it, like what is inside the book",
                        "`V = model.value(x)`",
                    ),
                    (
                        "linear layer",
                        "a table of learned weights that turns a row of numbers into another row (`nn.Linear`; see [A_04](../A_toolbox/A_04_pytorch_in_this_repo.ipynb))",
                        "32 numbers in, 32 numbers out",
                    ),
                ]
            ),
            *background(),
            q(
                "What was wrong with simple attention?",
                "In [03_01](03_01_simple_attention.ipynb) every letter looked mostly at **itself**, because the same row was used for everything: for looking, for being looked at, and for the mix. "
                'With only one row, a letter can\'t learn "I want the vowel just before me".\n\n'
                "So we give each letter three different rows, made from its embedding by three learned layers. Think of a library:\n\n"
                "```\nin a library                          in our model\nwhat I am looking for (my request)  →  query  (Q)\nthe label on each book's spine      →  key    (K)\nwhat is inside the book             →  value  (V)\n```\n\n"
                "Every letter is both a reader and a book. As a reader, it makes a query. As a book, it has a key (a label) and a value (its contents). "
                "A reader compares their query with every key (a dot product, which gives the score). Then they read the books that match well. And note that the mix is of the **values**, not of the labels.",
            ),
            step_intro(),
            q(
                "Step 1: how are Q, K and V made?",
                "By three **linear layers**. Each one is a table of learned weights that turns a row of 32 numbers into another row of 32. "
                "In `model.py` that's `self.query = nn.Linear(32, 32, bias=False)`, and the same for `key` and `value`. "
                "`bias=False` means the layer has no bias, the extra learned number a linear layer normally adds to each output ([01_02](../01_foundations/01_02_neural_networks.ipynb)). Attention works fine without it. "
                "The layers start out random and are learned in training, so the model finds out for itself what's worth asking, labelling and handing over.\n\n"
                "We load our trained model (we only read it, we never save it) and look at the shapes for `ἄνδρα`.\n\n"
                "👉 **Predict first:** the embedding rows `x` have shape `(5, 32)`. What shape will `Q`, `K` and `V` have? How many weights are in one layer, and in all three?",
            ),
            code(CH3_LOAD),
            code(
                """ids = torch.tensor(tokenizer.encode("ἄνδρα"))
with torch.no_grad():
    x = model.embedding(ids)
    Q, K, V = model.query(x), model.key(x), model.value(x)

print("x", tuple(x.shape), " Q", tuple(Q.shape))
print("K", tuple(K.shape), " V", tuple(V.shape))
one_layer = model.query.weight.numel()
print("weights in one layer:", one_layer, " in all three:", 3 * one_layer)"""
            ),
            saw(
                "All four are `(5, 32)`: one row of 32 numbers per letter. Each layer has 32 × 32 = 1,024 weights, so that's 3,072 for the three. "
                "(Out of our model's 11,840 in all. The rest is the embedding, 4,384, and the `head`, 4,384.)"
            ),
            q(
                "Step 2: do queries and keys really change who looks at whom?",
                "Yes. Here's a tiny example with the invented 2-number rows from 03_01 (`ἄ = [1, 0]`, `ν = [0, 1]`, `δ = [1, 1]`). "
                "Let the key be the row as it is, but let the **query** swap its two numbers (a learned layer could learn exactly this): `ἄ → [0, 1]`, `ν → [1, 0]`, `δ → [1, 1]`.\n\n"
                "```\nscore of ν looking at:  ἄ: [1,0]·[1,0] = 1     ν: [1,0]·[0,1] = 0     δ: [1,0]·[1,1] = 1\n```\n\n"
                "👉 **Predict first:** in 03_01 each letter's top score was itself. Will `ν`'s top score still be itself?",
            ),
            code(
                """X = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
swap = torch.tensor([[0.0, 1.0], [1.0, 0.0]])  # a table that swaps the two numbers

tiny_Q = X @ swap  # queries: the rows with their two numbers swapped
tiny_K = X          # keys: the rows as they are
print(tiny_Q @ tiny_K.T)"""
            ),
            saw(
                "Row `ν` is `[1, 0, 1]`. Its score with itself is now **0**, and it prefers `ἄ` and `δ`. "
                "So a query that differs from the key lets a letter look for something other than itself. And training finds a good table on its own."
            ),
            md(
                "**Look at:** the picture, starting from the left. `ν` is the reader. Its query (orange) is compared with each letter's key (green), "
                "the scores become shares, and the values (grey) are mixed by those shares. "
                "To keep it small, each value is just the letter's row as it is."
            ),
            code(DIAGRAM_QKV),
            q(
                "Step 3: why divide the scores by the square root of 32?",
                "The scores go into softmax, and softmax gets **too sharp** when the scores are big. One letter takes nearly all the share, the others get almost nothing, "
                "and training learns slowly (the model can't try out the other letters). A tiny example:\n\n"
                "```\nscores [1, 2, 3]     → shares?\nscores [10, 20, 30]  → shares?     (same order, ten times bigger)\n```\n\n"
                "👉 **Predict first:** will the second list's shares be ten times different, or something more extreme?",
            ),
            code(
                """print(torch.softmax(torch.tensor([1.0, 2.0, 3.0]), dim=0))
print(torch.softmax(torch.tensor([10.0, 20.0, 30.0]), dim=0))"""
            ),
            saw(
                "`[0.09, 0.24, 0.67]`, then `[0.0000, 0.0000, 1.0000]`. The biggest score takes everything."
            ),
            q(
                "But why would our scores be big?",
                "A dot product adds up one product per number, so more numbers (a longer row) means bigger scores. "
                "Random rows with 2 numbers give small scores, and rows with 512 numbers give huge ones. The score grows roughly like `√d`, where `d` is the row length "
                "(`√` is the square root: `√32` is about 5.7). Dividing by `√d` brings it back to a steady size.\n\n"
                "Let's measure it with random numbers: 10,000 dot products for each row length `d`, and the typical size of a score (the standard deviation, a measure of spread).\n\n"
                "👉 **Predict first:** for `d = 512`, about how big will a raw score be (is it near 1, or near `√512 ≈ 22.6`)?",
            ),
            code(
                """torch.manual_seed(0)
for d in [2, 32, 512]:
    q = torch.randn(10000, d)  # 10,000 random rows (random numbers around 0)
    k = torch.randn(10000, d)
    raw = (q * k).sum(dim=-1)  # 10,000 dot products
    scaled = raw / d**0.5
    print(f"d = {d:>3}: typical raw score {raw.std():5.1f}", end="   ")
    print(f"after dividing by √d {scaled.std():4.1f}")"""
            ),
            saw(
                "The raw score grows with the row length (about 1.4, 5.7, 22.6, the same as `√d`), but after dividing by `√d` it stays near 1 every time. "
                '(`d**0.5` means "`d` to the power of one half", which is the square root.) '
                "Next, the effect on softmax itself: one query against 5 random keys, 1,000 times, and the average of the **biggest** share."
            ),
            code(
                """torch.manual_seed(0)
for d in [2, 32, 512]:
    raw_top = scaled_top = 0.0
    for _ in range(1000):
        keys = torch.randn(5, d)
        query = torch.randn(d)
        s = keys @ query  # 5 scores
        raw_top += torch.softmax(s, dim=0).max().item() / 1000
        scaled_top += torch.softmax(s / d**0.5, dim=0).max().item() / 1000
    print(f"d = {d:>3}: biggest share, raw scores {raw_top:.2f}", end="   ")
    print(f"scaled scores {scaled_top:.2f}")"""
            ),
            saw(
                "With raw scores, the biggest share climbs towards 1 as `d` grows (0.51, 0.85, 0.96), so the softmax goes sharp. With scaled scores it stays about the same, near 0.45, whatever `d` is. "
                "That's why `model.py` divides by `x.shape[-1] ** 0.5`, which for us is `√32`."
            ),
            q(
                "Step 4: can I rebuild the attention step of `model.py` myself?",
                "Yes. We take the same `x`, `Q`, `K`, `V` as above and do each line ourselves, then peek inside the real model to compare. "
                "Scores first: every query against every key, divided by `√32`.\n\n"
                "👉 **Predict first:** what shape will `scores` have, and what does the box in row 2, column 3 mean?",
            ),
            code(
                """scores = (Q @ K.T) / (x.shape[-1] ** 0.5)
print(tuple(scores.shape))
print(scores)"""
            ),
            saw(
                "A `(5, 5)` table. The box in row 2, column 3 is how well the 2nd letter's query matches the 3rd letter's key. In other words, how much `ν` wants to listen to `δ`."
            ),
            q(
                "Step 5: softmax, then the mix",
                "Same as in 03_01, except that the scores now come from queries and keys, and the mix is of the **values**:\n\n"
                "```\nweights = softmax(scores)        (each row adds up to 1)\ncontext = weights @ V            (a mix of the value rows)\n```",
            ),
            code(
                """weights = torch.softmax(scores, dim=-1)
my_context = weights @ V
print("weights", tuple(weights.shape), " context", tuple(my_context.shape))
print("each row adds up to:", weights.sum(dim=-1))"""
            ),
            saw(
                "`(5, 5)` weights, each row adding up to 1, and a `(5, 32)` context: one mixed row per letter."
            ),
            q(
                "Does it match the real model?",
                "`forward` only gives back the final scores, which are `model.head` applied to the context. "
                "So we check our context by passing it through `model.head` ourselves and comparing with what the real model returns. "
                "(`head` here is the last layer, not an attention head. Those come in 03_04.)\n\n"
                "👉 **Predict first:** will `model.head(my_context)` equal the real model's scores?",
            ),
            code(
                """with torch.no_grad():
    real_scores = model(ids)  # the real model's scores, one row per letter
my_scores = model.head(my_context).detach()

print("same as the real model?", torch.allclose(my_scores, real_scores, atol=1e-5))
rows = [torch.allclose(my_scores[i], real_scores[i], atol=1e-5) for i in range(5)]
print("row by row:   ", rows)"""
            ),
            saw(
                "No, not the same: `False`. The row-by-row list shows that the first four rows differ. Only the last row matches, because the last letter is allowed to see all five letters anyway. "
                "We're missing one line from `model.py`: the **mask**, which hides the letters that come *after* each one. "
                "03_03 explains it. Here we just copy the real line, so the check can finish."
            ),
            code(
                """mask = torch.triu(torch.ones(5, 5) * float("-inf"), diagonal=1)  # explained in 03_03
weights = torch.softmax(scores + mask, dim=-1)
my_context = weights @ V
my_scores = model.head(my_context).detach()

print("same as the real model now?", torch.allclose(my_scores, real_scores, atol=1e-5))
rows = [torch.allclose(my_scores[i], real_scores[i], atol=1e-5) for i in range(5)]
print("row by row:", rows)"""
            ),
            saw(
                "`True` for all five rows. Our step-by-step attention is the same as the one inside `HomerGreekLLM`, number for number."
            ),
            *questions(
                [
                    (
                        "Why does the model need both a key and a value? Why not one row?",
                        "Because what makes a letter easy to **find** (its label) isn't always what we want to take from it (its contents). "
                        "With two separate layers, the model can learn each on its own.",
                    ),
                    (
                        "Why is `bias=False` in the three layers?",
                        "A bias is one extra learned number added to every output. Attention works fine without it, so we leave it out. Fewer weights, same job.",
                    ),
                ]
            ),
            recap(
                [
                    "Three learned layers turn each embedding into a query (what I look for), a key (my label) and a value (my contents); scores are queries against keys.",
                    "Scores are divided by √32 so softmax doesn't get too sharp: raw scores grow like √d with the row length, scaled ones stay near 1.",
                    "Scores, softmax, then `weights @ V` rebuilds the real model's context exactly, once the mask is added.",
                ],
                "[03_03: Hiding the future](03_03_hiding_the_future.ipynb): the mask line we just copied.",
                can=[
                    "say what the query, the key and the value are for",
                    "compute trainable attention in three lines: scores, softmax, `weights @ V`",
                ],
            ),
        ],
    )

    # =====================================================================
    # 03_attention/03_03_hiding_the_future.ipynb
    # =====================================================================
    save(
        "03_attention/03_03_hiding_the_future.ipynb",
        [
            title("03_03 Hiding the future", "section 3.5"),
            where("model reads", ["src/homer_ai/model.py"]),
            learn(
                [
                    "Why a letter must not see the letters after it",
                    "How -inf and softmax hide the future: softmax of -inf is 0",
                    "How `torch.triu` builds the mask, and the triangle it makes",
                    "Where the mask is in `model.py`, and a test that nothing leaks",
                ]
            ),
            words(
                [
                    (
                        "causal mask",
                        "a table added to the scores that hides the letters after each one, so only earlier letters can influence it (cause before effect)",
                        "`torch.triu(..., diagonal=1)`",
                    ),
                    (
                        "-inf",
                        "minus infinity: a number lower than every other number; softmax turns it into a share of exactly 0",
                        '`float("-inf")`',
                    ),
                ]
            ),
            *background(),
            q(
                "Why can't a letter look at the letters after it?",
                "Remember how the practice questions are made ([02_04](../02_text_to_numbers/02_04_sliding_window.ipynb)): the answer for each letter is simply the **next** letter. "
                "If the model could see the next letter, it would just copy it. That's a perfect score in practice, and useless when it has to write new text, "
                "where the future doesn't exist yet. It's like taking a test with the answers printed on the next page. No peeking!\n\n"
                "For `ἄνδρα`, here's who may look at what:\n\n"
                "```\nthe letter that guesses:   ἄ      ν      δ      ρ      α\nmay look at:               ἄ      ἄν     ἄνδ    ἄνδρ   ἄνδρα\nmust NOT see:              νδρα   δρα    ρα     α      (nothing)\n```\n\n"
                "(In the same way, when guessing the 5th letter of any text, letters 6 and later are hidden.) "
                "In the heatmap of [03_01](03_01_simple_attention.ipynb), row = the letter looking, column = the letter looked at. So the hidden letters are the cells **above the diagonal**: a triangle.",
            ),
            step_intro(),
            q(
                "Step 1: how can softmax give a letter a share of exactly 0?",
                "Inside, softmax turns each score `s` into `e^s` and then divides by the total. Here `e` is a fixed number, about 2.718. "
                "`e^s` is always positive, and bigger for a bigger `s` (`e^2` = 7.39, `e^1` = 2.72, `e^0` = 1). "
                "If a score is **minus infinity** (`-inf`, lower than any number), `e^-inf` is exactly 0, so that letter gets a share of exactly 0. "
                "So before softmax, we set the score of every hidden letter to `-inf`.\n\n"
                "For `[2, 1, -inf]`: 7.39, 2.72 and 0 add up to 10.11, so the shares are 0.731, 0.269 and 0.\n\n"
                "👉 **Predict first:** what will softmax give for the scores `[2.0, 1.0, -inf]`?",
            ),
            code(
                """import torch

scores = torch.tensor([2.0, 1.0, float("-inf")])
print(torch.softmax(scores, dim=0))
print(torch.softmax(scores[:2], dim=0), "<- the same, with the third score left out")"""
            ),
            saw(
                "`[0.7311, 0.2689, 0.0000]`. The hidden letter gets exactly 0, and the other two share all of the 1. "
                "It's just as if the third score had been left out."
            ),
            q(
                "Step 2: how do we make a table with `-inf` above the diagonal?",
                'With `torch.triu` ("**tri**angle **u**pper"). It keeps the upper triangle of a table and sets everything else to 0. '
                '`diagonal=1` means "start one step above the main diagonal", so each letter still sees itself. We build it in small steps for 5 letters.\n\n'
                "👉 **Predict first:** after `triu`, which boxes will be `-inf` and which 0?",
            ),
            code(
                """ones = torch.ones(5, 5)
print(ones * float("-inf"))  # a table full of -inf
mask = torch.triu(ones * float("-inf"), diagonal=1)
print(mask)"""
            ),
            saw(
                "The first table is `-inf` everywhere. After `triu(..., diagonal=1)` only the boxes **above** the diagonal are `-inf`, and the diagonal and everything below are `0`. "
                "Adding 0 to a score changes nothing, and adding `-inf` hides it. So `scores + mask` hides exactly the future."
            ),
            *exercise(
                "can you build the mask for 3 letters?",
                "The mask above is for 5 letters. Build the one for 3, then see what it does to three letters whose scores are all 0.\n\n"
                "👉 **Predict first:** with all scores equal, what shares does the third letter give to letters 1, 2 and 3? And the first letter?",
                starter="""
            # 👉 do: a 3×3 mask: -inf above the diagonal, 0 on it and below it
            small_mask = None

            # 👉 do: add the mask to 3×3 scores that are all 0, then softmax along each row
            small_shares = None
            """,
                solution="""
            small_mask = torch.triu(torch.ones(3, 3) * float("-inf"), diagonal=1)
            small_shares = torch.softmax(torch.zeros(3, 3) + small_mask, dim=-1)
            print(small_shares)
            """,
                check="""
            check(
                "the 3×3 mask",
                small_mask,
                [[0.0, float("-inf"), float("-inf")], [0.0, 0.0, float("-inf")], [0.0, 0.0, 0.0]],
                coach="If the diagonal is -inf too, you left out diagonal=1: then the first "
                "letter can't even see itself, and softmax of a row of only -inf gives nan. "
                "If the -inf are BELOW the diagonal, you used tril: that hides the past.",
            )
            check(
                "the shares of 3 letters with equal scores",
                small_shares,
                [[1.0, 0.0, 0.0], [0.5, 0.5, 0.0], [1 / 3, 1 / 3, 1 / 3]],
                coach="Each ROW has to add up to 1, so softmax runs along the rows: dim=-1. "
                "And the mask is added to the scores before softmax, not after.",
            )
            """,
                hint='The mask is the line of step 2, `torch.triu(ones * float("-inf"), diagonal=1)`, with a 3 × 3 table of ones. The scores are `torch.zeros(3, 3)`.',
                answer="The shares come out as a staircase:\n\n```\n1.00  0.00  0.00\n0.50  0.50  0.00\n0.33  0.33  0.33\n```\n\n"
                "The first letter sees only itself, the second shares between two, the third between three. Nothing goes to a later letter.",
            ),
            q(
                "Step 3: what does it look like on the real attention of `ἄνδρα`?",
                "We use our model's real queries, keys and values (as in [03_02](03_02_trainable_attention.ipynb)), written once as a function that can switch the mask on or off. "
                "Then we draw the weights as a heatmap.\n\n"
                "👉 **Predict first:** what will the top row (the letter `ἄ`, which may only see itself) be?",
            ),
            code(CH3_LOAD),
            code(CH3_HEATMAP),
            code(
                """def attend(x, masked):
    Q, K, V = model.query(x), model.key(x), model.value(x)
    scores = (Q @ K.T) / (x.shape[-1] ** 0.5)
    if masked:
        n = len(x)
        scores = scores + torch.triu(torch.ones(n, n) * float("-inf"), diagonal=1)
    weights = torch.softmax(scores, dim=-1)
    return weights, weights @ V


ids = torch.tensor(tokenizer.encode("ἄνδρα"))
with torch.no_grad():
    weights, context = attend(model.embedding(ids), masked=True)

fig = studio_figure("The causal mask", "The future is hidden: a triangle", (8, 6.5))
draw_weights(
    fig, (0.14, 0.08, 0.66, 0.62), weights.numpy(), list("ἄνδρα"),
    "attention weights with the mask: each row adds up to 1",
)
plt.show()
print("each row adds up to:", weights.sum(dim=-1))"""
            ),
            saw(
                "A triangle. Everything above the diagonal is exactly `0.00`. The top row `ἄ` is `[1, 0, 0, 0, 0]` (it can only see itself), "
                "the second row spreads its share over `ἄ` and `ν`, and so on. Every row still adds up to 1, because softmax shares out the 1 among the letters that are left."
            ),
            q(
                "Can I prove nothing leaks?",
                "Here's a strong test: change only the **last** letter (`ἄνδρα` → `ἄνδρο`) and see whether the output for the earlier letters changes. "
                "With the mask it mustn't, because the earlier letters can't see the last one. We test the mixed result of each letter, with the mask on and off, "
                "and then the real model's final scores for the first four positions.\n\n"
                "👉 **Predict first:** with the mask on, which of the 5 letters change? And with the mask off?",
            ),
            code(
                """other = torch.tensor(tokenizer.encode("ἄνδρο"))  # only the last letter differs

with torch.no_grad():
    for masked in (True, False):
        _, context_a = attend(model.embedding(ids), masked)
        _, context_b = attend(model.embedding(other), masked)
        unchanged = [torch.allclose(context_a[i], context_b[i]) for i in range(5)]
        state = "on " if masked else "off"
        print(f"mask {state}: context unchanged for ἄ ν δ ρ α? {unchanged}")

    same_real = torch.allclose(model(ids)[:4], model(other)[:4])
    print("real model: first 4 outputs unchanged?", same_real)"""
            ),
            saw(
                "With the mask **on**, the first four letters are unchanged (`True`) and only the last changes, because the last letter itself changed. "
                "With the mask off, all five change. The earlier letters had been peeking at the last one. "
                "And the real model passes the test too."
            ),
            q(
                "Where is the mask in `model.py`?",
                "Here are the real lines, printed straight from the code. Our function above does the same thing and, as the check below shows, gives the same answer as the real model.\n\n"
                "👉 **Predict first:** which two lines mention the mask?",
            ),
            code(
                """import inspect

from homer_ai.model import HomerGreekLLM

for line in inspect.getsource(HomerGreekLLM.forward).splitlines():
    if "mask" in line or "softmax" in line:
        print(line.strip())

with torch.no_grad():
    mine = model.head(attend(model.embedding(ids), masked=True)[1])
    print("same scores as the real model?", torch.allclose(mine, model(ids)))"""
            ),
            saw(
                '`mask = torch.triu(torch.ones(seq_len, seq_len) * float("-inf"), diagonal=1)` builds the triangle, '
                "and `weights = F.softmax(scores + mask, dim=-1)` adds it to the scores before softmax. "
                "The check prints `True`, so our function and the real model agree."
            ),
            q(
                "Can the model still tell the order of the letters before the one it is guessing from?",
                "We can finally test what [02_06](../02_text_to_numbers/02_06_positions.ipynb) promised. We give our real `HomerGreekLLM` two inputs with the same last letter, "
                "where the letters before it are only swapped: `ἄνδρα` and `νἄδρα` (the first two letters swapped).\n\n"
                "👉 **Predict first:** will its scores after the final `α` (137 numbers) be the same for both?",
            ),
            code(
                """first = torch.tensor(tokenizer.encode("ἄνδρα"))
swapped = torch.tensor(tokenizer.encode("νἄδρα"))

with torch.no_grad():
    scores_first, scores_swapped = model(first), model(swapped)
same_first = torch.allclose(scores_first[0], scores_swapped[0])
same_last = torch.allclose(scores_first[-1], scores_swapped[-1], atol=1e-5)
print("after the 1st letter, same scores?", same_first)
print("after the last letter, same scores?", same_last)"""
            ),
            saw(
                "After the first letter the scores differ, because the first letters themselves differ (`ἄ` against `ν`). "
                "After the last letter they're **identical**. The model can't tell the order of the letters before it, because attention mixes their rows with no idea where each one sits. "
                "That's what the position table from chapter 2 is for, and `model.py` doesn't have one."
            ),
            q(
                "What is dropout, and does our model use it?",
                "**Dropout** is a training trick. At random, it switches off some of the numbers (the attention weights, for example), so the model can't lean too much on any single one. "
                "Our `model.py` doesn't use dropout.",
            ),
            recap(
                [
                    "A letter must not see later letters, or it would copy the answer; they are the cells above the diagonal.",
                    "We add `-inf` to those scores before softmax: softmax of `-inf` is exactly 0, and the other shares still add up to 1.",
                    "`torch.triu(ones * -inf, diagonal=1)` builds the mask, and it is one line of `model.py`; a changed last letter does not move the earlier outputs.",
                ],
                "[03_04: Many heads](03_04_many_heads.ipynb): what if several attention heads look at once?",
                can=[
                    "build the causal mask for any number of letters with `torch.triu`",
                    "say why the hidden scores are `-inf` and not 0",
                    "prove nothing leaks: change the last letter and compare the earlier outputs",
                ],
            ),
        ],
    )

    # =====================================================================
    # 03_attention/03_04_many_heads.ipynb
    # =====================================================================
    save(
        "03_attention/03_04_many_heads.ipynb",
        [
            title("03_04 Many heads", "section 3.6"),
            where("model reads", ["src/homer_ai/model.py"]),
            learn(
                [
                    "Why one way of looking may not be enough",
                    "How to build a tiny 2-head attention and join the heads",
                    "That splitting into heads doesn't add weights",
                    "That our `model.py` has ONE head, and the full GPT of chapter 4 has several",
                ]
            ),
            words(
                [
                    (
                        "head",
                        "a word with three uses: an **attention head** (one complete attention step with its own query, key and value layers), "
                        "`self.head` (the last layer of `model.py`, which makes the 137 scores), and the **output head** (the same last layer, in a general GPT)",
                        "our `model.py` has one of each",
                    ),
                    (
                        "multi-head attention",
                        "several heads running side by side on the same letters, their results joined together",
                        "2 heads of 16 numbers",
                    ),
                    (
                        "concatenate",
                        "join rows end to end into a longer row (`torch.cat`)",
                        "two rows of 16 numbers make one row of 32",
                    ),
                ]
            ),
            *background(),
            q(
                "Why would one head not be enough?",
                "A head makes **one** set of attention weights per letter, so one way of looking. But guessing a letter may need several things at once: "
                "the letter just before, the start of this word, the last vowel. "
                "With one head, the model has to squeeze all of that into one mix. "
                "With several heads, each can search for its own thing. Take `μοῦσα`:\n\n"
                "```\nhead 1: looks at the previous letter      (σ before α)\nhead 2: looks at the start of the word   (μ)\n        └── their results are joined, so the next layer gets both\n```\n\n"
                "That picture is the *idea*. We haven't measured it in our model. Real models built like this work well, and a fair experiment ([05_04](../05_training_on_the_odyssey/05_04_running_fair_experiments.ipynb)) could test it for ours.",
            ),
            q(
                "How do several heads fit in the same space?",
                "Each head works in a smaller space. With 32 numbers per letter and 2 heads, each head makes queries, keys and values of 16 numbers. "
                "Each head returns a context of 16 numbers per letter, and we **concatenate** the two (join them end to end): 16 + 16 = 32 again. "
                "So the result has the same shape as one big head, and it fits wherever one big head fitted.\n\n"
                "```\nhead 1:  (5, 16) ┐\n                  ├─ joined →  (5, 32)\nhead 2:  (5, 16) ┘\n```\n\n"
                "👉 **Predict first:** will 2 heads of 16 have more weights than our 1 head of 32, fewer, or the same?",
            ),
            step_intro(),
            q(
                "Step 1: one head, as a function",
                "We pack the attention of 03_02 and 03_03 (scores, mask, softmax, mix) into one function, so we can call it once per head. "
                "One difference: each head divides by `√16`, the size of its **own** queries, not `√32`. "
                "We load our model only to borrow the real embedding rows of `ἄνδρα`.",
            ),
            code(CH3_LOAD),
            code(CH3_HEATMAP),
            code(
                """from torch import nn


def one_head(x, query, key, value):
    Q, K, V = query(x), key(x), value(x)
    scores = (Q @ K.T) / (Q.shape[-1] ** 0.5)  # Q.shape[-1] is the head's size
    mask = torch.triu(torch.ones(len(x), len(x)) * float("-inf"), diagonal=1)
    weights = torch.softmax(scores + mask, dim=-1)
    return weights, weights @ V


ids = torch.tensor(tokenizer.encode("ἄνδρα"))
with torch.no_grad():
    x = model.embedding(ids)
print("x", tuple(x.shape))"""
            ),
            saw("`x` is `(5, 32)`, the real embedding rows of `ἄνδρα`."),
            q(
                "Step 2: build two heads and join them",
                "Each head has its own three layers, going from 32 numbers down to 16. We make two heads, run each one, and join the results. "
                "These heads are **untrained** (random numbers, seeded so you get the same ones), so all they show is how the pieces fit.\n\n"
                "👉 **Predict first:** what shapes will one head's weights and context have, and the joined result?",
            ),
            code(
                """torch.manual_seed(0)
head_size = 16
heads = [
    (
        nn.Linear(32, head_size, bias=False),  # query
        nn.Linear(32, head_size, bias=False),  # key
        nn.Linear(32, head_size, bias=False),  # value
    )
    for _ in range(2)
]

all_weights, all_contexts = [], []
with torch.no_grad():
    for query, key, value in heads:
        w, c = one_head(x, query, key, value)
        all_weights.append(w)
        all_contexts.append(c)

print("one head: weights", tuple(all_weights[0].shape))
print("one head: context", tuple(all_contexts[0].shape))
joined = torch.cat(all_contexts, dim=-1)  # join end to end, along the numbers
print("joined:", tuple(joined.shape))"""
            ),
            saw(
                "Each head gives `(5, 5)` weights and a `(5, 16)` context. Joined, they make `(5, 32)`, the same shape as our single head's context."
            ),
            q(
                "Do the two heads really look in different ways?",
                "Here are the two sets of weights side by side. Remember these heads are random and untrained, so any difference is just chance. "
                "In training, each head would be pushed towards something useful, and the heads would be pushed to differ.\n\n"
                "👉 **Predict first:** will the two heatmaps be identical?",
            ),
            code(
                """fig = studio_figure("Two heads", "Each head makes its own attention pattern", (12, 6.2))
letters = list("ἄνδρα")
left, right = all_weights[0].numpy(), all_weights[1].numpy()
draw_weights(fig, (0.09, 0.08, 0.38, 0.58), left, letters, "head 1 (untrained)")
draw_weights(fig, (0.56, 0.08, 0.38, 0.58), right, letters, "head 2 (untrained)")
plt.show()"""
            ),
            saw(
                "Both are triangles (each head has the mask), and the numbers differ a little (the 2nd row, for example, is `0.49 / 0.51` against `0.51 / 0.49`). Untrained heads spread their shares almost evenly, so the difference is small. After training, each head would be pushed to look in its own, sharper way."
            ),
            q(
                "How many weights do many heads cost?",
                "You might expect twice the heads to cost twice the weights. Let's count them with real layers: 1, 2, 4 and 8 heads, always splitting the 32 numbers among the heads, "
                "and compare with our model's own one head.\n\n"
                "👉 **Predict first:** the answer for 2 heads. Is it more than our model's 3,072?",
            ),
            code(
                """for n_heads in [1, 2, 4, 8]:
    size = 32 // n_heads
    layers = [nn.Linear(32, size, bias=False) for _ in range(3 * n_heads)]
    total = sum(layer.weight.numel() for layer in layers)
    print(f"{n_heads} head(s) of {size:>2} numbers: {total:,} weights")

ours = sum(layer.weight.numel() for layer in (model.query, model.key, model.value))
print(f"our model.py (1 head of 32): {ours:,} weights")"""
            ),
            saw(
                "3,072 every time, the same as our model's one head. Several smaller heads cost the **same** weights as one big one, because each is smaller by exactly as much as there are more of them."
            ),
            *exercise(
                "what would 16 heads cost?",
                "The table stops at 8 heads. Work out the number for 16 heads yourself, with plain arithmetic.\n\n"
                "👉 **Predict first:** more or fewer weights than 8 heads?",
                starter="""
            n_heads = 16
            head_size = 32 // n_heads

            # 👉 do: the weights of 16 heads: each head has a query, a key and a value layer,
            #        and each layer takes 32 numbers in and gives head_size numbers out (no bias)
            weights_16_heads = None
            """,
                solution="""
            n_heads = 16
            head_size = 32 // n_heads

            weights_16_heads = n_heads * 3 * 32 * head_size
            """,
                check="""
            check(
                "weights of 16 heads",
                weights_16_heads,
                16 * 3 * 32 * (32 // 16),
                coach="One layer from 32 numbers to head_size numbers has 32 × head_size "
                "weights. A head has 3 of them, and there are 16 heads.",
            )
            """,
                hint="Multiply four numbers: the heads, the 3 layers per head, the 32 numbers going in, and `head_size` coming out.",
                answer="3,072 again. Twice the heads, each half the size, so nothing changes. What does change is how little room each head has: with 16 heads, a head works with only 2 numbers per letter.",
            ),
            q(
                "Does the joined result still work with our model?",
                "Yes. It's `(5, 32)`, which is what `model.head` expects. We feed it in just to see the shape (the numbers mean nothing, because these heads are untrained).\n\n"
                "👉 **Predict first:** what shape will the output have?",
            ),
            code(
                """with torch.no_grad():
    out = model.head(joined)
print(tuple(out.shape), "= 5 letters × 137 scores")"""
            ),
            saw(
                "`(5, 137)`. So joined heads can replace our single head without changing anything after it."
            ),
            q(
                "Does our `model.py` have many heads?",
                "No, just **one**. Its three layers are 32 × 32 and it makes one set of weights. "
                "The GPT of [chapter 4](../04_building_the_model/04_00_chapter_overview.ipynb) (`src/homer_ai/gpt.py`) has several. `model.py` stays the small one-head model.",
            ),
            recap(
                [
                    "Several heads each look for something different; each works in a smaller space and their results are joined (`torch.cat`) back to the full size.",
                    "Splitting 32 numbers into heads costs the same weights: 3,072 for 1, 2, 4 or 8 heads.",
                    "Our `model.py` has one head; the GPT of chapter 4 has several (4 in the shipped one).",
                ],
                "[04_00: Building the model](../04_building_the_model/04_00_chapter_overview.ipynb): the full model around attention.",
                can=[
                    "run several heads over the same letters and join their results with `torch.cat`",
                    "work out what any number of heads costs in weights",
                ],
            ),
        ],
    )
