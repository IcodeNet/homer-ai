# ruff: noqa: E501
"""Appendix A, the toolbox: the tools the course uses."""

from .cells import (
    code,
    learn,
    md,
    q,
    questions,
    recap,
    save,
    saw,
    step_intro,
    where,
    words,
)

# =====================================================================
# A_toolbox: Appendix A, the toolbox
# =====================================================================
TOOLBOX_BUILD = (
    "We're building a small **LLM** (large language model) that learns Homer's Greek and guesses what comes next. "
    "This appendix explains the tools we use to build it."
)


def toolbox_background(where_now=None, problem=None):
    """Section 3 heading only. The one-line 'what this solves' lives in SOLVES."""
    return [md("## 3. The idea")]


def toolbox_title(heading):
    return md(
        f"# {heading}\n\n*Course appendix A: the toolbox (our own, not the book's appendix A). Dip in whenever you meet a tool you don't know yet.*\n\n"
        "**How to use this notebook:** click a cell and press Shift+Enter to run just that cell.\n"
        "Before each code cell, write down what you think it will do. Then run it and compare.\n\n"
        "Run the cells in order, from the top. **A cell can use what earlier cells made**, "
        "so if you skip ahead you'll see `NameError: name '…' is not defined`. "
        "If that happens, run the cells above it, or just click Run All."
    )


TOOL_TABLE = """| Tool | What it is, in one line | Why we need it | First needed in |
|---|---|---|---|
| **Python** | the programming language the whole project is written in | everything is Python | every chapter |
| **PyTorch** (`torch`) | a library for maths on big lists of numbers, built for AI | it holds our model, and does the learning | chapters 2-6 |
| **tiktoken** | OpenAI's fast BPE tokenizer | to compare our tokens with GPT-2's | chapter 2 (BPE) |
| **Jupyter** + **ipykernel** | the notebook format, and the helper that runs Python inside it | the course itself is notebooks | every chapter |
| **uv** | installs Python and every library on the list | one command sets everything up | the start |
| **Ruff** | checks and tidies Python code | catches mistakes before you run anything | while writing code |
| **git** | saves snapshots of the project | to go back if something breaks | while writing code |
| **MCP** | a standard way for AI apps to call outside tools | so Claude can ask our model for Greek | chapter 6 |"""


def build():
    # ---------------------------------------------------------------- A_00
    save(
        "A_toolbox/A_00_overview.ipynb",
        [
            toolbox_title("A_00 The toolbox: an overview"),
            where(None, ["pyproject.toml"]),
            learn(
                [
                    "Every tool this course uses, in one line each",
                    "Which chapter needs which tool",
                    "Where the list of tools lives: `pyproject.toml`",
                ]
            ),
            words(
                [
                    (
                        "library",
                        "code someone else wrote and shared, so you can use it instead of writing it yourself",
                        "PyTorch",
                    ),
                    (
                        "`pyproject.toml`",
                        "the project's shopping list: every library it needs",
                        "it lists `torch` and `tiktoken`",
                    ),
                ]
            ),
            *toolbox_background(
                "Everywhere. Each chapter uses some of these tools, and each one is explained the first time it appears. "
                "This appendix is the place to come back to when you want the full story of one tool.",
                "A beginner meets many names at once: PyTorch, tiktoken, uv, Ruff, Jupyter. This notebook gives each one a single plain line, "
                "so none of them is a mystery.",
            ),
            q(
                "Which tools do we use, and why?",
                "Think of this appendix as the tool wall in a workshop, the board where every tool hangs on its own hook. "
                "Nobody studies the wall before they start building. You're in the middle of a job, you need a hole, and you take down the drill. "
                "It's the same here: each notebook is one tool, with a label that says when to reach for it. "
                "The picture is a bit too easy in one way: a hammer works the minute you pick it up, and PyTorch or the dot product need a short read first, which is what these notebooks are for.\n\n"
                "Here's the whole wall, one line per tool:\n\n" + TOOL_TABLE,
            ),
            q(
                "Where is the list of libraries kept?",
                "In `pyproject.toml`, the project's shopping list. When you run `uv sync`, uv reads this list and installs everything on it. "
                "Next to it, `uv.lock` writes down the **exact** version of each library, so everyone gets the same ones.",
            ),
            md("## 4. Step by step\n\nOne small check: can Python see the libraries?"),
            q(
                "How do I check that a library is installed?",
                "Import it. If it isn't installed, Python stops with `ModuleNotFoundError` (see [00_02](../00_start_here/00_02_reading_errors.ipynb)). "
                "Most libraries also print their version number.\n\n👉 **Predict first:** will both imports work?",
            ),
            code(
                """
import tiktoken
import torch

print("torch", torch.__version__)
print("tiktoken", tiktoken.__version__)
"""
            ),
            saw(
                "Two version numbers, so both libraries are installed. "
                "If you see a NumPy warning above them, don't worry. It's harmless, and the questions at the end explain it."
            ),
            md(
                "## 5. Full example\n\nThe notebooks in this appendix, in the order you'll probably need them:\n\n"
                "| Notebook | Open it when… |\n|---|---|\n"
                "| [A_01 Python essentials](A_01_python_essentials.ipynb) | a piece of Python looks strange (an optional Python primer) |\n"
                "| [A_02 PyTorch and tensors](A_02_pytorch_tensors.ipynb) | you meet `torch` or the word *tensor* |\n"
                "| [A_03 The dot product](A_03_dot_product.ipynb) | you meet `@`, or before chapter 3 (attention) |\n"
                "| [A_04 PyTorch in this repo](A_04_pytorch_in_this_repo.ipynb) | you want to understand `model.py` and the training loop |\n"
                "| [A_05 tiktoken and BPE](A_05_tiktoken_bpe.ipynb) | after 02_03 (BPE) |\n"
                "| [A_06 The other tools](A_06_other_tools.ipynb) | uv, Ruff, Jupyter, git or MCP confuse you |\n"
                "| [A_07 One shared model file](A_07_organising_code.ipynb) | you wonder why `model.py` is imported by two other files |"
            ),
            *questions(
                [
                    (
                        "I see 'Failed to initialize NumPy'. Is something broken?",
                        "No, it's harmless. NumPy is another maths library that PyTorch can use, and PyTorch prints this warning when it can't find it. This project installs numpy with `uv sync`, so if you do see the warning, run `uv sync` again.",
                    ),
                ]
            ),
            recap(
                [
                    "PyTorch builds the model, tiktoken compares tokenizers, and uv, Ruff, Jupyter, git and MCP do the rest.",
                    "`pyproject.toml` is the shopping list, and `uv sync` installs it.",
                ],
                "[A_01 Python essentials](A_01_python_essentials.ipynb).",
            ),
        ],
    )

    # ---------------------------------------------------------------- A_01
    save(
        "A_toolbox/A_01_python_essentials.ipynb",
        [
            toolbox_title("A_01 Python essentials"),
            where(None, ["src/homer_ai/tokenizer.py", "src/homer_ai/train_homer.py"]),
            learn(
                [
                    "Just the Python this course uses, one idea at a time",
                    "Storing values, text, lists and dictionaries",
                    "Loops, decisions and functions",
                    "How to explore on your own with `print`, `type` and `help`",
                ]
            ),
            words(
                [
                    ("variable", "a name that points to a value", '`word = "μῆνιν"`'),
                    (
                        "string",
                        "a piece of text, written between quotes",
                        '`"Ἀχιλῆος"`',
                    ),
                    (
                        "list",
                        "an ordered row of values, in square brackets",
                        '`["α", "β"]`',
                    ),
                    (
                        "dictionary",
                        "a lookup table from keys to values, in curly brackets",
                        '`{"α": 0}`',
                    ),
                    (
                        "function",
                        "a named block of code that runs when you call it",
                        "`def greet(name):`",
                    ),
                    (
                        "loop",
                        "code that repeats for every item in a list",
                        "`for letter in word:`",
                    ),
                ]
            ),
            *toolbox_background(
                "Everywhere: every code cell in the course is Python.",
                "The notebooks use a small set of Python ideas again and again. Learning just these is enough to follow the whole course.",
            ),
            step_intro(),
            q(
                "How do I store a value?",
                'Give it a name with `=`. The name is a **variable**: a label pointing at a value. `=` here means "let this name point to", not "is equal to".\n\n'
                "Text goes between quotes, and is called a **string**.\n\n👉 **Predict first:** what will `print` show?",
            ),
            code(
                """
word = "μῆνιν"
print(word)
print(len(word))
"""
            ),
            saw("The word, then `5`. `len` counts how many characters a string has."),
            q(
                "How do I keep several values together?",
                "In a **list**: values in square brackets, separated by commas. Positions start at **0**, so `[0]` is the first one.\n\n"
                "👉 **Predict first:** what is `letters[0]`, and what is `letters[-1]`?",
            ),
            code(
                """
letters = ["μ", "ῆ", "ν", "ι", "ν"]
print(letters[0])
print(letters[-1])
"""
            ),
            saw(
                "`μ` and `ν`. A negative position counts from the end: `-1` is the last item."
            ),
            q(
                "How do I take a piece of a string or list?",
                "With a **slice**: `[start:end]`. It takes from `start` up to, but **not** including, `end`. "
                'A blank start means "from the beginning"; a blank end means "to the end". The training code uses this to cut the text into pieces.\n\n'
                "👉 **Predict first:** what is `word[1:3]`?",
            ),
            code(
                """
word = "μῆνιν"
print(word[1:3])
print(word[:2], word[2:])
"""
            ),
            saw(
                "`ῆν`, then `μῆ νιν`. Positions 1 and 2, then the word split at position 2."
            ),
            q(
                "What's the difference between `append` and `extend`?",
                "Both add to a list. `append` adds **one** item, even if that item is itself a list. `extend` pours the items in **one by one**.\n\n"
                "👉 **Predict first:** what do `x` and `y` look like?",
            ),
            code(
                """
x = [1]
x.append([2, 3])
y = [1]
y.extend([2, 3])
print(x)
print(y)
"""
            ),
            saw(
                "`[1, [2, 3]]` (a box inside the box) and `[1, 2, 3]`. `load_data` uses `extend` to add a whole book of numbers."
            ),
            q(
                "How do I look something up?",
                "With a **dictionary**: pairs of `key: value` in curly brackets. Square brackets look up a key, and crash with `KeyError` if it's missing. "
                "`.get(key, backup)` gives the backup value instead. Our tokenizer uses this for unknown characters.\n\n"
                '👉 **Predict first:** what does `.get("ω", 99)` give?',
            ),
            code(
                """
char_to_id = {"α": 0, "β": 1}
print(char_to_id["α"])
print(char_to_id.get("ω", 99))
"""
            ),
            saw("`0`, then `99`, because `ω` isn't in the dictionary."),
            q(
                "How do I repeat something for every item?",
                "With a **for loop**. The indented lines run once for each item. **Indentation** (the spaces at the start of a line) shows which lines belong to the loop.\n\n"
                "`if` runs its indented lines only when something is true.\n\n👉 **Predict first:** which letters get printed?",
            ),
            code(
                """
for letter in "μῆνιν":
    if letter == "ν":
        print("found ν")
"""
            ),
            saw(
                '`found ν` twice, because `ν` appears twice. `==` means "is equal to" (one `=` gives a name; two compare).'
            ),
            q(
                "What is a list comprehension?",
                "A short way to build a list from a loop, in one line: `[what_to_keep for item in things]`. Our tokenizer's `encode` is one.\n\n"
                "👉 **Predict first:** what list do we get?",
            ),
            code(
                """
char_to_id = {"μ": 0, "ῆ": 1, "ν": 2, "ι": 3}
[char_to_id[c] for c in "μῆνιν"]
"""
            ),
            saw(
                "`[0, 1, 2, 3, 2]`: one number per letter. That's exactly what encoding text means."
            ),
            q(
                "What is a function, and why didn't my `print` inside it run?",
                "A **function** is a named block of code. `def` only **defines** it, like building a machine. Nothing happens until you **call** it, with brackets, "
                "like switching the machine on. `return` sends a value back to whoever called it.\n\n"
                "👉 **Predict first:** does anything print when the `def` block runs?",
            ),
            code(
                """
def greet(name):
    return "Χαῖρε " + name


print("defined, not called yet")
print(greet("Homer"))
"""
            ),
            saw(
                'The `def` printed nothing. Only `greet("Homer")`, the call, ran the code inside.'
            ),
            q(
                "What is an f-string?",
                "A string with `f` in front. Anything in `{curly brackets}` is replaced by its value. The training loop prints its progress this way.\n\n"
                "👉 **Predict first:** what is printed?",
            ),
            code(
                """
epoch = 50
loss = 2.38123
print(f"Epoch {epoch} | loss {loss:.2f}")
"""
            ),
            saw('`Epoch 50 | loss 2.38`. `:.2f` means "show 2 decimal places".'),
            q(
                "What is a class?",
                "A **class** is a blueprint for making objects that keep their own data and functions together. `CharTokenizer` is a class: "
                "each tokenizer you make from it remembers its own vocabulary. A function that belongs to a class is called a **method**, like `encode`.\n\n"
                '👉 **Predict first:** what does `t.vocab_size` give for the text `"abc"`?',
            ),
            code(
                """
from homer_ai.tokenizer import CharTokenizer

t = CharTokenizer("abc")
print(t.vocab_size)
print(t.encode("ab"))
"""
            ),
            saw(
                "`5`: the 3 letters plus our 2 special tokens. Then `[0, 1]`. `from … import …` loaded the class from our own file, `tokenizer.py`."
            ),
            q(
                "How do I explore something I don't understand?",
                "Three helpers:\n\n- `print(x)` shows the value\n- `type(x)` shows what kind of thing it is\n- `help(x)` shows its description\n\n"
                'In VS Code you can also **hover** the mouse over a name.\n\n👉 **Predict first:** what is the type of `{"α": 0}`?',
            ),
            code(
                """
word = "μῆνιν"
numbers = [1, 2]
lookup = {"α": 0}
print(type(word))
print(type(numbers))
print(type(lookup))
"""
            ),
            saw("`str` (string), `list`, `dict` (dictionary)."),
            md(
                "## 5. Full example\n\nEverything together: turn a Greek line into numbers and back, with only the ideas from this notebook."
            ),
            code(
                """
line = "μῆνιν ἄειδε θεὰ"
chars = sorted(set(line))  # every different character, in order
char_to_id = {ch: i for i, ch in enumerate(chars)}
id_to_char = {i: ch for ch, i in char_to_id.items()}

ids = [char_to_id[c] for c in line]
back = "".join(id_to_char[i] for i in ids)

print(f"{len(chars)} different characters")
print(ids)
print(back == line)
"""
            ),
            saw(
                "The numbers for each character, and `True`: decoding gives back the same line. "
                '(`set` keeps one of each, `sorted` puts them in order, `enumerate` numbers them 0, 1, 2…, and `"".join` glues characters into a string.)'
            ),
            *questions(
                [
                    (
                        "I typed a function at the `>>>` prompt and got `SyntaxError`. Why?",
                        "At the `...` prompt, Python thinks you're still inside the function. Press Enter on an empty line to finish it, then call it at a fresh `>>>`.",
                    ),
                    (
                        "Why is the code split into several files?",
                        "So each idea lives in **one** place. Each file is a **module** that others can `import`. If the model's design were copied into two files, "
                        "changing one and forgetting the other would break things.",
                    ),
                ]
            ),
            recap(
                [
                    "Variables name values; strings, lists and dictionaries hold text, rows and lookups.",
                    "Slices cut pieces; `for` repeats, `if` decides, and `def` defines a function that runs only when called.",
                    "`print`, `type` and `help` let you explore anything.",
                ],
                "[A_02 PyTorch and tensors](A_02_pytorch_tensors.ipynb).",
            ),
        ],
    )

    # ---------------------------------------------------------------- A_02
    save(
        "A_toolbox/A_02_pytorch_tensors.ipynb",
        [
            toolbox_title("A_02 PyTorch and tensors"),
            where("numbers", ["src/homer_ai/train_homer.py"]),
            learn(
                [
                    "What PyTorch is, and why AI engineers use it",
                    "What a tensor is, and its shape and type",
                    "How to take pieces of a tensor",
                    'Why a seed makes "random" numbers repeat',
                ]
            ),
            words(
                [
                    (
                        "PyTorch",
                        "a free library for maths on big lists of numbers, built for AI",
                        "`import torch`",
                    ),
                    (
                        "tensor",
                        "PyTorch's list (or table) of numbers",
                        "`torch.tensor([1, 2, 3])`",
                    ),
                    (
                        "shape",
                        "how many numbers a tensor has along each direction",
                        "`(2, 3)`: 2 rows, 3 columns",
                    ),
                    (
                        "dtype",
                        "the kind of number stored: whole numbers or decimals",
                        "`torch.long`, `torch.float32`",
                    ),
                    (
                        "seed",
                        'a starting number for randomness; the same seed gives the same "random" numbers',
                        "`torch.manual_seed(42)`",
                    ),
                ]
            ),
            *toolbox_background(
                "From the stage **numbers** onwards. Once the text is turned into numbers, those numbers live in a tensor, and the model is built from tensors too.",
                "Our model is nothing but numbers being multiplied and added, millions of times. PyTorch does that fast, and keeps track of the maths needed for learning.",
            ),
            q(
                "What's PyTorch?",
                "**PyTorch** is a free library for doing maths on big lists of numbers, made for AI. It was first built at Meta (Facebook's company), and the PyTorch Foundation looks after it.\n\n"
                "Most AI researchers and engineers use it, for two reasons. It's fast, and it can work out how to adjust the numbers so a model improves. That second part is what makes learning possible.\n\n"
                "In code it's called `torch`.",
            ),
            step_intro(),
            q(
                "What's a tensor?",
                "A **tensor** is PyTorch's list of numbers. It looks like a Python list, but PyTorch can do maths on all of it at once, very fast.\n\n"
                "```\nπολλὰ   →   π    ο    λ    λ    ὰ\n            41   40   36   36   110     one tensor of 5 numbers\n```\n\n"
                "👉 **Predict first:** what does `ids * 2` give?",
            ),
            code(
                """
import torch

ids = torch.tensor([41, 40, 36, 36, 110])
print(ids)
print(ids * 2)
"""
            ),
            saw(
                "Every number doubled at once. With a plain list you'd need a loop. (These 5 numbers are `πολλὰ` in our tokenizer.)"
            ),
            q(
                "What's a shape?",
                "The **shape** says how many numbers a tensor has in each direction. A row of 5 numbers has shape `(5,)`. "
                "A table with 2 rows and 3 columns has shape `(2, 3)`.\n\n👉 **Predict first:** what shape does the table have?",
            ),
            code(
                """
row = torch.tensor([41, 40, 36, 36, 110])
table = torch.tensor([[1, 2, 3], [4, 5, 6]])
print(row.shape)
print(table.shape)
"""
            ),
            saw("`torch.Size([5])` and `torch.Size([2, 3])`."),
            q(
                "What's a dtype, and why does the code say `dtype=torch.long`?",
                "The **dtype** (data type) is the kind of number stored. `torch.long` holds whole numbers, which is what token IDs are. There's no token number 4.5. "
                "`torch.float32` holds decimals, which is what the model's weights are.\n\n"
                "`load_data` in `train_homer.py` writes `torch.tensor(ids, dtype=torch.long)` for exactly this reason.\n\n👉 **Predict first:** which dtype does each tensor get?",
            ),
            code(
                """
token_ids = torch.tensor([41, 40], dtype=torch.long)
weights = torch.tensor([0.5, -1.2])
print(token_ids.dtype)
print(weights.dtype)
"""
            ),
            saw(
                "`torch.int64` (that's what `long` means) and `torch.float32`. PyTorch picks decimals for numbers with a point."
            ),
            q(
                "How do I take a piece of a tensor?",
                "With a slice, exactly like a Python list: `[start:end]`. The training loop uses `data[i : i + chunk_size]` to cut a window of 24 tokens from the book.\n\n"
                "👉 **Predict first:** what's `data[1:4]`?",
            ),
            code(
                """
data = torch.tensor([10, 11, 12, 13, 14, 15])
print(data[1:4])
print(data[0].item())
"""
            ),
            saw(
                "`tensor([11, 12, 13])`, then `10`. `.item()` turns a one-number tensor back into a plain Python number."
            ),
            q(
                "Why does the code set `torch.manual_seed(42)`?",
                'A model starts with random weights. Computers make "random" numbers from a starting number, the **seed**. Same seed, same numbers, every run. '
                "That's why our training gives the same loss every time, so experiments can be compared.\n\n"
                "👉 **Predict first:** will the two lines be the same?",
            ),
            code(
                """
torch.manual_seed(42)
print(torch.rand(3))
torch.manual_seed(42)
print(torch.rand(3))
"""
            ),
            saw(
                'The same three "random" numbers twice. `torch.rand(3)` makes 3 random decimals between 0 and 1.'
            ),
            md(
                "## 5. Full example\n\nThe real Book 1 of the Odyssey, as one tensor, the way `train_homer.py` builds it."
            ),
            code(
                """
from homer_ai.train_homer import BOOKS_DIR, load_data

data, tokenizer, text = load_data(BOOKS_DIR)
print(data.shape, data.dtype)
print(data[:10])
print(tokenizer.decode(data[:10].tolist()))
"""
            ),
            saw(
                "One long row of 19,701 whole numbers, one per character. The first 10 decode back to the opening of the poem. "
                "`.tolist()` turns a tensor back into a plain Python list."
            ),
            *questions([]),
            recap(
                [
                    "PyTorch does fast maths on tensors: lists or tables of numbers with a shape.",
                    "Token IDs are whole numbers (`torch.long`). Weights are decimals (`torch.float32`).",
                    "Slices cut pieces of a tensor, and a seed makes randomness repeat.",
                ],
                "[A_03 The dot product](A_03_dot_product.ipynb).",
            ),
        ],
    )

    # ---------------------------------------------------------------- A_03
    save(
        "A_toolbox/A_03_dot_product.ipynb",
        [
            toolbox_title("A_03 The dot product"),
            where("model reads", ["src/homer_ai/model.py"]),
            learn(
                [
                    "How multiplying two lists of numbers gives a similarity score, shown with movies",
                    "What a dot product is, and how to draw it as arrows",
                    "Why we sometimes divide by the lengths (cosine similarity)",
                    "How matrix multiplication is many dot products at once",
                    "Where our model uses it: `Q @ K.T`",
                ]
            ),
            words(
                [
                    (
                        "vector",
                        "a list of numbers; drawn as an arrow from the centre of a graph",
                        "`[5, -4]`",
                    ),
                    (
                        "dot product",
                        "multiply two vectors number by number, then add the results",
                        "`[1, 2]·[3, 4] = 1×3 + 2×4 = 11`",
                    ),
                    (
                        "length (of a vector)",
                        "how long its arrow is: square every number, add them up, take the square root",
                        "`[3, 4]` has length 5",
                    ),
                    (
                        "cosine similarity",
                        "the dot product divided by both lengths, so it only compares direction; always between -1 and 1",
                        "`1` = same way, `-1` = opposite",
                    ),
                    (
                        "matrix",
                        "a table of numbers: rows and columns",
                        "a `2 × 3` table",
                    ),
                    (
                        "matrix multiplication",
                        "every row of one matrix dotted with every column of another",
                        "`A @ B`",
                    ),
                    ("transpose", "flip a matrix so its rows become columns", "`K.T`"),
                    (
                        "query and key",
                        "two vectors the model makes for every character: what it's looking for, and what it offers",
                        "`Q` and `K` in `model.py`",
                    ),
                    (
                        "heatmap",
                        "a table drawn as coloured boxes, so big and small numbers are easy to see",
                        "the `Q @ K.T` picture below",
                    ),
                ]
            ),
            *toolbox_background(),
            q(
                "How can a few numbers say whether two things are similar?",
                "Let's start with movies. Rate each one on two traits, [Action, Romance], from -5 (hate) to +5 (love). "
                "Alex loves action and hates romance, so **Alex = [5, -4]**.\n\n"
                "To score a movie for Alex, multiply the numbers that belong together (action with action, romance with romance), then add:\n\n"
                "| Movie | Vector [Action, Romance] | The maths | Score | Meaning |\n"
                "|---|---|---|---|---|\n"
                "| Die Hard | [5, -3] | (5 × 5) + (-4 × -3) = 25 + 12 | **37** | very similar: both love action, both dislike romance |\n"
                "| The Notebook | [-4, 5] | (5 × -4) + (-4 × 5) = -20 + -20 | **-40** | opposite: the signs clash |\n"
                "| A cooking show | [0, 0] | (5 × 0) + (-4 × 0) = 0 + 0 | **0** | unrelated: no shared traits |\n\n"
                "👉 **Predict first:** will the code below print the same three scores?",
            ),
            code(
                """
alex = [5, -4]
movies = {"Die Hard": [5, -3], "The Notebook": [-4, 5], "A cooking show": [0, 0]}

for name, v in movies.items():
    parts = [f"({a} × {b})" for a, b in zip(alex, v, strict=True)]
    score = sum(a * b for a, b in zip(alex, v, strict=True))
    print(f"{name:<15}{v}  {' + '.join(parts)} = {score}")
"""
            ),
            saw(
                "`37`, `-40` and `0`, the same as the table. "
                '`zip` walks along two lists side by side, and `strict=True` means "they must be the same length". '
                "This multiply-then-add is the **dot product**, written `a · b`."
            ),
            q(
                "Why does multiplying measure similarity?",
                "- **Matching signs boost the score.** `+ × +` and `- × -` are both positive. So two people who both *dislike* romance get a boost, just like two people who both love action.\n"
                "- **Clashing signs lower the score.** `+ × -` is negative: one loves what the other hates.\n"
                "- **Zeros cancel.** If one side doesn't care (0), that trait adds nothing, whatever the other side feels.",
            ),
            q(
                "Can I see it?",
                "Yes. A **vector** is a list of numbers. With two numbers we can draw it as an arrow from the centre `(0, 0)`: "
                "the first number says how far right, the second how far up. Here the right-hand direction is Action and the up direction is Romance.\n\n"
                "👉 **Predict first:** which movie's arrow will point the same way as Alex's?",
            ),
            code(
                """
import matplotlib.pyplot as plt

from homer_ai.studio_style import (
    ACCENT,
    BAD,
    GOOD,
    MUTED,
    studio_axes,
    studio_figure,
)

alex = [5, -4]
arrows = [
    # (name, vector, colour)
    ("Alex", alex, ACCENT),
    ("Die Hard", [5, -3], GOOD),
    ("The Notebook", [-4, 5], BAD),
]

fig = studio_figure("Films as arrows", "Same direction, high score")
ax = studio_axes(fig, (0.1, 0.1, 0.55, 0.7))
for _, v, colour in arrows:
    style = {"arrowstyle": "-|>", "color": colour, "lw": 3, "mutation_scale": 18}
    ax.annotate(
        "", xy=v, xytext=(0, 0),
        arrowprops=style,
    )
ax.axhline(0, color=MUTED, linestyle="--", linewidth=0.8)
ax.axvline(0, color=MUTED, linestyle="--", linewidth=0.8)
ax.set(xlim=(-6, 6), ylim=(-6, 6), aspect="equal")
ax.set_xlabel("Action  (-5 hate, +5 love)")
ax.set_ylabel("Romance  (-5 hate, +5 love)")

# the legend: coloured text beside the plot, with the scores worked out by code
y = 0.72
for name, v, colour in arrows:
    score = sum(a * b for a, b in zip(alex, v, strict=True))
    fig.text(0.7, y, name, color=colour, fontsize=14, fontweight="bold")
    fig.text(0.7, y - 0.05, str(v), color=MUTED, fontsize=12, family="monospace")
    if name != "Alex":
        line = f"Alex · {name} = {score:+d}"
        fig.text(0.7, y - 0.1, line, color=colour, fontsize=11, family="monospace")
    y -= 0.2
plt.show()
"""
            ),
            saw(
                "Alex and Die Hard point almost the same way, and their score is high (`37`). "
                "The Notebook points the opposite way, and its score is very negative (`-40`). "
                "Two arrows at 90° (right angles, like Action-only `[1, 0]` and Romance-only `[0, 1]`) would score `0`, because they have nothing in common."
            ),
            q(
                "One quick catch: does a longer arrow score higher?",
                "Yes, and that can mislead. `[100, 100]` and `[1, 1]` point exactly the same way, but one is much longer.\n\n"
                "The fix is to divide the dot product by the two lengths of the arrows. That gives the **cosine similarity**, "
                "which only looks at direction: `1` means the same way, `0` right angles, `-1` opposite ways.\n\n"
                "👉 **Predict first:** will `[100, 100]·[1, 1]` score more than `[1, 1]·[1, 1]`? And what will their cosine similarities be?",
            ),
            code(
                """
from math import sqrt


def dot(a, b):
    return sum(x * y for x, y in zip(a, b, strict=True))


def length(a):
    return sqrt(dot(a, a))


def cosine(a, b):
    return dot(a, b) / (length(a) * length(b))


print("dot    [1, 1]·[1, 1]    =", dot([1, 1], [1, 1]))
print("dot    [100, 100]·[1, 1] =", dot([100, 100], [1, 1]))
print("cosine [1, 1]·[1, 1]    =", round(cosine([1, 1], [1, 1]), 3))
print("cosine [100, 100]·[1, 1] =", round(cosine([100, 100], [1, 1]), 3))
print("cosine Alex·Die Hard     =", round(cosine([5, -4], [5, -3]), 3))
print("cosine Alex·The Notebook =", round(cosine([5, -4], [-4, 5]), 3))
"""
            ),
            saw(
                "The longer vector has a much bigger dot product, but both pairs have cosine `1.0`. "
                "Alex and Die Hard are close to `1`, and Alex and The Notebook are close to `-1`."
            ),
            step_intro(),
            q(
                "How does PyTorch do it?",
                "`torch.dot` does the whole multiply-and-add in one call. The `@` sign does the same for two vectors. "
                "We use PyTorch because the model's numbers are PyTorch tensors.\n\n"
                "👉 **Predict first:** the dot product of `[1, 2, 3]` and `[4, 5, 6]` is `1×4 + 2×5 + 3×6`. Will both lines print that?",
            ),
            code(
                """
import torch

a = torch.tensor([1.0, 2.0, 3.0])
b = torch.tensor([4.0, 5.0, 6.0])
print(torch.dot(a, b))
print(a @ b)
"""
            ),
            saw("`tensor(32.)` twice: `4 + 10 + 18 = 32`."),
            q(
                "What's matrix multiplication?",
                "A **matrix** is a table of numbers. Multiplying two matrices with `@` means: take every row of the first, and dot it with every column of the second. "
                "Each answer fills one box of the result.\n\n"
                "So it's really just many dot products at once.\n\n👉 **Predict first:** what goes in the top-left box of the result?",
            ),
            code(
                """
A = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
B = torch.tensor([[5.0, 6.0], [7.0, 8.0]])
print(A @ B)
"""
            ),
            saw(
                "The top-left box is `19`: row 1 of `A` `[1, 2]` dotted with column 1 of `B` `[5, 7]`, which is `1×5 + 2×7 = 19`. "
                "Every other box is another row-column dot product."
            ),
            q(
                "What does `.T` (transpose) do?",
                "**Transpose** flips a matrix so its rows become columns. You need it when you want to dot rows with rows. Flip the second matrix, and its rows become columns.\n\n"
                "👉 **Predict first:** what shape does a `2 × 3` matrix have after `.T`?",
            ),
            code(
                """
M = torch.tensor([[1, 2, 3], [4, 5, 6]])
print(M.shape)
print(M.T)
print(M.T.shape)
"""
            ),
            saw(
                "`(2, 3)` becomes `(3, 2)`. The first row `[1, 2, 3]` became the first column."
            ),
            md(
                "## 5. Full example: from movies to characters, `Q @ K.T`\n\n"
                "Now swap movies for characters. In `model.py`, the model gives every character two vectors of 32 numbers each: "
                'a **query** ("what am I looking for?", like Alex\'s tastes) and a **key** ("what do I offer?", like a movie\'s traits). '
                "The dot product of one character's query with another's key says how well those two characters match.\n\n"
                "The model wants that score for every pair of characters, so it writes `scores = Q @ K.T`. That's all the dot products at once, "
                "one row per query and one column per key.\n\n"
                "First, 3 characters with made-up 2-number vectors, so you can see the shape of the result."
            ),
            code(
                """
# one row per character; made-up numbers to show the idea
Q = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])  # queries for 3 characters
K = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])  # keys for the same 3 characters

scores = Q @ K.T
print(scores)
print(scores.shape)
"""
            ),
            saw(
                'A `3 × 3` table. The box in row 1, column 3 says how well character 1 "matches" character 3.'
            ),
            q(
                "What do the real scores look like for `ἄνδρα`?",
                "Here are the trained model's own numbers: the 5 characters of `ἄνδρα`, each with a real 32-number query and key. "
                "Green boxes are good matches (positive), pink-red boxes clash (negative), and dark is near zero.\n\n"
                "👉 **Predict first:** is the table 5 × 5? Is it symmetrical (the same flipped over the diagonal)?",
            ),
            code(
                """
import matplotlib.pyplot as plt
import numpy as np
import torch

from homer_ai.lessons import load_homer_model
from homer_ai.studio_style import (
    GOOD_BAD,
    INK,
    MUTED,
    studio_figure,
    text_on,
)

tok, model = load_homer_model()  # yours if you trained one, otherwise models/
if model is None:
    print("no trained model found")
else:

    ids = torch.tensor(tok.encode("ἄνδρα"))
    with torch.no_grad():
        x = model.embedding(ids)
        Q, K = model.query(x), model.key(x)
        scores = Q @ K.T
    print("Q", tuple(Q.shape), " K", tuple(K.shape), " scores", tuple(scores.shape))

    letters = list("ἄνδρα")
    grid = scores.numpy()
    limit = float(np.abs(grid).max())
    fig = studio_figure(
        "Attention scores", "How well each character matches", size=(8, 6.5)
    )
    ax = fig.add_axes((0.14, 0.08, 0.66, 0.68))
    ax.set_facecolor("none")
    ax.imshow(grid, cmap=GOOD_BAD, vmin=-limit, vmax=limit)
    for i in range(5):
        for j in range(5):
            ax.text(
                j, i, f"{grid[i, j]:+.1f}", ha="center", va="center",
                color=text_on(grid[i, j], limit), fontsize=12,
                family="monospace", fontweight="bold",
            )
    ax.set_xticks(range(5), letters, fontsize=16, color=INK)
    ax.set_yticks(range(5), letters, fontsize=16, color=INK)
    ax.xaxis.tick_top()
    ax.tick_params(length=0)
    for side in ax.spines.values():
        side.set_visible(False)
    ax.set_xlabel("key: what the character offers", color=MUTED)
    ax.xaxis.set_label_position("top")
    ax.set_ylabel("query: what it looks for", color=MUTED)
    plt.show()
    print("symmetrical?", bool(np.allclose(grid, grid.T)))
"""
            ),
            saw(
                "A `5 × 5` heatmap: row `i`, column `j` is the dot product of character `i`'s query with character `j`'s key. "
                "It isn't symmetrical, because a query and a key are two different vectors, even for the same character. "
                "Chapter 3 turns this table into attention."
            ),
            *questions(
                [
                    (
                        "Why is it called a *dot* product?",
                        "Because in maths it's written with a dot between the two vectors: `a · b`.",
                    ),
                    (
                        "Do the two vectors have to be the same length?",
                        "Yes. Every number needs a partner to multiply with. PyTorch stops with an error if the lengths differ.",
                    ),
                ]
            ),
            recap(
                [
                    "A dot product multiplies pairs and adds them up. Matching signs boost the score, clashing signs lower it, zeros cancel.",
                    "Arrows pointing the same way score high. Dividing by the lengths gives cosine similarity, which only compares direction.",
                    "`Q @ K.T` in `model.py` is every character's query dotted with every character's key, all at once. That's the heart of attention.",
                ],
                "[A_04 PyTorch in this repo](A_04_pytorch_in_this_repo.ipynb).",
            ),
        ],
    )

    # ---------------------------------------------------------------- A_04
    save(
        "A_toolbox/A_04_pytorch_in_this_repo.ipynb",
        [
            toolbox_title("A_04 PyTorch in this repo"),
            where(
                "model learns", ["src/homer_ai/model.py", "src/homer_ai/train_homer.py"]
            ),
            learn(
                [
                    "How `model.py` builds the model from PyTorch parts",
                    "What each part does: the embedding table, linear layers, softmax",
                    "How the loss is measured, and how the weights get adjusted",
                    "How many weights our model has, and why",
                ]
            ),
            words(
                [
                    (
                        "weight",
                        "one adjustable number inside the model; learning means changing weights",
                        "our model has 11,840",
                    ),
                    (
                        "layer",
                        "one part of the model that turns numbers in into numbers out",
                        "`nn.Linear`",
                    ),
                    (
                        "embedding table",
                        "a lookup table: one row of numbers for each token",
                        "137 rows × 32 columns",
                    ),
                    (
                        "softmax",
                        "turns any scores into chances that add up to 1",
                        "`[2, 1] → [0.73, 0.27]`",
                    ),
                    (
                        "loss",
                        "a mistake score: how wrong the guesses were; lower is better",
                        "`F.cross_entropy`",
                    ),
                    (
                        "gradient",
                        "for each weight, which way is downhill: how to change it to lower the loss",
                        "worked out by `loss.backward()`",
                    ),
                    (
                        "optimizer",
                        "the part that adjusts the weights using the gradients",
                        "`AdamW`",
                    ),
                ]
            ),
            *toolbox_background(
                "At the stages **model reads** and **model learns**: `model.py` is the model, and `train()` in `train_homer.py` is the learning.",
                "`model.py` and the training loop are short, but every line is a PyTorch tool. This notebook explains them one at a time, with tiny examples.",
            ),
            step_intro(),
            q(
                "What's `nn.Module`?",
                "`nn` is PyTorch's toolbox for building models (`nn` stands for neural network). `nn.Module` is the blueprint every model starts from. "
                'Our `HomerGreekLLM` class says `class HomerGreekLLM(nn.Module)`, which means "a model, built on PyTorch\'s blueprint".\n\n'
                "The blueprint keeps track of all the model's weights, so the optimizer can find and adjust them.\n\n👉 **Predict first:** how many weights does our model have?",
            ),
            code(
                """
import torch
import torch.nn.functional as F
from torch import nn

from homer_ai.model import HomerGreekLLM

torch.manual_seed(42)
model = HomerGreekLLM(vocab_size=137, embed_dim=32)
for name, w in model.named_parameters():
    print(name, tuple(w.shape))
print("total weights:", sum(w.numel() for w in model.parameters()))
"""
            ),
            saw(
                '5 groups of weights, **11,840** in total. `numel` means "number of elements", so it says how many numbers a tensor holds. '
                "(`F` is PyTorch's collection of ready-made maths functions. We use it below.)"
            ),
            q(
                "What's `nn.Embedding`, and why is its table 137 × 32?",
                "An **embedding table** is a lookup table with one row of numbers for each token in the vocabulary. Give it a token ID, and it hands back that row.\n\n"
                "Its shape is rows = vocabulary size, columns = `embed_dim`. We have 137 tokens (135 characters plus 2 special tokens) and `embed_dim=32`, so the table is 137 × 32. "
                "Each token gets 32 numbers that the model learns, so that tokens used in similar ways end up with similar rows.\n\n"
                "👉 **Predict first:** what shape comes out for 3 token IDs?",
            ),
            code(
                """
table = nn.Embedding(137, 32)
ids = torch.tensor([41, 40, 36])  # π ο λ
rows = table(ids)
print(table.weight.shape)
print(rows.shape)
"""
            ),
            saw(
                "The table is `(137, 32)`, and 3 IDs give back 3 rows of 32 numbers: `(3, 32)`."
            ),
            q(
                "What does a real row of the table look like?",
                "In our trained model, each character's row holds 32 learned numbers. Here are the first 4 of two rows (the `…` stands for the other 28):\n\n"
                "```\nID 58  'ἄ'  →  [-0.15, 0.21, -0.58, -0.27] …\nID 38  'ν'  →  [0.21, -0.65, 0.28, 0.13] …\n```\n\n"
                'No single number means anything on its own. Together, the 32 numbers are how the model "sees" the character.\n\n'
                "👉 **Predict first:** will `ἄ` and `ν` have similar rows? (Nothing told the model they're related.)",
            ),
            code(
                """from homer_ai.lessons import load_homer_model

tok, saved_model = load_homer_model()  # yours if you trained one, otherwise models/
if saved_model is not None:
    learned = saved_model.embedding.weight.detach()
    for ch in ["ἄ", "ν"]:
        i = tok.encode(ch)[0]
        first4 = [round(v, 2) for v in learned[i][:4].tolist()]
        print(f"ID {i}  {ch!r}  →  {first4} …")
else:
    print("no trained model found")"""
            ),
            saw(
                "Two rows of learned numbers, and they look nothing alike. "
                "`saved_model.embedding.weight` is the whole table, one row per token. In the saved file the same numbers sit under the name `embedding.weight`, inside `model_state` ([05_06](../05_training_on_the_odyssey/05_06_saving_and_loading.ipynb))."
            ),
            q(
                "Can I see the rows of a real embedding table?",
                "Yes, as a heatmap with one row per character and one column for each of the 32 numbers. "
                "Green is a positive number, pink-red a negative one, and dark is near zero.\n\n"
                "👉 **Predict first:** will two different characters have the same colour pattern?",
            ),
            code(
                """import matplotlib.pyplot as plt

from homer_ai.studio_style import GOOD_BAD, INK, MUTED, studio_figure

if saved_model is not None:
    chars = list("ἄνδρα μοι")
    ids = [tok.encode(ch)[0] for ch in chars]
    grid = learned[ids].detach().numpy()
    limit = abs(grid).max()

    fig = studio_figure(
        "Embedding rows", f"{len(chars)} characters, 32 learned numbers each"
    )
    ax = fig.add_axes((0.13, 0.1, 0.8, 0.68))
    ax.set_facecolor("none")
    ax.imshow(grid, cmap=GOOD_BAD, vmin=-limit, vmax=limit, aspect="auto")
    ax.set_yticks(range(len(chars)), [repr(ch) for ch in chars], fontsize=13, color=INK)
    ax.set_xticks(range(0, 32, 4), [str(i) for i in range(0, 32, 4)])
    ax.tick_params(length=0)
    for side in ax.spines.values():
        side.set_visible(False)
    ax.set_xlabel("the 32 numbers in the row", color=MUTED)
    plt.show()
else:
    print("no trained model found")"""
            ),
            saw(
                "Every character has its own pattern of colours, and no single square means anything alone. "
                "Don't try to read much into a single row. In a model this small the likeness between rows is faint, and [02_05](../02_text_to_numbers/02_05_token_embeddings.ipynb) measures it properly."
            ),
            q(
                "What's `nn.Linear`?",
                "A **linear layer** turns a vector of one size into a vector of another size, using a table of weights (it's matrix multiplication, from A_03). "
                "Our model's last layer, `head`, turns 32 numbers into 137 scores, one score per possible next token.\n\n"
                "👉 **Predict first:** what shape comes out?",
            ),
            code(
                """
head = nn.Linear(32, 137, bias=False)
scores = head(rows)
print(scores.shape)
"""
            ),
            saw(
                "`(3, 137)`: for each of the 3 positions, a score for every token in the vocabulary. These raw scores are called **logits**."
            ),
            q(
                "What does `F.softmax` do?",
                "**Softmax** turns any list of scores into chances between 0 and 1 that add up to 1. Bigger score, bigger chance.\n\n"
                "```\nscores    ι  2.0     ν  1.0     ς  0.1\n             ↓ softmax\nchances   ι  ?       ν  ?       ς  ?        (together: 1)\n```\n\n"
                "👉 **Predict first:** which of the three gets the biggest chance?",
            ),
            code(
                """
s = torch.tensor([2.0, 1.0, 0.1])
chances = F.softmax(s, dim=-1)
print(chances)
print(chances.sum())
"""
            ),
            saw(
                'About `[0.66, 0.24, 0.10]`, adding up to 1. `dim=-1` means "along the last direction", here the row. Our model uses softmax inside attention.'
            ),
            q(
                "Can I see what softmax does?",
                "Here are two bar charts side by side: the raw scores on the left, the chances softmax makes from them on the right.\n\n"
                "👉 **Predict first:** will the order of the bars change?",
            ),
            code(
                """import matplotlib.pyplot as plt

from homer_ai.studio_style import ACCENT, INK, MUTED, studio_axes, studio_figure

letters = ["ι", "ν", "ς"]
fig = studio_figure("Softmax", "Scores in, chances out", size=(9, 5))
panels = [
    ("scores", s.tolist(), (0.08, 0.14, 0.38, 0.62), "{:.1f}"),
    ("chances", chances.tolist(), (0.56, 0.14, 0.38, 0.62), "{:.2f}"),
]
for label, values, rect, fmt in panels:
    ax = studio_axes(fig, rect)
    ax.set_axisbelow(True)  # grid lines behind the bars
    bars = ax.bar(letters, values, color=MUTED)
    bars[0].set_color(ACCENT)
    for bar, value in zip(bars, values, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2, value + max(values) * 0.02,
            fmt.format(value), ha="center", color=INK, family="monospace",
        )
    ax.set_ylim(0, max(values) * 1.15)
    ax.tick_params(axis="x", labelsize=14, labelcolor=INK)
    ax.grid(axis="x", visible=False)
    ax.set_title(label, color=ACCENT, fontsize=13, fontweight="bold")
plt.show()"""
            ),
            saw(
                "The order stays the same, so the biggest score gets the biggest chance. "
                "But the chances are squeezed into 0 to 1 and add up to 1, so they can be read as probabilities."
            ),
            q(
                "What does `F.cross_entropy` measure?",
                "It's the **loss**: how surprised the model was by the right answer. You give it the model's scores and the correct next token. "
                "If the model gave the right token a high chance, the loss is low.\n\n👉 **Predict first:** which case has the lower loss?",
            ),
            code(
                """
right_answer = torch.tensor([0])
confident_and_right = torch.tensor([[5.0, 0.0, 0.0]])
confident_and_wrong = torch.tensor([[0.0, 5.0, 0.0]])
print(F.cross_entropy(confident_and_right, right_answer).item())
print(F.cross_entropy(confident_and_wrong, right_answer).item())
"""
            ),
            saw(
                "About `0.01` when confident and right, about `5.01` when confident and wrong. Training pushes this number down."
            ),
            q(
                "What do `backward`, `zero_grad` and `step` do?",
                "They're the three lines at the heart of every training loop:\n\n"
                "1. `loss.backward()` works out the **gradient** for every weight, which is the way to change it to lower the loss. Working it out backwards through the model is called backpropagation.\n"
                "2. `optimizer.step()`: the optimizer nudges every weight a small step downhill. Ours is AdamW, a popular one.\n"
                "3. `optimizer.zero_grad()`: wipes the old gradients before the next round. Without it they pile up, like not clearing a calculator.\n\n"
                "👉 **Predict first:** after 50 rounds, will the loss be lower than at the start?",
            ),
            code(
                """
torch.manual_seed(42)
tiny = nn.Linear(3, 3, bias=False)
optimizer = torch.optim.AdamW(tiny.parameters(), lr=0.1)
x = torch.tensor([[1.0, 0.0, 0.0]])
target = torch.tensor([2])  # the right answer is token 2

for step in range(50):
    loss = F.cross_entropy(tiny(x), target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if step in (0, 49):
        print(f"round {step + 1}: loss {loss.item():.3f}")
"""
            ),
            saw(
                "The loss falls a lot in 50 rounds. And that's all learning is: the same three lines, repeated."
            ),
            md(
                "## 5. Full example: one training step on real Greek\n\nOur real model, one window of Book 1, one step: the same lines as `train()` in `train_homer.py`."
            ),
            code(
                """
from homer_ai.train_homer import BOOKS_DIR, load_data

data, tokenizer, text = load_data(BOOKS_DIR)
torch.manual_seed(42)
model = HomerGreekLLM(vocab_size=tokenizer.vocab_size, embed_dim=32)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)

inputs = data[0:24]  # 24 characters
targets = data[1:25]  # the same, shifted by one: the next character each time
for step in range(3):
    logits = model(inputs)
    loss = F.cross_entropy(logits, targets)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(f"step {step + 1}: loss {loss.item():.3f}")
"""
            ),
            saw(
                "The loss goes down a little with each step on this one window. The real training does this for every window of the book, 800 times over. "
                'The targets are the inputs shifted by one, because the job is always "guess the next character".'
            ),
            *questions([]),
            recap(
                [
                    "`nn.Module` is the blueprint. It keeps track of the model's 11,840 weights.",
                    "`nn.Embedding` is a lookup table (137 × 32), `nn.Linear` turns vectors into scores, and softmax turns scores into chances.",
                    "`F.cross_entropy` is the loss. `backward` finds the gradients, `step` adjusts the weights, and `zero_grad` clears the gradients.",
                ],
                "[A_05 tiktoken and BPE](A_05_tiktoken_bpe.ipynb).",
            ),
        ],
    )

    # ---------------------------------------------------------------- A_05
    save(
        "A_toolbox/A_05_tiktoken_bpe.ipynb",
        [
            toolbox_title("A_05 tiktoken and BPE"),
            where("numbers", ["pyproject.toml"]),
            learn(
                [
                    "What tiktoken is",
                    "How to use GPT-2's tokenizer on English and Greek",
                    "Why GPT-2's tokenizer is a poor fit for Homer's Greek",
                    "Why that's a reason to train our own BPE",
                ]
            ),
            words(
                [
                    (
                        "tiktoken",
                        "OpenAI's fast BPE tokenizer library",
                        "`import tiktoken`",
                    ),
                    (
                        "encoding",
                        "in tiktoken: one finished vocabulary with its merges",
                        '`"gpt2"`',
                    ),
                    (
                        "byte",
                        "the small basic unit computers store text in; one Greek letter takes 2 or 3 bytes",
                        "`α` = 2 bytes",
                    ),
                    (
                        "GPT-2",
                        "an older language model by OpenAI; its tokenizer is the book's example",
                        "50,257 tokens",
                    ),
                ]
            ),
            *toolbox_background(
                "At the stage **numbers**, right after [02_03 BPE](../02_text_to_numbers/02_03_bpe.ipynb).",
                "In 02_03 you built a small BPE yourself. Real projects often use a ready-made one instead. Here we try GPT-2's, and see whether it suits Homer.",
            ),
            q(
                "What's tiktoken?",
                "**tiktoken** is a library made by OpenAI (the company behind ChatGPT) that runs BPE tokenizers very fast. "
                "It comes with finished vocabularies, called encodings, including the one GPT-2 used. It's on our shopping list in `pyproject.toml`.",
            ),
            step_intro(),
            q(
                "How do I load GPT-2's tokenizer?",
                '`tiktoken.get_encoding("gpt2")`. The first time, it downloads a file from the internet and saves a copy (it\'s **cached**), so later runs work offline.\n\n'
                "👉 **Predict first:** how many tokens does GPT-2's vocabulary have?",
            ),
            code(
                """
import tiktoken

enc = tiktoken.get_encoding("gpt2")
print(enc.n_vocab)
"""
            ),
            saw("`50257`. Compare our character vocabulary: 137."),
            q(
                "How many tokens does an English line take?",
                "`enc.encode` turns text into token IDs, and `enc.decode` turns them back.\n\n👉 **Predict first:** more or fewer tokens than characters?",
            ),
            code(
                """
english = "Sing, O goddess, the anger of Achilles"
ids = enc.encode(english)
print(len(english), "characters ->", len(ids), "tokens")
print([enc.decode([i]) for i in ids])
"""
            ),
            saw(
                "38 characters → **9 tokens**. Every word is a single token here, even ` Achilles` (with the space in front of it)."
            ),
            q(
                "And Homer's Greek?",
                "👉 **Predict first:** the opening of the Iliad has 15 characters. More or fewer tokens?",
            ),
            code(
                """
greek = "μῆνιν ἄειδε θεὰ"
ids = enc.encode(greek)
print(len(greek), "characters ->", len(ids), "tokens")
print(enc.decode(ids) == greek)
"""
            ),
            saw(
                "15 characters → **20 tokens**. That's more tokens than characters! The round trip still works (`True`), but it's very wasteful."
            ),
            q(
                "Why does Greek take more tokens than letters?",
                "GPT-2's BPE learned its merges from mostly English web text. It almost never saw Homer's Greek, so it learned no Greek pieces to merge.\n\n"
                "And it gets worse. When it has no merge, it falls back to **bytes**. A plain Greek letter is 2 bytes, and an accented one like `ῆ` is 3. "
                "So one Greek letter can become 3 tokens.\n\n👉 **Predict first:** how many tokens is `ῆ`?",
            ),
            code(
                """
for letter in ["α", "ῆ"]:
    ids = enc.encode(letter)
    print(letter, len(letter.encode("utf-8")), "bytes ->", len(ids), "tokens", ids)
"""
            ),
            saw(
                '`α` is 2 bytes but 1 token. `ῆ` is 3 bytes and **3 tokens**. `.encode("utf-8")` shows how a computer stores the letter as bytes.'
            ),
            md(
                "## 5. Full example: GPT-2's tokenizer on Book 1\n\nThe whole of Odyssey Book 1, counted three ways."
            ),
            code(
                """
from homer_ai.train_homer import BOOKS_DIR, load_books

text = load_books(BOOKS_DIR)[0]
print(f"characters (our tokenizer): {len(text):,}")
print(f"GPT-2 tokens:               {len(enc.encode(text)):,}")
"""
            ),
            saw(
                "GPT-2's tokenizer needs **more** tokens than our simple one-character-per-token tokenizer. "
                "So for Homer, a ready-made English tokenizer makes things worse. That's the reason to train our own BPE on Greek, as in [02_03](../02_text_to_numbers/02_03_bpe.ipynb)."
            ),
            *questions(
                [
                    (
                        "So is BPE bad for Greek?",
                        "No, BPE is fine. GPT-2's *merges* are the problem, because they were learned from English. A BPE trained on Homer would learn Greek pieces like `πολλ` and common endings.",
                    ),
                    (
                        "Would GPT-2's tokenizer work for the English translation?",
                        "Yes, very well, as the 9-token example shows. [07_05](../07_advanced_iliad_and_combining/07_05_english_and_greek.ipynb) counts it on Butler's English.",
                    ),
                ]
            ),
            recap(
                [
                    "tiktoken runs GPT-2's BPE: English 38 characters → 9 tokens, but Greek 15 characters → 20 tokens.",
                    "GPT-2 learned its merges from English, so Greek falls back to bytes, up to 3 tokens per letter.",
                    "For Homer's Greek, we should train our own BPE.",
                ],
                "[A_06 The other tools](A_06_other_tools.ipynb).",
            ),
        ],
    )

    # ---------------------------------------------------------------- A_06
    save(
        "A_toolbox/A_06_other_tools.ipynb",
        [
            toolbox_title("A_06 The other tools: uv, Ruff, Jupyter, git and MCP"),
            where(
                "MCP tool",
                [
                    "pyproject.toml",
                    "src/homer_ai/homer_mcp_server.py",
                    ".vscode/settings.json",
                ],
            ),
            learn(
                [
                    "How uv installs and runs the project",
                    "How Ruff checks your code, and how to silence one warning",
                    "How Jupyter notebooks and kernels work",
                    "What git commits and branches are",
                    "How MCP lets Claude use our model",
                ]
            ),
            words(
                [
                    (
                        "terminal",
                        "a window where you type commands instead of clicking; in VS Code: Terminal → New Terminal",
                        "`uv sync`",
                    ),
                    (
                        "command",
                        "an instruction typed into the terminal",
                        "`uv run python …`",
                    ),
                    (
                        "linter",
                        "a tool that reads code and points out mistakes without running it",
                        "Ruff",
                    ),
                    (
                        "commit",
                        "a saved snapshot of the project in git",
                        "a save point in a game",
                    ),
                    (
                        "branch",
                        "a separate line of work in git",
                        "`chapter2-tokenizer`",
                    ),
                ]
            ),
            *toolbox_background(
                "Around every stage: uv and Ruff while you work, Jupyter for the course, git to save your progress, and MCP at the last stage.",
                "These tools aren't about AI, but every AI engineer uses tools like them every day. Knowing them stops small problems from blocking you.",
            ),
            md(
                "## 4. Step by step\n\nThese tools run in the terminal, so this notebook shows the commands rather than running them."
            ),
            q(
                "What does uv do?",
                "**uv** installs Python and every library the project needs, and runs your code with them. The three commands you'll use:\n\n"
                "```\nuv sync                 # install everything on the shopping list\nuv run python file.py   # run a file with the project's Python\nuv add numpy            # add a new library to the list\n```\n\n"
                "`pyproject.toml` is the shopping list, and `uv.lock` records the exact versions, so everyone gets the same ones.",
            ),
            q(
                "What does Ruff do?",
                "**Ruff** is a linter. It reads your code and points out mistakes without running it, like a spell-checker for code. It also formats code (fixes spacing and layout). "
                "In this project VS Code runs it every time you save.\n\n```\nuv run ruff check src/      # look for mistakes\nuv run ruff format src/     # tidy the layout\n```\n\n"
                'Each problem has a code, like `F821` ("undefined name": you used a name that doesn\'t exist) or `E501` ("line too long").',
            ),
            q(
                "How do I silence one Ruff warning?",
                "Add `# noqa: E501` at the end of that line, with the rule's code. Always **name the rule**. A bare `# noqa` silences *every* rule on the line, "
                "so a real mistake there would be hidden too. Silence the smallest thing possible.",
            ),
            q(
                "How do notebooks run Python, and why do we need `ipykernel`?",
                "A notebook doesn't run Python itself. It sends each cell to the **kernel**, a Python program running in the background, and shows the result. "
                "`ipykernel` is the helper that turns Python into a kernel. It's on the shopping list (`pyproject.toml`), so `uv sync` installs it for every new developer. "
                "Choose the kernel in `.venv` (top right: Select Kernel).",
            ),
            q(
                "Should I run one cell or run all?",
                "Shift+Enter (one cell) while you're learning, so you can predict each result. Run All after reopening a notebook, because the kernel forgets everything when it restarts.",
            ),
            q(
                "My notebook opens as JSON. Why?",
                "A notebook is a **JSON** file underneath (plain text that stores data as names and values), and VS Code opened it with the text editor. "
                "Right-click it → Open With… → Jupyter Notebook.",
            ),
            q(
                "What are git commits and branches?",
                "**git** saves snapshots of the project. A commit is one snapshot, like a save point in a game, and you can always go back to it. "
                "A branch is a separate line of work, so the main version stays safe while you try things.\n\n"
                '```\ngit status                  # what has changed?\ngit add file.py             # choose what goes in the next snapshot\ngit commit -m "message"     # save the snapshot\ngit switch -c my-branch     # start a new branch\n```',
            ),
            q(
                "What's MCP, and how does our server use it?",
                "**MCP** (Model Context Protocol) is a standard way for AI apps like Claude to call outside tools. "
                "`homer_mcp_server.py` loads our trained model and offers it as a tool, `consult_homer_llm`. Give it some Greek, and it writes a continuation.\n\n"
                "A line starting with `@` above a function, like `@mcp.tool()`, is a **decorator**. It wraps the function. Here it registers the function as a tool that Claude can see and call.",
            ),
            md(
                '## 5. Full example: a working day\n\n```\nuv sync                                      # 1. set up (once)\nuv run homer-train --model gpt               # 2. train the GPT (about 3 minutes)\nuv run ruff check src/                       # 3. check the code\ngit add src/ && git commit -m "Try lr 0.003"  # 4. save a snapshot\n```\n\n'
                "Then any MCP app, like Claude, can use the trained model through `homer_mcp_server.py`."
            ),
            *questions([]),
            recap(
                [
                    "uv installs and runs the project. Ruff checks it (name the rule when you silence one).",
                    "Notebooks send cells to a kernel, and git commits are save points.",
                    "MCP lets Claude call our model as a tool.",
                ],
                "[A_07: One shared model file](A_07_organising_code.ipynb): why the model lives in `model.py`. Or back to the course: [the course map](../00_start_here/00_00_course_map.ipynb).",
            ),
        ],
    )

    # =====================================================================
    # A_toolbox/A_07_organising_code.ipynb
    # =====================================================================
    save(
        "A_toolbox/A_07_organising_code.ipynb",
        [
            toolbox_title("A_07 One shared model file"),
            where(
                "model learns",
                [
                    "src/homer_ai/train_homer.py",
                    "src/homer_ai/model.py",
                    "src/homer_ai/homer_mcp_server.py",
                ],
            ),
            learn(
                [
                    "Why one copy of the model lives in `model.py`, and both other files import it",
                    "Which of our files use which",
                    "What goes wrong with two copies of a model: a saved file that won't load",
                ]
            ),
            words([]),
            *toolbox_background(),
            q(
                "Why is the model in `model.py`, and not inside each file that uses it?",
                "Two jobs need the model class, `HomerGreekLLM`: training a new model (`train_homer.py`), and loading a saved one back so the MCP server can use it. The quick way is to copy the class into both files. We keep one copy and import it.\n\n"
                "```\n"
                "THE TRAP: two copies                WHAT WE DO: one blueprint\n\n"
                "train_homer.py                      model.py\n"
                "   class HomerGreekLLM  (copy 1)       class HomerGreekLLM  (the only one)\n"
                "   ...training...                        ▲            ▲\n"
                "                                  import │            │ import\n"
                "homer_mcp_server.py                train_homer.py   checkpoint.py  ◄── homer_mcp_server.py\n"
                "   class HomerGreekLLM  (copy 2)                    (loads a saved model)\n"
                "```\n\n"
                "The danger is specific to AI work. A saved model is just numbers, and they only make sense for the exact shape that saved them. "
                "Change one copy and forget the other, and the saved model **won't load**. "
                "So the class lives in `model.py` and nowhere else. The training script imports it, and so does `checkpoint.py`, the one place that turns a saved file back into a model. The server asks `checkpoint.py` and never names the class itself. One blueprint, used everywhere.\n\n"
                "*Coming from C#, this is the first rule I'd set in any codebase: one definition, imported everywhere, never copied.*",
            ),
            step_intro(),
            q(
                "Which of our files use which?",
                "This code reads our own source files and prints the lines that start with `from homer_ai`. It's a live map of who uses whom. "
                "(We read the files as text, so there's no need to import the MCP server.)\n\n"
                "👉 **Predict first:** which files import `model.py`?",
            ),
            code(
                """from pathlib import Path

import homer_ai

folder = Path(homer_ai.__file__).parent
for file in sorted(folder.glob("*.py")):
    lines = file.read_text(encoding="utf-8").splitlines()
    imports = [line for line in lines if line.startswith("from homer_ai")]
    print(file.name, "imports:", imports if imports else "nothing from homer_ai")"""
            ),
            saw(
                "Three files import `model.py`: `train_homer.py`, `checkpoint.py` and `lessons.py` (the notebooks' helpers). `homer_mcp_server.py` isn't one of them. "
                "It imports `checkpoint.py`, and `checkpoint.py` imports the two model files, `model.py` and `gpt.py`. "
                "At the top, `model.py`, `gpt.py` and `tokenizer.py` import nothing from the project. The main lines, as a picture:\n\n"
                "```\n"
                "model.py    gpt.py    tokenizer.py       import nothing from our project\n"
                "    ▲          ▲           ▲\n"
                "    └──────────┼───────────┘\n"
                "         checkpoint.py                   saves a model, and builds one from a saved file\n"
                "           ▲         ▲\n"
                "           │         │\n"
                "train_homer.py     homer_mcp_server.py\n"
                "```\n\n"
                "An arrow points from the file that imports to the file it uses. (`train_homer.py` also imports the three top files directly, and the server imports `tokenizer.py` and `generation.py`.)"
            ),
            q(
                "What really goes wrong with two copies?",
                "Say you change `embed_dim` (the size of each character's list of numbers) in one copy and forget the other. "
                "The saved weights then don't fit the other model.\n\n"
                "👉 **Predict first:** a model with `embed_dim=32` saves its weights. Can a model with `embed_dim=16` load them?",
            ),
            code(
                """from homer_ai.model import HomerGreekLLM

saved = HomerGreekLLM(vocab_size=10, embed_dim=32).state_dict()  # like homer_model.pt
forgotten_copy = HomerGreekLLM(vocab_size=10, embed_dim=16)  # a copy nobody updated

try:
    forgotten_copy.load_state_dict(saved)
except RuntimeError as error:
    print("Could not load:", str(error).splitlines()[1].strip())"""
            ),
            saw(
                "You get an error about a size mismatch. The saved weights are shaped for 32, the copy for 16. "
                "With one `model.py`, there's only one place to change, so this can't happen."
            ),
            recap(
                [
                    "Put the model in one file and import it everywhere, so copies can't drift apart.",
                    "A saved model only fits the exact shape that saved it, so two copies that drift apart break loading.",
                ],
                "back to the course: [the course map](../00_start_here/00_00_course_map.ipynb).",
            ),
        ],
    )
