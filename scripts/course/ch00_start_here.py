# ruff: noqa: E501
"""Chapter 0, Start here: the course map, setup, and reading errors."""

from .cells import (
    START_HERE,
    WHAT_WE_BUILD_FULL,
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
from .diagrams import DIAGRAM_JOURNEY

TOOL_CHAIN = [
    (
        'If an LLM only guesses the next token, how can Claude "use a tool" or "take an action"?',
        "It can't, not by itself. The model never runs anything; it only writes text, one **token** at a time "
        "(a token is one piece of text; for our model, one character). The trick is in the training: when a tool would help, "
        "the most likely next tokens are a special, strictly formatted message. Think of it as filling in a form.",
    ),
    (
        'What does that "form" look like?',
        'Something like `{"tool": "consult_homer_llm", "greek_seed": "ἄνδρα"}`. That way of writing data is called **JSON**: '
        'plain text, with names and values in curly brackets. It\'s often wrapped in special tokens that mean "a tool call starts here" '
        'and "it ends here". Same idea as our `<|endoftext|>` marker in [02_02](../02_text_to_numbers/02_02_special_tokens.ipynb).',
    ),
    (
        "Then who actually runs the tool?",
        "The program around the model. People often call it the **harness**: the app that feeds text in and reads text out. "
        "It watches the tokens as they come out, and when it spots a tool-call form, it stops the model, runs the tool "
        "(say, our MCP server's `consult_homer_llm`) and pastes the result back into the conversation as more text.",
    ),
    (
        "And then?",
        "The model just carries on guessing next tokens. Only now the tool's answer is right there in front of it, so its next words can use it:\n\n"
        "```\nguess → guess → [tool form] → app runs the tool → [result text] → guess → guess\n```",
    ),
    (
        "How did the model learn to write the form at the right moment?",
        "Extra training, on lots of example conversations where the right move was a tool call. Extra training like this, on top of a finished model, "
        "is called **fine-tuning** (chapter 8 does a small one, on GPT-2). That's way beyond our toy model, though, which only writes Greek. "
        "So in this repo it's Claude that writes the tool-call form, and **our model is the tool**.",
    ),
    (
        "How does Claude know our tool exists, and when to call it?",
        "Have a look in `src/homer_ai/homer_mcp_server.py`. The line `@mcp.tool()` registers the function `consult_homer_llm` as a tool. "
        "Its **docstring** (the text in triple quotes under the function) is the description Claude reads when it decides whether to call it.",
    ),
]


# =====================================================================
# 00_start_here/00_00_course_map.ipynb
# =====================================================================
COURSE_TABLE = """| Chapter | What you learn | Stage |
|---|---|---|
| `00_start_here` | The course map, setup, and reading error messages | getting ready |
| `01_foundations` | What a language model is; a model made of counting; neural networks; how a program learns; our model as a black box | the basics |
| `02_text_to_numbers` | Turning Greek text into numbers the model can use | numbers |
| `03_attention` | How the model decides which earlier letters matter | model reads |
| `04_building_the_model` | Putting the parts together into a full model | model reads |
| `05_training_on_the_odyssey` | Teaching the model, and checking it really learned | model learns |
| `06_making_it_talk` | Writing Greek, and offering it to other AI apps | model writes, MCP tool |
| `07_advanced_iliad_and_combining` | The Iliad model, comparing models fairly, and combining both | all of them again |
| `08_advanced_fine_tuning` | Fine-tuning a pretrained model: GPT-2's weights in our own class, a spam filter, instructions, LoRA | a borrowed model |
| `09_advanced_grounded_answers` | Answers grounded in the real text: passages, a word search (BM25) from scratch, scoring the search, a second MCP tool | Odyssey text, MCP tool |
| `10_advanced_your_domain` | The same pattern on a company's own text, with invented data: support answers with citations, spotting complaints in email, service-desk triage, and the pitfalls | your own data, MCP tool |
| `A_toolbox` | Appendix: PyTorch, the dot product, tiktoken, and the other tools; dip in any time | all of them |"""


# =====================================================================
# 00_start_here/00_01_setup.ipynb
# =====================================================================
SETUP_CHECK = r"""import importlib
import sys
from pathlib import Path


def check(ok, good, bad):
    print(("✅ " + good) if ok else ("❌ " + bad))


# 1. Python itself
v = sys.version_info
version = f"Python {v.major}.{v.minor}"
check(v >= (3, 12), version, f"{version} is too old: run `uv sync`")

# 2. The libraries the course needs
for name in ["torch", "tiktoken"]:
    try:
        importlib.import_module(name)
        check(True, f"{name} is installed", "")
    except ImportError:
        check(False, "", f"{name} is missing: run `uv sync`, then pick .venv")

# 3. Our own code
try:
    import homer_ai.tokenizer  # noqa: F401

    check(True, "the homer_ai code imports", "")
except ImportError:
    check(False, "", "homer_ai doesn't import: pick the .venv kernel")

# 4. The Greek and English texts
project = Path.cwd()
while not (project / "pyproject.toml").exists() and project != project.parent:
    project = project.parent  # walk up from notebooks/... to the project folder

books = sorted((project / "data" / "homer").glob("*/*/book_*.txt"))
check(
    len(books) == 96,
    f"data/homer/ has all {len(books)} book files",
    f"data/homer/ has only {len(books)} of the 96 book files",
)
book_1 = project / "data" / "homer" / "odyssey" / "greek" / "book_01.txt"
check(
    book_1.exists(),
    "the training text is there (Book 1 of the Odyssey)",
    "data/homer/odyssey/greek/book_01.txt is missing",
)

# 5. The two trained models (yours if you made them, else the ready-made ones)
MODELS = [
    ("the small model", "homer_model.pt", "odyssey_book1.pt"),
    ("the GPT", "homer_gpt.pt", "odyssey_book1_gpt.pt"),
]
for what, own, ready_made in MODELS:
    if (project / "src" / "homer_ai" / own).exists():
        print(f"✅ {what}: your own is there ({own})")
    elif (project / "models" / ready_made).exists():
        print(f"✅ {what}: the ready-made one is there (models/{ready_made})")
    else:
        print(f"❌ models/{ready_made} is missing: `git status` shows what is gone")"""


def build():
    save(
        "00_start_here/00_00_course_map.ipynb",
        [
            md(
                "# 00_00 Course map: build Greek LLMs from Homer\n\n*Our own page: not in the book.*\n\nStart here. This page covers the questions you'll probably have on day one."
            ),
            where(None, []),
            learn(
                [
                    "What we're building, and where it ends up",
                    "The path the Greek text takes through the project",
                    "Which chapter teaches which part, and in what order",
                    "How to use a notebook in VS Code",
                ]
            ),
            words(
                [
                    (
                        "LLM",
                        "large language model: a program that reads text and guesses what comes next",
                        "after `ἄνδρα μοι` it guesses `ἔννεπε`",
                    ),
                    (
                        "notebook",
                        "a file of small boxes called cells; each cell holds code or notes",
                        "this file, `00_00_course_map.ipynb`",
                    ),
                    (
                        "cell",
                        "one box in a notebook; you run it and its result appears underneath",
                        "the box you're reading now",
                    ),
                    (
                        "kernel",
                        "the Python program that runs your notebook's cells in the background",
                        "the one in `.venv`",
                    ),
                    (
                        "MCP",
                        "Model Context Protocol: the standard way for AI apps like Claude to use outside tools",
                        "our `homer_mcp_server.py`",
                    ),
                ]
            ),
            md("## 3. The idea"),
            q(*WHAT_WE_BUILD_FULL),
            q(
                "What happens to one word on the way through?",
                "Let's follow `ἄνδρα` (\"man\", the Odyssey's first word) through our real, trained model. "
                "The **tokenizer** gives each character a number. The **model** turns those numbers into a guess for the next character. "
                "Then we add the best guess to the end and guess again.\n\n"
                "And these are real numbers, from the small model in this repo. You'll meet that model in "
                "[01_04](../01_foundations/01_04_our_model_as_a_black_box.ipynb), and print these very chances yourself in "
                "[06_01](../06_making_it_talk/06_01_writing_greek.ipynb).",
            ),
            md(
                "**Look at:** the four rows, top to bottom. The orange bar is the model's best guess, and the green square is the letter that gets added."
            ),
            code(DIAGRAM_JOURNEY),
            q(
                "Where are we, and what do we have so far?",
                "Right at the start, with nothing built yet. You don't need to know anything about AI. "
                "Every new AI word gets explained before we use it, and every idea comes with a tiny example you can try.",
            ),
            q(
                "What problem does this page solve?",
                "A course with this many notebooks is easy to get lost in. So this page is the map. It shows every chapter, "
                "what it teaches and where it fits, so you always know where you are, and why you're doing what you're doing.",
            ),
            md("### The guessing machine, and how it uses tools"),
            *[q(a, b) for a, b in TOOL_CHAIN],
            q(
                "What is the end goal?",
                "Three models, one after the other:\n\n```\n Odyssey model  →  Iliad model  →  combine both\n (chapters 1-6)    (same steps)    (chapter 7)\n```\n\n"
                "1. A model that learns from the **Odyssey**.\n2. The same steps again for the **Iliad**, Homer's other great poem.\n3. In advanced chapter 7, the two **combined**.\n\nAfter that, advanced chapter 8 borrows a model somebody else trained (GPT-2) and fine-tunes it for a new job, and advanced chapter 9 answers questions about Homer from the real text, with the book and line beside each answer. Advanced chapter 10 leaves Homer and does the same three things (look up, fine-tune, measure) on the kind of text a company has: product sheets, customer emails and service-desk tickets, all invented.",
            ),
            q(
                "Which chapters are there?",
                COURSE_TABLE
                + "\n\nEvery chapter folder also has a `README.md` that lists its notebooks in order.",
            ),
            q(
                "What's the toolbox, and when do I use it?",
                "[Appendix A, the toolbox](../A_toolbox/A_00_overview.ipynb), covers the tools we use: PyTorch, the dot product, tiktoken, uv, Ruff, Jupyter, git and MCP. "
                "It's not something you read first. You dip into it whenever a tool comes up that you don't know yet. New to tensors, say? That's A_02.",
            ),
            q(
                "Do I need the book?",
                "No. The course follows Sebastian Raschka's book *Build a Large Language Model (From Scratch)*, "
                "but every idea is explained in full here in the notebooks. If you do have the book, each notebook tells you which section it matches. "
                "And where a beginner needs more background than the book gives, we've added chapters of our own.",
            ),
            q(
                "What does every notebook look like?",
                "The same parts, in the same order, every time:\n\n1. What you'll learn\n2. Words you'll meet\n3. The idea: background first, then the idea, as questions and answers\n"
                "4. Step by step: one question per step; predict, run, look\n5. A full example on the real Greek text (when the topic has one)\n6. The questions other learners asked, then a recap\n\n(The section numbers just count up, so a notebook with no full example has one number fewer.)",
            ),
            md("## 4. Step by step: how to start"),
            q(
                "How do I get set up?",
                "Go through [00_01: Setup](00_01_setup.ipynb). You install VS Code, git and uv, grab the project, run `uv sync` "
                "and pick the `.venv` kernel. The last cell there checks that it all works.",
            ),
            q(
                "What if I get an error?",
                "You will, everyone does. [00_02: Reading errors](00_02_reading_errors.ipynb) shows you how to read one, using the actual errors we hit while building this project.",
            ),
            q(
                "Where do I start learning?",
                "With Chapter 1, [01_00: Foundations](../01_foundations/01_00_chapter_overview.ipynb). Then just go through the chapters in order. "
                "If you meet a tool or a word you don't know (**tensor** is a likely one), dip into "
                "[Appendix A, the toolbox](../A_toolbox/A_00_overview.ipynb). For tensors, that's [A_02](../A_toolbox/A_02_pytorch_tensors.ipynb).",
            ),
            *questions(
                [
                    (
                        "Who is this course for?",
                        "Developers who already know some language well (it doesn't have to be Python) but have never done machine learning. "
                        "School maths is enough. Matrices, `e` and logarithms get a gentle introduction when they first show up. "
                        "We explain the AI and the PyTorch, not general programming. (New to Python? [A_01](../A_toolbox/A_01_python_essentials.ipynb) is an optional primer.)\n\n"
                        "*I came to this from years of C#, .NET and TypeScript, not Python. So I explain the AI and the PyTorch, not the language.*",
                    ),
                    (
                        "Can I break anything by running the notebooks?",
                        "No, you can't. The notebooks only read the Greek text and run short demos. They never touch the project's code or the trained models.",
                    ),
                ]
            ),
            recap(
                [
                    "We're building LLMs that write Homer's Greek: the Odyssey first, then the Iliad, then both together.",
                    "The text's path: text → numbers → model reads → model learns → model writes → MCP tool.",
                    "Set up with [00_01](00_01_setup.ipynb), then start on Chapter 1. The toolbox is there whenever you need it.",
                ],
                "[00_01: Setup](00_01_setup.ipynb).",
            ),
        ],
    )

    save(
        "00_start_here/00_01_setup.ipynb",
        [
            md(
                "# 00_01 Setup: get the course running on your computer\n\n*Our own page: not in the book.*\n\n"
                "You only do this once. Most of this page is reading, because you can't run a notebook until setup is done. "
                "Then the last cell checks that it all works."
            ),
            where(None, ["pyproject.toml", ".vscode/extensions.json"], here=START_HERE),
            learn(
                [
                    "What a terminal is, and how to open one",
                    "How to install VS Code and uv, and get the project onto your computer",
                    "How to open a notebook and pick the right kernel",
                    "How to check that your setup works",
                ]
            ),
            md("## 3. The idea"),
            q(
                "What is a terminal?",
                "A **terminal** is a window where you type commands instead of clicking buttons. "
                "You type a command, press Enter, and the computer answers in text.\n\n"
                "In VS Code it's the menu **Terminal → New Terminal**. On a Mac without VS Code, open the app called **Terminal**.",
            ),
            q(
                "Which programs do I need?",
                "Three, all free:\n\n"
                "| Program | What it is | Get it from |\n|---|---|---|\n"
                "| **VS Code** | the editor: where you read code and run notebooks | <https://code.visualstudio.com/> |\n"
                "| **git** | the tool that copies the project and keeps its history | <https://git-scm.com/> (Macs often have it already) |\n"
                "| **uv** | the tool that installs Python and every library the project needs | <https://docs.astral.sh/uv/> |\n\n"
                "Not sure if you already have one? Type its name and `--version` in a terminal, like `git --version`, and see if it answers.",
            ),
            step_intro(),
            q(
                "How do I get the project onto my computer?",
                "Copy it with git. In a terminal:\n\n```\ngit clone https://github.com/IcodeNet/homer-ai\ncd homer-ai\n```\n\n"
                "`git clone` gives you your own copy of the project folder, and `cd` (**change directory**) moves the terminal into it.",
            ),
            q(
                "How do I install everything the project needs?",
                "Inside the project folder, run:\n\n```\nuv sync\n```\n\n"
                "uv reads `pyproject.toml`, the project's shopping list, and installs Python, PyTorch, tiktoken and the notebook tools "
                "into a folder called `.venv` (the project's own Python). The clone itself was small. This is the big download: about 900 MB, mostly PyTorch, and only the first time. `.venv` is never committed to git. You do this once, and again only if the list changes.",
            ),
            q(
                "How do I open the project in VS Code?",
                "In VS Code, go to **File → Open Folder…** and pick the `homer-ai` folder. "
                "VS Code will then suggest the extensions this project uses (Python, Pylance, Ruff, Jupyter), because they're listed in "
                "`.vscode/extensions.json`. Say yes. An **extension** is an add-on that teaches VS Code a new trick, like running notebooks.",
            ),
            q(
                'VS Code asks me to "Select Kernel". Which one?',
                "The **kernel** is the Python program that runs your notebook's cells in the background. "
                "Click **Select Kernel** (top right of the notebook) and choose the Python inside `.venv`. "
                "That's the one with PyTorch and our `homer_ai` code installed. VS Code remembers your choice for each notebook, so you only do it once per notebook.",
            ),
            q(
                "My notebook opens as a wall of text. What happened?",
                "A notebook is stored as **JSON**, a plain-text format for data, and VS Code has opened it in the text editor. "
                "Right-click the file, choose **Open With…**, then **Jupyter Notebook**. Usually, double-clicking it in the file list opens it the right way anyway.",
            ),
            q(
                "Shift+Enter or Run All: which do I use?",
                "- **Shift+Enter** runs only the cell you're in, then moves to the next one. Use it while you're learning, so you can predict each result first.\n"
                "- **Run All** (the button at the top) runs every cell, top to bottom. Use it when you reopen a notebook. "
                "The kernel forgets everything when it restarts, and a cell can only use what the cells above it made.",
            ),
            md(
                "## 5. Full example\n\n### ❓ Is my setup working?\n\n"
                "Pick the `.venv` kernel, then run the cell below. Every line should start with ✅. "
                "If one starts with ❌, it tells you what to fix."
            ),
            md(
                "You don't have to read this code. It only checks that things are installed, so just run it."
            ),
            code(SETUP_CHECK),
            saw(
                "One line per check, and two for the models: the small one of chapter 1 and the GPT of chapter 4. If any line says ❌, fix that one thing and run the cell again. "
                "You shouldn't see a red `Failed to initialize NumPy` warning, because NumPy comes with the project. If you do, it's harmless (more in 00_02). "
                "And if you get an error message instead, the next notebook shows you how to read it."
            ),
            *questions(
                [
                    (
                        "Do I need a GPU?",
                        "No. A **GPU** is a graphics chip that does maths very fast, and big models need one. Ours is small enough "
                        "for a laptop's normal processor, the CPU. Training the GPT on Book 1 takes about 3 minutes, and the small model about 8.",
                    ),
                ]
            ),
            recap(
                [
                    "A terminal is where you type commands. `git clone` copies the project and `uv sync` installs it.",
                    "Open the folder in VS Code, accept the extensions and pick the `.venv` kernel.",
                    "The check cell should show ✅ on every line.",
                ],
                "[00_02: Reading errors](00_02_reading_errors.ipynb).",
            ),
        ],
    )

    # =====================================================================
    # 00_start_here/00_02_reading_errors.ipynb
    # =====================================================================
    save(
        "00_start_here/00_02_reading_errors.ipynb",
        [
            md(
                "# 00_02 Reading error messages\n\n*Our own page: not in the book.*\n\n"
                "You already know how to read a traceback in your own language, so we'll skip the basics. This page is only about the errors that come from this course: "
                "notebooks, PyTorch and our own files. We really did hit every one of them while building the project."
            ),
            where(None, [], here=START_HERE),
            learn(
                [
                    "The notebook error you'll meet first: a name that 'is not defined'",
                    "The three PyTorch errors behind most beginner problems: shape, data type and device",
                    "A real traceback through our own `tokenizer.py`",
                ]
            ),
            md("## 3. The idea"),
            q(
                "Why do I need a page on errors if I can already read a traceback?",
                "Because in ML code the error often points at the **wrong place**. "
                "A shape mistake in a model gets reported on the line that *uses* the bad shape, not the line that *made* it. "
                "If you know the three or four usual suspects, you save yourself hours.\n\n"
                "Same rule as always, though: read the last line first, then look above it for a line in your own file.",
            ),
            step_intro(),
            q(
                "I ran a cell and got `NameError: name '…' is not defined`, but the name is right there in an earlier cell. Why?",
                "Because that earlier cell **hasn't run yet**. The notebook's kernel (the Python process behind it) only knows the names made by cells you've actually run, "
                "not the ones you can see on the page. It happens when you run a cell before the ones above it, "
                "or after a kernel restart, which wipes its memory.\n\n"
                "The fix: run the cells above it, in order, or click Run All. That's why the course rule is to run from the top. This one causes most of the errors you'll see.\n\n"
                "👉 **Predict first:** what will the last line of the traceback say?",
            ),
            code(
                """import traceback

try:
    print(epochs)  # nobody ever created `epochs`
except NameError:
    traceback.print_exc()  # print the error, but let the notebook carry on"""
            ),
            saw(
                "The last line is `NameError: name 'epochs' is not defined`. "
                "(The cells on this page catch their own errors, which is why Run All still works here.)"
            ),
            q(
                "ModuleNotFoundError: what does it mean here?",
                "In this course it nearly always means the notebook is on the **wrong kernel**, a Python that doesn't have our libraries (torch, tiktoken…). "
                "The fix: click **Select Kernel** and pick the Python in `.venv`.",
            ),
            q(
                "Shape mismatch: what does it look like?",
                "A **tensor** is PyTorch's grid of numbers, and its **shape** is its size in each direction: `(2, 3)` means 2 rows and 3 columns "
                "(more in [A_02](../A_toolbox/A_02_pytorch_tensors.ipynb)). Matrix multiplication (`@`) needs the two inside numbers to match: `(2, 3) @ (3, 4)` works, `(2, 3) @ (2, 3)` doesn't. "
                "It's the most common PyTorch error, and at least the message prints both shapes.\n\n"
                "👉 **Predict first:** which two shapes will the message show?",
            ),
            code(
                """import torch

try:
    torch.ones(2, 3) @ torch.ones(2, 3)
except RuntimeError:
    traceback.print_exc()"""
            ),
            saw(
                "So: `shapes cannot be multiplied (2x3 and 2x3)`. The thing is, the fix is almost never on this line. "
                "You have to find where the shape went wrong earlier, printing `.shape` along the way."
            ),
            q(
                "Data type mismatch: what does it look like?",
                "Every tensor has a **dtype** (data type). It holds either whole numbers (`int64`, which PyTorch calls **Long**) or decimals (`float32`, called **Float**). "
                "Token IDs, the numbers the tokenizer gives each character, are whole numbers. Weights and scores are decimals. Mix them up and you get an error.\n\n"
                "👉 **Predict first:** an **embedding table** is a lookup table with one row of numbers per token ID (chapter 2 builds one). What happens if we give it decimals instead of IDs?",
            ),
            code(
                """try:
    torch.nn.Embedding(5, 2)(torch.tensor([1.0, 2.0]))  # IDs must be whole numbers
except RuntimeError:
    traceback.print_exc()"""
            ),
            saw(
                "There it is: `Expected tensor … Long, Int; but got … FloatTensor`. The fix is to build the IDs as whole numbers (`torch.tensor([1, 2])`)."
            ),
            q(
                "Device mismatch: what is that?",
                "A **device** is where a tensor lives: the normal CPU, or a GPU (`cuda`, or `mps` on a Mac). "
                "If one tensor is on the GPU and the other on the CPU, you get *Expected all tensors to be on the same device*. "
                "Everything in this course stays on the CPU, so you shouldn't meet it here. You will the day you move to a GPU, though.",
            ),
            q(
                "What if a change makes no error but the results are wrong?",
                "That's the worst kind. Once, in our `train()` function, the lines that update the model slipped one level out of the inner loop, "
                "so the model learned from only one chunk of text per **epoch** (one pass through the training text). Nothing crashed. "
                "We only caught it because the output stopped matching our saved results. So the lesson is: **compare results before and after a change** "
                "([05_04](../05_training_on_the_odyssey/05_04_running_fair_experiments.ipynb) shows how).",
            ),
            q(
                "I see 'Failed to initialize NumPy'. Is something broken?",
                "No, it's only a **warning**. A warning means \"this isn't ideal, but I'm carrying on\", while an error stops the program. "
                "PyTorch is saying it can't find NumPy, another maths library. It's harmless. NumPy gets installed with the project, so running `uv sync` again makes it go away.",
            ),
            md(
                "## 5. Full example\n\n### ❓ Can I read a real traceback from deep inside our code?\n\n"
                "Let's find out. Here we ask our own tokenizer to decode an ID that doesn't exist (the vocabulary only goes up to 136).\n\n"
                "👉 **Predict first:** what kind of error, and which of our files will the traceback point to?"
            ),
            code(
                """from homer_ai.tokenizer import CharTokenizer
from homer_ai.train_homer import BOOKS_DIR, load_books

tokenizer = CharTokenizer("\\n".join(load_books(BOOKS_DIR)))

try:
    tokenizer.decode([41, 9999])
except KeyError:
    traceback.print_exc()"""
            ),
            saw(
                "This time the last line is `KeyError: 9999`. Just above it, the traceback points into **our** file, "
                "`tokenizer.py`, inside `decode`. That's where to look: the dictionary `id_to_char` simply has no entry 9999."
            ),
            recap(
                [
                    "Read the last line first, then find your own file's line. In a notebook, a NameError usually means a cell above hasn't run.",
                    "Shape, dtype and device are the three PyTorch mismatches. Print `.shape` and `.dtype` to track them down.",
                    "A warning isn't an error: the program carries on.",
                ],
                "Chapter 1: [Foundations](../01_foundations/01_00_chapter_overview.ipynb).",
            ),
        ],
    )
