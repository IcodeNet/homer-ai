# ruff: noqa: E501
"""The helpers every notebook is built from, and `save()`, which writes one notebook.

A cell is a small dict (see `md` and `code`). The other helpers build the
repeating parts of a notebook (title, "where we are", "what you will learn",
the word table, learner questions, the recap).
"""

import json
import re
import sys
import textwrap
from pathlib import Path

# Default: the notebooks/ folder next to the scripts/ folder
ROOT = (
    Path(sys.argv[1])
    if len(sys.argv) > 1
    else Path(__file__).parent.parent.parent / "notebooks"
)

STAGES = [
    "Odyssey text",
    "numbers",
    "model reads",
    "model learns",
    "model writes",
    "MCP tool",
]
JOURNEY = " → ".join(STAGES)

# What the "you are here" line says in the chapters that have no stage on the journey.
START_HERE = "**getting ready**: nothing to build yet"
CH1_HERE = "**the basics** (chapter 1): the ideas every stage uses"


# ---------------------------------------------------------------- helpers
def md(s):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": s.strip("\n").splitlines(keepends=True),
    }


def code(s):
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": s.strip("\n").splitlines(keepends=True),
    }


SUMMARIES = {
    "saw": "👀 What you saw (open it after you've run the cell)",
    "hint": "💡 Hint",
    "answer": "✅ Answer",
}


def fold(kind, text):
    """Wrap text in a collapsed <details> block so learners predict before peeking.

    kind is "saw", "hint" or "answer" (see SUMMARIES). The blank lines around the
    text are required, or the Markdown inside would not render.
    """
    return (
        f"<details><summary>{SUMMARIES[kind]}</summary>\n\n{text.strip()}\n\n</details>"
    )


def saw(text):
    """A "What you saw" cell, already folded, so its text can start any way it likes."""
    return md(fold("saw", text))


def fold_reveals(cells):
    """Fold every "What you saw" markdown cell. The one place this is done."""
    for c in cells:
        text = "".join(c["source"])
        if c["cell_type"] == "markdown" and text.startswith("**What you saw"):
            c["source"] = fold("saw", text).splitlines(keepends=True)
    return cells


def _tagged(cell, tag):
    cell["metadata"] = {"tags": [tag]}
    return cell


def exercise(title, task, starter, solution, check, hint, answer=""):
    """A "Your turn" exercise: five cells, to be unpacked into a notebook with `*`.

    1. the task, under the heading `### ❓ Your turn: <title>`
    2. the starter code cell: every blank is `None` with a `# 👉 do: ...` comment
       (and a `# 💡 hint: ...` comment where it helps), so running it as it is
       never fails
    3. the check cell: `check` is one or more calls of `homer_ai.lessons.check`,
       which prints ⏳, ✅ or ❌ with a 🚨 coaching line and never raises
    4. a folded hint
    5. a folded answer: `solution`, the whole starter cell filled in, then
       `answer`, a line or two on why

    The starter, check and answer cells carry the tags `exercise-starter`,
    `exercise-check` and `exercise-solution`. `scripts/run_notebooks.py
    --solutions` uses them to put each solution in place of its starter and
    to require a ✅ from every check.
    """
    starter, solution, check = (
        textwrap.dedent(s).strip("\n") for s in (starter, solution, check)
    )
    if "None" not in starter or "# 👉 do:" not in starter:
        raise ValueError(f"exercise {title!r}: the starter needs a None and a 👉 do")
    if "```" in solution or "= None" in solution:
        raise ValueError(f"exercise {title!r}: the solution still has a blank")
    gap = "\n\n" if check.startswith("def ") else "\n"  # the layout ruff wants
    solved = f"```python\n{solution}\n```"
    if answer:
        solved += f"\n\n{answer.strip()}"
    return [
        md(
            f"### ❓ Your turn: {title}\n\n{task.strip()}\n\n"
            "Replace each `None`, run the cell, then run the check cell under it."
        ),
        _tagged(code(starter), "exercise-starter"),
        _tagged(
            code(f"from homer_ai.lessons import check\n{gap}{check}"),
            "exercise-check",
        ),
        md(fold("hint", hint)),
        _tagged(md(fold("answer", solved)), "exercise-solution"),
    ]


def title(heading, book=None):
    how = (
        "**How to use this notebook:** click a cell and press Shift+Enter to run just that cell.\n"
        "Before each code cell, jot down what you think it will do. Then run it and see if you were right.\n\n"
        "Run the cells in order, from the top. **A cell can use what earlier cells made**, "
        "so if you skip ahead you'll get `NameError: name '…' is not defined`. "
        "If that happens, run the cells above it, or just click Run All."
    )
    # The book is introduced once, in the course map ("Do I need the book?"); here only a short tag.
    book_line = f"*📖 Book: {book}*" if book else "*📖 Not in the book*"
    return md(f"# {heading}\n\n{book_line}\n\n{how}")


# One sentence per notebook: what it solves. It goes into the opening "Where this fits"
# box, so no notebook repeats the big-picture background (that lives in the course map).
SOLVES = {
    "00_start_here/00_01_setup.ipynb": "No notebook will run until your computer has the project, Python and the libraries. This page gets you there, then checks it worked.",
    "00_start_here/00_02_reading_errors.ipynb": "Sooner or later a cell stops with a red error message. Here are the notebook and PyTorch errors you're most likely to hit, and what each one means.",
    "01_foundations/01_00_chapter_overview.ipynb": "Before building anything, we need the four ideas every later chapter uses: what a language model is, a neural network, learning, and what our own model does.",
    "01_foundations/01_01_a_language_model_without_a_brain.ipynb": "What is the simplest program that can guess the next character of Homer? Building it shows what a real model must do better.",
    "01_foundations/01_02_neural_networks.ipynb": "The model in `model.py` is a neural network. What is that, and what are its 11,840 weights?",
    "01_foundations/01_03_how_a_program_learns.ipynb": "A new model's weights are random. How does a program change them, by itself, until its guesses are good?",
    "01_foundations/01_04_our_model_as_a_black_box.ipynb": "Chapters 2-5 build and train `HomerGreekLLM`. Seeing first what it does from outside, untrained and trained, shows what all the parts are for.",
    "02_text_to_numbers/02_00_chapter_overview.ipynb": "A model only does maths, so this chapter turns Book 1 (444 lines, 19,701 characters) into numbers. Those numbers are all the model will ever see, so nothing may be lost or mixed up here.",
    "02_text_to_numbers/02_01_tokens_and_vocabulary.ipynb": "We need a **fixed** way to turn each character into a number and back: if `π` were 41 today and 12 tomorrow, a trained model would read nonsense.",
    "02_text_to_numbers/02_02_special_tokens.ipynb": "Two gaps in the tokenizer: characters it has never seen (like `!`) were silently thrown away, and nothing marks where one book ends and the next begins.",
    "02_text_to_numbers/02_04_sliding_window.ipynb": "Before the model can learn, it needs practice questions with answers. Where do they come from, and how does one piece of text give many of them?",
    "02_text_to_numbers/02_05_token_embeddings.ipynb": "A token ID like 41 is just a label. How does the model get something it can compute with, that can carry meaning?",
    "02_text_to_numbers/02_06_positions.ipynb": "The embedding of `α` is the same wherever `α` appears. How does the model know WHERE each character is?",
    "02_text_to_numbers/02_03_bpe.ipynb": "One character per token makes the text long and each token nearly meaningless. Can tokens be bigger than one character? BPE is the standard answer.",
    "03_attention/03_00_chapter_overview.ipynb": "To guess the next letter well, the model must look back at the letters before it, and know which ones matter. That is attention, the heart of every modern language model.",
    "03_attention/03_01_simple_attention.ipynb": "How can a model weigh the earlier letters and mix them into one summary? First with nothing to learn: just scores, softmax and a weighted average.",
    "03_attention/03_02_trainable_attention.ipynb": "With nothing to learn, every letter mostly looks at itself. Query, key and value let the model learn what to look for, which is the attention step in `model.py`.",
    "03_attention/03_03_hiding_the_future.ipynb": "In practice the next letter is the answer, so a model that can see it would just copy it. The causal mask hides every later letter.",
    "03_attention/03_04_many_heads.ipynb": "One head gives one way of looking. Several heads can each look for something different; our `model.py` has just one.",
    "04_building_the_model/04_00_chapter_overview.ipynb": "Attention alone is not a full model. A GPT stacks identical blocks, each with attention, a feed-forward step, layer norms and shortcuts; our `model.py` has none of these yet.",
    "04_building_the_model/04_01_layer_norm.ipynb": "Numbers passed through many layers drift too small or too big. Layer norm resets each row so a deep model stays steady.",
    "04_building_the_model/04_02_feed_forward.ipynb": "After attention has mixed in the earlier letters, each letter needs a step of its own to think about the result: a small 2-layer network with a GELU curve.",
    "04_building_the_model/04_03_shortcuts.ipynb": "In a tall stack the learning signal fades before it reaches the lower floors. A shortcut that adds the input back keeps it alive.",
    "04_building_the_model/04_04_one_block.ipynb": "The parts so far (attention, feed-forward, layer norms, shortcuts) are loose. How do they fit into one block that can be stacked?",
    "04_building_the_model/04_05_the_full_model.ipynb": "One block is not a model. Embeddings, a stack of blocks and an output head make the full GPT; it has far more weights than our `model.py`, and can already write (badly, untrained).",
    "05_training_on_the_odyssey/05_00_chapter_overview.ipynb": "A new model (`HomerGreekLLM` in `src/homer_ai/model.py`) is full of random numbers, so its guesses are random (you saw it in [01_04](../01_foundations/01_04_our_model_as_a_black_box.ipynb)). Training turns them into Homer-like Greek.",
    "05_training_on_the_odyssey/05_01_loss_and_epochs.ipynb": "Training prints a loss every so often. We need to read it correctly: print totals where you meant averages, or print too rarely, and big jumps in the loss stay hidden.",
    "05_training_on_the_odyssey/05_02_learning_rate.ipynb": "On a 5-line practice run the loss keeps jumping back up: 0.15 at epoch 200, but 0.48 at epoch 400. Why, and how do we stop it?",
    "05_training_on_the_odyssey/05_03_memorising_vs_understanding.ipynb": "The best loss of 05_02, 0.008, was measured on the same 5 lines the model read 800 times. Did it learn, or just memorise?",
    "05_training_on_the_odyssey/05_04_running_fair_experiments.ipynb": "We change a line of code or a setting, and the loss moves. How do we KNOW our change did it, and not luck or a hidden mistake?",
    "05_training_on_the_odyssey/05_05_validation.ipynb": "A falling training loss may only mean the model memorised the text. We hide the last 10% of Book 1, measure the loss on it, and see whether the model really learns.",
    "05_training_on_the_odyssey/05_06_saving_and_loading.ipynb": "Training took minutes and the result lives only in memory. How do we save it to a file, load it back exactly, and use it somewhere else, like the MCP server?",
    "05_training_on_the_odyssey/05_07_train_the_gpt.ipynb": "The course has two models, and the GPT of chapter 4 has its own training loop, `train_gpt`. What's different in it, why dropout 0.3, and how do you train your own?",
    "06_making_it_talk/06_00_chapter_overview.ipynb": "The model can only guess ONE next letter, so how does it write a whole line? And how can other AI apps, like Claude, ask our model for Greek?",
    "06_making_it_talk/06_01_writing_greek.ipynb": "We have a trained model. How do we turn its scores into written Greek, and why does the plainest way keep repeating itself?",
    "06_making_it_talk/06_02_less_repetition.ipynb": "Always taking the most likely letter loops (`και και και`). Choosing at random fixes that, but too much chance gives gibberish. Temperature and top-k set the balance.",
    "06_making_it_talk/06_03_the_mcp_server.ipynb": "Our model writes Greek in a notebook. How does `homer_mcp_server.py` offer it to Claude as a tool, and what does Claude need to know to use it well?",
    "07_advanced_iliad_and_combining/07_00_chapter_overview.ipynb": "We have an Odyssey model. What do we gain by doing it all again on the Iliad, how do we compare the two fairly, and how could we combine them?",
    "07_advanced_iliad_and_combining/07_01_the_corpus.ipynb": "Before training on a second poem, we should know what is in `data/homer/`: how big each text is, how many characters it uses, and where the two Greek poems differ.",
    "07_advanced_iliad_and_combining/07_02_an_iliad_model.ipynb": "Is our code really general? We run the same steps on the Iliad, change only the folder, and find the one place that needs a small fix.",
    "07_advanced_iliad_and_combining/07_03_comparing_fairly.ipynb": "Two models trained on different poems are only comparable if they read text the same way. A shared vocabulary and a 2×2 test make the comparison fair.",
    "07_advanced_iliad_and_combining/07_04_combining_both.ipynb": "Can ONE model learn both poems? We put `<|endoftext|>` between them, compare it fairly with the single-poem models, and say what it gives and costs.",
    "07_advanced_iliad_and_combining/07_05_english_and_greek.ipynb": "The same code can learn Butler's English too. How do its characters, line lengths and token counts differ from Greek's?",
    "08_advanced_fine_tuning/08_00_chapter_overview.ipynb": "Building a model from scratch taught us how it works. In real work you mostly take a model somebody already trained and adapt it. What does that involve, and what can a laptop do?",
    "08_advanced_fine_tuning/08_01_loading_gpt2.ipynb": "OpenAI's GPT-2 is one file of 548 MB. What's inside it, and how do its numbers get into the `GPTModel` we wrote in chapter 4?",
    "08_advanced_fine_tuning/08_02_spam_classifier.ipynb": "GPT-2 writes text. How do we make it sort text messages into spam and not spam, by training only a small part of it for about a minute?",
    "08_advanced_fine_tuning/08_03_instructions.ipynb": "Ask GPT-2 a question and it just keeps writing. How do chat models learn to answer a request, and how far does a small model get on a laptop?",
    "08_advanced_fine_tuning/08_04_lora.ipynb": "Fine-tuning changes millions of weights, and every job needs its own changed copy. LoRA trains two thin tables beside each layer and leaves the model itself alone.",
    "09_advanced_grounded_answers/09_00_chapter_overview.ipynb": "Ask a model about Homer and it answers from memory, with nothing you can check. How do we get an answer that comes from the real text, with the book and line beside it?",
    "09_advanced_grounded_answers/09_01_chunking_and_bm25.ipynb": "Before anything can be looked up, the poems must be cut into passages that know where they came from, and we need a score that says which passage fits a question.",
    "09_advanced_grounded_answers/09_02_search_and_scores.ipynb": "One search that worked proves nothing. How good is ours on 40 questions with known answers, and would vectors do better than counting words?",
    "09_advanced_grounded_answers/09_03_the_search_tool.ipynb": "The search runs in a notebook. How do Claude and other AI apps get to call it, and what must they be told about its limits?",
    "09_advanced_grounded_answers/09_04_with_and_without.ipynb": "Does handing a model the passages make its answers better? We ask the same questions with and without them, and then look at what changes when the text is a company's own.",
    "10_advanced_your_domain/10_00_chapter_overview.ipynb": "Homer was practice. Why would a company want its own LLM, what does that usually mean, and what does the course's pattern look like on a firm's own documents, emails and tickets?",
    "10_advanced_your_domain/10_01_support_with_citations.ipynb": "A customer asks about a savings account. How do we answer from the product sheet with its section and version, get a rate exactly right, and hand over to a person when the assistant shouldn't answer?",
    "10_advanced_your_domain/10_02_spotting_complaints.ipynb": "Hundreds of emails arrive, and a missed complaint is costly. How do we train a small model to flag complaints and vulnerable customers, and score it by what it misses?",
    "10_advanced_your_domain/10_03_service_desk_triage.ipynb": "Every company has a service desk. How do we send a ticket to the right team and suggest a how-to article with no training, and how does the way we split the data change the score?",
    "10_advanced_your_domain/10_04_pitfalls_and_next_steps.ipynb": "Three demos with good-looking scores. What would make them look naive to the people who'd run them for real, and what does an organisation add around them?",
    "A_toolbox/A_00_overview.ipynb": "A beginner meets many tool names at once. This notebook gives each one a single plain line, so none of them is a mystery.",
    "A_toolbox/A_01_python_essentials.ipynb": "Every code cell in the course uses the same small set of Python ideas. Learning just these is enough to follow along.",
    "A_toolbox/A_02_pytorch_tensors.ipynb": "Once the text is numbers, those numbers live in a tensor, and the model is built from tensors too. PyTorch does the fast maths, and the maths needed for learning.",
    "A_toolbox/A_03_dot_product.ipynb": "One small piece of maths sits at the heart of attention (chapter 3) and of `@` in `model.py`. Once you know it, the attention code stops looking like magic.",
    "A_toolbox/A_04_pytorch_in_this_repo.ipynb": "`model.py` and the training loop are short, but every line is a PyTorch tool. Here they're explained one at a time.",
    "A_toolbox/A_05_tiktoken_bpe.ipynb": "In 02_03 you built a small BPE yourself. Real projects often use a ready-made one: does GPT-2's suit Homer?",
    "A_toolbox/A_06_other_tools.ipynb": "uv, Ruff, Jupyter, git and MCP aren't about AI, but every AI engineer uses tools like them every day.",
    "A_toolbox/A_07_organising_code.ipynb": "Two files need the model class. Why does it live in one file, `model.py`, that both of them import, instead of being copied into each?",
}


def where(stage, files, here=None):
    """The 'Where this fits' box: the journey with this stage in [brackets]."""
    line = " → ".join(f"[{s}]" if s == stage else s for s in STAGES)
    if here:
        here = f"You are here: {here}"
    else:
        here = (
            f"You are here: **{stage}**"
            if stage
            else "You are here: **the start**, looking at the whole map"
        )
    out = f"> **Where this fits**\n>\n> ```\n> {line}\n> ```\n>\n> {here}"
    if files:
        out += "\n>\n> Project files this connects to:\n>\n" + "\n".join(
            f"> - `{f}`" for f in files
        )
    return md(out)


def learn(items):
    return md("## 1. What you'll learn\n\n" + "\n".join(f"- {x}" for x in items))


def words(rows):
    body = "| Word | Meaning in plain English | Example |\n|---|---|---|\n"
    body += "\n".join(f"| **{w}** | {m} | {e} |" for w, m, e in rows)
    return md(
        "## 2. Words you'll meet\n\nThe new words, explained before we use them. Come back to this table whenever one slips your mind.\n\n"
        + body
    )


def q(question, answer):
    """One question-and-answer cell."""
    return md(f"### ❓ {question}\n\n{answer.strip()}")


WHAT_WE_BUILD_FULL = (
    "What are we actually building?",
    "Have you ever sung along to a song in a language you don't speak? After enough listens you know which sound comes next, and you still have no idea what the words mean. "
    "That's what we build here, for Homer's Greek: a program that goes through the poem over and over, practises guessing what comes next, and ends up writing lines of its own that sound like Homer. "
    "At the very end, other AI apps like Claude can ask it for a line, the way a friend might ask you to hum that song. "
    "The picture has one limit: you started with ears and a brain, and our program starts as a pile of random numbers, so most of the course goes into building those.\n\n"
    "The proper name for such a program is a **large language model**, or **LLM** for short: a program that reads some text and guesses what comes next. "
    "Ours learns from Homer's *Odyssey*, in the original Greek, so it learns to write Greek that sounds like Homer.\n\n"
    "Really, the whole course is about making those guesses better. On the way, the Greek text passes through six stages:\n\n"
    f"```\n{JOURNEY}\n```\n\n"
    "The last stage, the MCP tool, is how other AI apps, like Claude, can ask our model for Greek text.",
)


def background(where_now=None, problem=None):
    """Section 3 heading only. The one-line 'what this solves' lives in SOLVES."""
    return [md("## 3. The idea")]


def step_intro():
    return md(
        "## 4. Step by step\n\nEach step is one question. Read it, guess what the code will do, run it, then look at what you got."
    )


def questions(pairs):
    if not pairs:
        return []
    cells = [
        md(
            "## Questions learners ask\n\nThese are real questions that came up while we were building this project, with short answers."
        )
    ]
    cells += [q(a, b) for a, b in pairs]
    return cells


def recap(points, nxt, can=None):
    """The recap: at most 3 points, then `can`, a short "You can now" list of skills."""
    out = "## 6. Recap\n\n" + "\n".join(f"- {p}" for p in points)
    if can:
        out += "\n\n**You can now:**\n\n" + "\n".join(f"- {c}" for c in can)
    return md(out + f"\n\n**Next:** {nxt}")


def escape_table_pipes(cells):
    """In markdown tables, write `<|unk|>` as `<\\|unk\\|>` so the | doesn't split the cell."""
    for c in cells:
        if c["cell_type"] != "markdown":
            continue
        fixed = []
        for line in "".join(c["source"]).splitlines(keepends=True):
            if line.startswith("|") or line.startswith("> |"):
                line = line.replace("<|", "<\\|").replace("|>", "\\|>")
            fixed.append(line)
        c["source"] = fixed
    return cells


# Words already defined, per reading track, in the order notebooks are built.
# The main course and the toolbox are separate tracks (the toolbox is dipped into).
DEFINED = {"main": {}, "toolbox": {}}


def _norm(word):
    return word.replace("`", "").replace("*", "").strip().lower()


def open_with_solves(rel, cells):
    """Put this notebook's one-line purpose and a course-map link in the opening box."""
    if rel not in SOLVES:
        return cells
    depth = rel.count("/")
    course_map = "../" * depth + "00_start_here/00_00_course_map.ipynb"
    for c in cells:
        src = "".join(c["source"])
        if c["cell_type"] == "markdown" and src.startswith("> **Where this fits**"):
            src += f"\n>\n> **This notebook:** {SOLVES[rel]}\n>\n> For the big picture, what we're building and why, see the [course map]({course_map})."
            c["source"] = src.splitlines(keepends=True)
            break
    return cells


def only_new_words(rel, cells):
    """'Words you'll meet' lists only words NEW to this track; older ones get a link."""
    if "/" not in rel or rel.startswith("00_start_here"):
        return cells
    track = "toolbox" if rel.startswith("A_toolbox") else "main"
    seen = DEFINED[track]
    depth = rel.count("/")
    for c in cells:
        src = "".join(c["source"])
        if not (
            c["cell_type"] == "markdown" and src.startswith("## 2. Words you'll meet")
        ):
            continue
        lines = src.split("\n")
        rows = [line for line in lines if line.startswith("| **")]
        keep, older = [], []
        for row in rows:
            word = re.match(r"^\| (\*\*.+?\*\*) \|", row).group(1)
            key = _norm(word)
            if key in seen:
                older.append((word.strip(), seen[key]))
            else:
                keep.append(row)
                seen[key] = rel
        head = "## 2. Words you'll meet"
        body = ""
        if keep:
            body += (
                "\n\n| Word | Meaning in plain English | Example |\n|---|---|---|\n"
                + "\n".join(keep)
            )
        else:
            body += "\n\nNo new words this time."
        if older:
            links = []
            for word, where_rel in older:
                link = "../" * depth + where_rel
                links.append(
                    f"{word} ([{where_rel.split('/')[-1].split('_')[0]}_{where_rel.split('/')[-1].split('_')[1]}]({link}))"
                )
            body += "\n\nWords you've met before: " + ", ".join(links) + "."
        c["source"] = (head + body).splitlines(keepends=True)
        break
    return cells


def renumber_sections(cells):
    """Number the '## N. Title' headings 1, 2, 3 ... in order, so no notebook skips a number."""
    n = 0
    for c in cells:
        src = "".join(c["source"])
        if c["cell_type"] == "markdown" and re.match(r"## \d+\. ", src):
            n += 1
            c["source"] = re.sub(r"^## \d+\. ", f"## {n}. ", src).splitlines(
                keepends=True
            )
    return cells


def save(rel, cells, root=None):
    cells = open_with_solves(rel, cells)
    cells = only_new_words(rel, cells)
    cells = escape_table_pipes(cells)
    cells = renumber_sections(cells)
    cells = fold_reveals(cells)
    for i, c in enumerate(cells):
        c["id"] = f"cell-{i}"
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    path = (root or ROOT) / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
        f.write("\n")
