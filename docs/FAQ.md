# FAQ

Setup, troubleshooting and how the course works. The [README](../README.md) is the short version, and the questions about the ideas themselves are answered in the notebooks.

## ❓ How do I get set up?

**1. Install three free tools.**
- **VS Code**, a code editor: <https://code.visualstudio.com/>.
- **git**, a tool that downloads and saves snapshots of projects: <https://git-scm.com/>.
- **uv**, a tool that installs Python and all the libraries this project needs, in one go: <https://docs.astral.sh/uv/>.
  (A library is code someone else wrote and shared, so you don't have to write it yourself.)

**2. Open a terminal.** That's the window where you type commands instead of clicking.
In VS Code it's the menu **Terminal → New Terminal**.

**3. Copy the project to your computer and install it:**

```
git clone https://github.com/IcodeNet/homer-ai
cd homer-ai
uv sync
```

`uv sync` reads `pyproject.toml`, the project's shopping list, and installs everything on it.
The clone is small, a few MB. `uv sync` is the big download: PyTorch and the other libraries, about 900 MB, into a folder called `.venv` inside the project.
You only do it once, and `.venv` is never committed to git.

**4. Open the folder in VS Code.** It will suggest the extensions this project uses (Python, Pylance, Ruff, Jupyter). Say yes.

**5. Check it works.** Open [00_01_setup.ipynb](../notebooks/00_start_here/00_01_setup.ipynb), pick the `.venv` kernel, and run its last cell.
Every line should show ✅. If one shows ❌, it tells you what to fix, and [00_02: reading errors](../notebooks/00_start_here/00_02_reading_errors.ipynb) helps if you get an error message instead.

## ❓ I'm on Windows. Do the commands work?

The commands in this course are written for macOS and Linux. On Windows, the easy way is to run them in **WSL** (Windows Subsystem for Linux, a Linux terminal inside Windows) or in **Git Bash**, which comes with git. There `cp`, `diff`, `time` and `>` behave as the notebooks describe.
Two things have another name on Windows. The project's Python is `.venv\Scripts\python` (on macOS and Linux it's `.venv/bin/python`), so that's the kernel to pick in VS Code. And `uv run …` works the same everywhere.
One note for macOS: `code .` (open this folder in VS Code from the terminal) only works after you've run **Shell Command: Install 'code' command in PATH** once, from VS Code's Command Palette (Cmd+Shift+P). Or just use **File → Open Folder**.

## ❓ Something doesn't work. Where do I look?

- **"Select Kernel" in VS Code, a notebook that opens as a wall of text (JSON), or `Failed to initialize NumPy`:** answered in [00_01](../notebooks/00_start_here/00_01_setup.ipynb).
- **`NameError` or another red message:** [00_02](../notebooks/00_start_here/00_02_reading_errors.ipynb). Nine times out of ten, the fix is to run the cells from the top, in order.
- **Ruff complains about a line you want to keep:** add `# noqa: E501` (with that rule's code) at the end of the line. That silences just that rule, on just that line.
- **A tool you don't know (tensor, tiktoken, uv…):** have a look in the [toolbox](../notebooks/A_toolbox/A_00_overview.ipynb).

## ❓ How does this course work?

You open the notebooks in number order. The [course map](../notebooks/00_start_here/00_00_course_map.ipynb) shows the whole thing: what we build, the chapters, and what a notebook looks like.
A **notebook** (an `.ipynb` file) is a page of small boxes called cells. You run one with Shift+Enter and its result appears underneath. The idea is that you guess the result first, then run it.
Each notebook stands on its own and fits in a short session. The milestones in the [README](../README.md) are just checkpoints along the way.

## ❓ Do I need a GPU? Do I have to train a model first?

No to both. A **GPU** is a graphics chip that does lots of sums at once. Our model is small enough for a laptop's normal processor, the CPU.
And the repo ships ready-made models, `models/odyssey_book1_gpt.pt` and `models/odyssey_book1.pt` ([what they are](../models/README.md)).
A notebook that needs a trained model loads yours if you have one (`src/homer_ai/homer_gpt.pt`, or `homer_model.pt` for the small model), and the ready-made one if not. It prints one line telling you which.

## ❓ How do I train my own Homer model?

In a terminal inside the project folder, run `uv run homer-train --model gpt`. It takes about 3 minutes. It trains on the first 90% of Book 1 and prints the held-out loss every few epochs
(an **epoch** is about one pass through the text). Press Ctrl+C if you want to stop early. The result is saved as `src/homer_ai/homer_gpt.pt`.
For the small one-head model of chapter 1, it's `uv run homer-train` with no option (about 8 to 10 minutes), saved as `src/homer_ai/homer_model.pt`. Each run replaces the file of the run before, so copy it first if you want to keep it. The ready-made files in `models/` are never touched.
Chapter 5 explains what training actually does, and [05_07](../notebooks/05_training_on_the_odyssey/05_07_train_the_gpt.ipynb) walks through the GPT's training loop and what the command prints. Which text it reads is set by `BOOKS_DIR` and `BOOKS` at the top of `src/homer_ai/train_homer.py`.

## ❓ Does the MCP server need a trained model?

Yes, but it doesn't have to be your own. It uses the file in `$HOMER_MODEL` if that's set, else your own `homer_gpt.pt`, else the ready-made GPT, else the small models. If there's none at all, it stops and tells you how to train one.
[06_03](../notebooks/06_making_it_talk/06_03_the_mcp_server.ipynb) shows how to connect it to Claude Code. And if you're wondering how an LLM can "use a tool" at all, that's in the [course map](../notebooks/00_start_here/00_00_course_map.ipynb).

## ❓ Can I ask it questions about Homer?

Not the model. It only continues Greek letters. But the same MCP server has a second tool, `search_homer`, that finds real passages of both poems for a question and returns them with their book and line. Claude, or another AI app, reads those and answers, and you can check the reference.
[Chapter 9](../notebooks/09_advanced_grounded_answers/09_00_chapter_overview.ipynb) builds it, and [09_03](../notebooks/09_advanced_grounded_answers/09_03_the_search_tool.ipynb) has the settings for Claude Code, Claude Desktop, Codex CLI, LM Studio and nanobot. It needs no download and no training.

## ❓ Does a company need its own LLM? And is the company data in chapter 10 real?

Usually *its own LLM* means a model that answers from the firm's documents, or a small one that runs on the firm's machines. Training one from scratch is rare. [Chapter 10](../notebooks/10_advanced_your_domain/10_00_chapter_overview.ipynb) goes through the reasons and works three examples: support answers with citations, complaints spotted in email, and service-desk triage.
None of its data is real. The firm, the product sheets, the emails and the tickets in `data/domain/` are invented ([how they were made](../data/domain/README.md)), so the scores there show a method and say nothing about a real firm's mail.
The chapter has its own small MCP server, started with `uv run python -m homer_ai.domain_server`. Notebook 10_02 fine-tunes GPT-2, so it uses the 548 MB file of chapter 8; the other four need no download.

## ❓ What is the `rag` extra in `pyproject.toml`?

It's optional. `uv sync --extra rag` installs `sentence-transformers`, which one cell of [09_02](../notebooks/09_advanced_grounded_answers/09_02_search_and_scores.ipynb) uses to compare our word search with a small pretrained embedding model (it downloads about 90 MB on its first run). Without the extra, that cell prints one line and the rest of the notebook runs as usual.

## ❓ I don't read Greek. Can I still follow?

Yes. I mean, the model doesn't read Greek either. You only need to spot patterns, like lots of words ending in `-ν`. The repo also has Butler's English translation, so you can build an English model too ([07_05](../notebooks/07_advanced_iliad_and_combining/07_05_english_and_greek.ipynb)).

## ❓ Is mine really "large"?

No, ours is tiny. The GPT has 835,072 weights and the small model 11,840, against about **124 million** in GPT-2 small. It's the same idea at toy size, which is why it trains on a laptop ([01_00](../notebooks/01_foundations/01_00_chapter_overview.ipynb)).

## ❓ What is `docs/neural-network-building.html`?

A picture-book that shows a neural network as a building with floors. GitHub shows HTML files as plain text, so open it from your own copy of the repo, in your browser. It works offline.

## ❓ Where does the Greek text come from?

The Greek is from the Perseus Digital Library (Tufts University), licensed **CC BY-SA 4.0**: A. T. Murray's 1919 edition
of the Odyssey, and Monro and Allen's edition of the Iliad. The English is Samuel Butler's prose translations, which are in the public domain.
See [data/homer/README.md](../data/homer/README.md) for the sources and [data/homer/LICENSE.md](../data/homer/LICENSE.md) for the licences.

## ❓ The first run of a tiktoken notebook needs the internet. Why?

Because [A_05](../notebooks/A_toolbox/A_05_tiktoken_bpe.ipynb) and [07_05](../notebooks/07_advanced_iliad_and_combining/07_05_english_and_greek.ipynb) load GPT-2's vocabulary through `tiktoken`. It downloads it once and keeps it on your computer after that. Chapter 8 also downloads once: GPT-2's weights (548 MB) and a small set of text messages, to `~/.cache/homer-ai/`, outside the project. Every other notebook works offline.
