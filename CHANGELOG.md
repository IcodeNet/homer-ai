# Changelog

Every notable change to the project goes here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## Initial release

What the repository contains:

- A course of notebooks that builds a small language model on Homer's Greek, from setup (chapter 0) through foundations, text to numbers, attention, the full GPT, training, and making it talk (chapters 1 to 6), an advanced chapter 7 with the Iliad, a fair comparison, a combined model, and English, an advanced chapter 8 that loads GPT-2's weights into the course's own GPT and fine-tunes it (a spam classifier, a small instruction set, LoRA), an advanced chapter 9 that answers English questions about Homer from the real text, with book and line (passages, BM25 from scratch, scores on labelled questions, a search tool over MCP), and an advanced chapter 10 that takes the same pattern to a company's kind of work, on invented data: support answers with citations and handovers, a fine-tuned GPT-2 that flags complaints and vulnerable customers in email, service-desk triage, and a list of pitfalls.
- An appendix toolbox (A_00 to A_07): Python essentials, PyTorch tensors, the dot product, tiktoken, and the other tools.
- Ready-made trained models in `models/`: the GPT of chapter 4 and the small one-head model, so every notebook works before you train your own.
- The code in `src/homer_ai/`: the character tokenizer, the small model, the GPT, checkpoint saving and loading, training, text generation, a word search over both poems (`retrieval.py`: passages, BM25, recall@k and MRR), and an MCP server with two tools: one lets Claude ask your model for Greek text, the other finds real passages with their book and line.
- The `homer-train` and `homer-serve` commands.
- The code of chapter 10: `domain.py` (versioned product sheets, an exact rate lookup, a handover, a word-list guard, recall, precision and macro-F1, two ticket routers) and `domain_server.py`, a second MCP server with the tools `search_terms`, `get_rate` and `escalate` and an audit trail (`uv run python -m homer_ai.domain_server`).
- Invented company data in `data/domain/`: 4 product sheets, a FAQ, a rates table and 39 labelled support questions; 400 template emails and 24 handwritten ones, labelled complaint and vulnerable; 200 service-desk tickets and 10 knowledge-base articles. `scripts/make_domain_data.py` writes the emails and tickets from a seed, byte for byte.
- `scripts/benchmark_domain.py`, which asks Claude the support questions with and without the chapter's MCP tools (run by hand).
- The Odyssey and the Iliad in Greek and English, one file per book, with scripts that download and check them (`data/homer/`).
- 40 labelled questions about the two poems in `data/eval/`, and `scripts/benchmark_grounding.py`, which asks Claude those questions with and without the retrieved passages (run by hand).
- An optional extra, `rag` (`uv sync --extra rag`), for the one notebook cell that compares BM25 with a pretrained sentence-embedding model.
- Eighteen of the course's diagrams as PNG images in `docs/images/`, so they show on GitHub (`scripts/export_diagrams.py` redraws them).
- A FAQ, a contributing guide, and a picture-book of a neural network (`docs/neural-network-building.html`, opened locally in a browser).
- Tests for `src/homer_ai/`, including the chapter 8 helpers, the chapter 9 search and the chapter 10 tools and data, a script that runs every notebook top to bottom, a link checker, and continuous integration that runs them all.
- An architecture overview (`docs/ARCHITECTURE.md`) and seven short decision records (`docs/decisions/`).
- Issue templates, a pull request template, a citation file and a pre-commit configuration.
- The course source in `scripts/course/` (one module per chapter). `uv run python scripts/build_notebooks.py` writes every notebook and chapter README from it, and the result is the same on every run.
