## What does this change, and why?

## Checklist

Notebooks and chapter READMEs are generated: change the chapter module in `scripts/course/`, never a `.ipynb`.

- [ ] `uv run python scripts/build_notebooks.py` (run twice: the second run changes nothing)
- [ ] `uv run pytest`
- [ ] `uv run python scripts/run_notebooks.py`
- [ ] `uv run python scripts/check_links.py`
- [ ] `uv run python scripts/export_diagrams.py` (if a diagram changed; commit the new PNGs in `docs/images/`)
- [ ] `uv run ruff check` and `uv run ruff format --check`
- [ ] New words are explained the first time they appear; each heading is a learner question
