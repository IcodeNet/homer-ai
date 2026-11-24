# 0002. Notebooks generated from one script

Status: Accepted

## Context

The course is 51 notebooks (`git ls-files notebooks | grep -c ipynb`) plus a README per chapter. The same ideas, numbers and diagrams appear in several of them, and in the chapter READMEs and the README on GitHub. Notebooks are JSON files. Hand-editing them gives noisy diffs and stale outputs, and a fix in one place easily misses another.

GitHub also shows notebooks without their outputs, so key diagrams need to exist as image files too.

## Decision

The `scripts/course/` package, run by `scripts/build_notebooks.py`, is the single source of truth. It writes every notebook, every chapter README and the blank template `scripts/templates/00_template.ipynb`. Nobody edits a `.ipynb` or a chapter README by hand; a hand edit is overwritten by the next build (`CONTRIBUTING.md`).

The build is deterministic. `scripts/export_diagrams.py` draws the `DIAGRAM_*` code of the same package into `docs/images/*.png` with a fixed font, a fixed dpi and no metadata, so its output is also identical from run to run. The diagram list (`DIAGRAMS` in `scripts/course/diagrams.py`) is plain data that the chapter READMEs and the export script import, so file names and alt texts live once.

CI ([.github/workflows/ci.yml](../../.github/workflows/ci.yml)) enforces it: it builds twice and runs `git diff --exit-code`, then re-exports the diagrams and runs `git diff --exit-code docs/images`. The optional pre-commit hook does the notebook check locally.

## Consequences

- One edit changes every place a fact appears, and a number can be fixed once.
- Reviews of the generated files are large, so review the script change instead.
- Notebooks are committed without outputs; CI proves they run (`scripts/run_notebooks.py`) and links are valid (`scripts/check_links.py`).
- The course source is long (about 13,000 lines in total), because it holds all the course text and code cells. One module per chapter under `scripts/course/` keeps each file readable.
- Contributors need to know about the script before touching a notebook. `CONTRIBUTING.md` says it first.
- Code cells inside the script are strings, so ruff does not check them directly. The notebooks are excluded from formatting (`pyproject.toml`), and `run_notebooks.py` catches code that does not run.
