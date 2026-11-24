"""Run every notebook, top to bottom, in a real kernel, each from its own folder.

Run: uv run python scripts/run_notebooks.py [--solutions] [project folder]

Notebooks run in order (00_..., 01_..., A_...). The notebook files on disk are
not changed. Exits with 1 if any notebook fails.

The "Your turn" exercises are checked too. Their cells carry tags (see
`exercise()` in scripts/course/cells.py). In a normal run the blanks are still
`None`, so every check cell must print ⏳ and nothing else. With `--solutions`,
only the notebooks that have exercises run, each starter cell is replaced (in
memory) by the solution from its folded answer, and every check must print ✅.

The notebooks in `NEEDS_DOWNLOAD` fetch GPT-2's weights (548 MB) on their first
run (chapter 8, and 10_02). With the environment variable
`HOMER_SKIP_DOWNLOADS=1` (set in CI) they are not run: they are only checked for
a valid notebook format, and shown as "skipped".
"""

import argparse
import os
import re
import sys
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient

# Notebooks that download something big (path below notebooks/, as a prefix).
NEEDS_DOWNLOAD = (
    "08_advanced_fine_tuning/08_01_",
    "08_advanced_fine_tuning/08_02_",
    "08_advanced_fine_tuning/08_03_",
    "08_advanced_fine_tuning/08_04_",
    "10_advanced_your_domain/10_02_",
)


SOLUTION = re.compile(r"```python\n(.*?)\n```", re.DOTALL)


def has_tag(cell, tag):
    return tag in cell.metadata.get("tags", [])


def use_solutions(notebook):
    """Put each exercise's solution in place of its starter cell. Returns how many."""
    starters = [c for c in notebook.cells if has_tag(c, "exercise-starter")]
    answers = [c for c in notebook.cells if has_tag(c, "exercise-solution")]
    if len(starters) != len(answers):
        raise ValueError("every exercise needs one starter and one solution cell")
    for starter, answer in zip(starters, answers, strict=True):
        starter.source = SOLUTION.search(answer.source).group(1)
    return len(starters)


def check_lines(notebook):
    """What each exercise's check cell printed, one list of lines per exercise."""
    printed = []
    for cell in notebook.cells:
        if has_tag(cell, "exercise-check"):
            outputs = [o for o in cell.get("outputs", []) if o.get("name") == "stdout"]
            text = "".join(o["text"] for o in outputs)
            printed.append(text.strip().splitlines())
    return printed


def wrong_checks(printed, solutions):
    """A message if a check did not print what this kind of run expects, else None."""
    for lines in printed:
        marks = {line[0] for line in lines if line}
        if solutions and not ("✅" in marks and marks <= {"✅"}):
            return "a solution did not get ✅: " + " | ".join(lines)[:200]
        if not solutions and marks != {"⏳"}:
            return "a starter did not get ⏳: " + " | ".join(lines)[:200]
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("root", nargs="?", default=".", help="the project folder")
    parser.add_argument(
        "--solutions",
        action="store_true",
        help="run the notebooks that have exercises, with every solution filled in",
    )
    args = parser.parse_args()
    root = Path(args.root).resolve()
    skip_downloads = os.environ.get("HOMER_SKIP_DOWNLOADS") == "1"
    failed = skipped = exercises = 0
    for path in sorted((root / "notebooks").glob("**/*.ipynb")):
        if ".ipynb_checkpoints" in path.parts:
            continue
        notebook = nbformat.read(path, as_version=4)
        name = path.relative_to(root / "notebooks").as_posix()
        if args.solutions and not check_lines(notebook):
            continue  # no exercises in this notebook
        started = time.time()
        if skip_downloads and name.startswith(NEEDS_DOWNLOAD):
            try:
                nbformat.validate(notebook)
                skipped += 1
                status = "skipped (needs a download; format ok)"
            except nbformat.ValidationError as error:
                failed += 1
                status = "FAIL " + str(error).strip().splitlines()[0][:150]
            print(f"{time.time() - started:6.1f}s  {path.name}  {status}", flush=True)
            continue
        if args.solutions:
            use_solutions(notebook)
        client = NotebookClient(
            notebook,
            timeout=1200,
            kernel_name="python3",
            resources={"metadata": {"path": str(path.parent)}},
        )
        printed = []
        try:
            client.execute()
            printed = check_lines(notebook)
            status = wrong_checks(printed, args.solutions)
            status = "FAIL " + status if status else "ok"
        except Exception as error:
            status = "FAIL " + str(error).strip().splitlines()[-1][:150]
        failed += status != "ok"
        exercises += len(printed)
        print(f"{time.time() - started:6.1f}s  {path.name}  {status}", flush=True)
        if args.solutions:
            for lines in printed:
                for line in lines:
                    print(f"         {line}", flush=True)

    if skipped:
        print(f"{skipped} notebooks skipped (HOMER_SKIP_DOWNLOADS=1)")
    kind = "solved" if args.solutions else "left blank"
    print(f"{exercises} exercises checked ({kind})")
    print(f"{failed} notebooks failed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
