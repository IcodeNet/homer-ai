"""Draw the course's studio-look diagrams to PNG files in docs/images/.

Run: uv run python scripts/export_diagrams.py

GitHub shows notebooks without their outputs, so a learner browsing the repo
sees no pictures. These PNGs are the same diagrams the notebooks draw: the
drawing code is read from `DIAGRAM_*` in scripts/course/diagrams.py (the single
source of truth), run with the same small inputs the notebooks compute first.

The output is deterministic on one machine: a fixed font (the DejaVu Sans that
ships with matplotlib), a fixed dpi and no metadata, so a second run gives
identical bytes. Other operating systems can differ in a few pixels (font
hinting, compression), so CI runs `--check`: it draws into a temporary folder and
fails if a picture is missing, changes size, or differs by more than a tiny
amount from the committed one.

`DIAGRAMS` is plain data in scripts/course/diagrams.py: the chapter READMEs use it
to show each chapter's pictures, so the file names and alt texts live there once.
"""

import argparse
import sys
import tempfile
from pathlib import Path

from course import diagrams
from course.diagrams import DIAGRAMS

HERE = Path(__file__).parent
ROOT = HERE.parent
OUT = ROOT / "docs" / "images"


def diagram_sources():
    """The DIAGRAM_* strings of scripts/course/diagrams.py, by constant name."""
    return {
        name: value
        for name, value in vars(diagrams).items()
        if name.startswith("DIAGRAM_") and isinstance(value, str)
    }


def export(out):
    """Draw every diagram into the folder `out`; return the written paths."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    import homer_ai.studio_style  # noqa: F401  (sets the look; fonts reset below)

    # Same fonts on every machine: DejaVu Sans ships inside matplotlib.
    plt.rcParams["font.family"] = ["DejaVu Sans"]

    sources = diagram_sources()
    shown = []
    plt.show = lambda: shown.append(plt.gcf())  # the diagram code ends in plt.show()
    out.mkdir(parents=True, exist_ok=True)
    written = []
    for file_name, constant, prelude, _notebook, _alt in DIAGRAMS:
        namespace = {}
        exec(prelude, namespace)  # noqa: S102  (our own code, read from this repo)
        shown.clear()
        exec(sources[constant], namespace)  # noqa: S102
        fig = shown[-1]
        path = out / f"{file_name}.png"
        fig.savefig(path, dpi=120, metadata={"Software": None})
        plt.close("all")
        written.append(path)
    return written


MAX_PIXEL_DIFFERENCE = 0.01  # mean absolute difference, colours from 0 to 1


def check():
    """Draw into a temporary folder and compare with docs/images/; return 0 or 1."""
    import matplotlib.image as mpimg
    import numpy as np

    problems = 0
    with tempfile.TemporaryDirectory() as folder:
        for new in export(Path(folder)):
            old = OUT / new.name
            if not old.exists():
                print(f"missing in docs/images: {new.name}")
                problems += 1
                continue
            a, b = mpimg.imread(old), mpimg.imread(new)
            if a.shape != b.shape:
                print(f"{new.name}: size {a.shape} committed, {b.shape} drawn now")
                problems += 1
                continue
            difference = float(np.abs(a - b).mean())
            status = "ok" if difference <= MAX_PIXEL_DIFFERENCE else "CHANGED"
            print(f"{new.name}  difference {difference:.4f}  {status}")
            problems += status != "ok"
    if problems:
        print(
            f"{problems} picture(s) differ: run "
            "`uv run python scripts/export_diagrams.py` and commit docs/images/"
        )
    return 1 if problems else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--check",
        action="store_true",
        help="compare with docs/images/ instead of writing",
    )
    if parser.parse_args().check:
        sys.exit(check())
    for path in export(OUT):
        print(f"{path.relative_to(ROOT)}  {path.stat().st_size // 1024} KB")


if __name__ == "__main__":
    main()
