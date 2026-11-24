"""Draw the course's studio-look diagrams to PNG files in docs/images/.

Run: uv run python scripts/export_diagrams.py

GitHub shows notebooks without their outputs, so a learner browsing the repo
sees no pictures. These PNGs are the same diagrams the notebooks draw: the
drawing code is read from `DIAGRAM_*` in scripts/course/diagrams.py (the single
source of truth), run with the same small inputs the notebooks compute first.

The output is deterministic: a fixed font (the DejaVu Sans that ships with
matplotlib), a fixed dpi and no metadata, so a second run gives identical bytes.
CI re-runs this and fails if docs/images/ changed (`git diff --exit-code`).

`DIAGRAMS` is plain data in scripts/course/diagrams.py: the chapter READMEs use it
to show each chapter's pictures, so the file names and alt texts live there once.
"""

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


def main():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    import homer_ai.studio_style  # noqa: F401  (sets the look; fonts reset below)

    # Same fonts on every machine: DejaVu Sans ships inside matplotlib.
    plt.rcParams["font.family"] = ["DejaVu Sans"]

    sources = diagram_sources()
    shown = []
    plt.show = lambda: shown.append(plt.gcf())  # the diagram code ends in plt.show()
    OUT.mkdir(parents=True, exist_ok=True)
    for file_name, constant, prelude, _notebook, _alt in DIAGRAMS:
        namespace = {}
        exec(prelude, namespace)  # noqa: S102  (our own code, read from this repo)
        shown.clear()
        exec(sources[constant], namespace)  # noqa: S102
        fig = shown[-1]
        path = OUT / f"{file_name}.png"
        fig.savefig(path, dpi=120, metadata={"Software": None})
        plt.close("all")
        print(f"{path.relative_to(ROOT)}  {path.stat().st_size // 1024} KB")


if __name__ == "__main__":
    main()
