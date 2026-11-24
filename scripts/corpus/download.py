"""Download the original Homer sources into data/raw/ (not stored in git).

Run from the project folder:

    uv run python scripts/corpus/download.py

Then build the clean text files with scripts/corpus/build.py.
"""

import sys
import urllib.request
from pathlib import Path

# Pinned to one Perseus commit, so everyone gets exactly the same Greek text
PERSEUS = "https://raw.githubusercontent.com/PerseusDL/canonical-greekLit/dacb049/data/tlg0012"

ODYSSEY_GREEK = "tlg0012.tlg002.perseus-grc2.xml"
ILIAD_GREEK = "tlg0012.tlg001.perseus-grc2.xml"

# Project Gutenberg mirror (the main site is sometimes down for maintenance)
GUTENBERG = "https://aleph.pglaf.org"

SOURCES = {
    # Greek, CC BY-SA 4.0
    "greek/tlg0012.tlg002.perseus-grc2.xml": f"{PERSEUS}/tlg002/{ODYSSEY_GREEK}",
    "greek/tlg0012.tlg001.perseus-grc2.xml": f"{PERSEUS}/tlg001/{ILIAD_GREEK}",
    # English, Samuel Butler, public domain
    "english/pg1727.txt": f"{GUTENBERG}/1/7/2/1727/1727-0.txt",  # Odyssey
    "english/pg2199.txt": f"{GUTENBERG}/2/1/9/2199/2199-0.txt",  # Iliad
}


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    raw = Path(sys.argv[1]) if len(sys.argv) > 1 else root / "data" / "raw"
    for name, url in SOURCES.items():
        target = raw / name
        target.parent.mkdir(parents=True, exist_ok=True)
        request = urllib.request.Request(url, headers={"User-Agent": "homer-ai-course"})
        with urllib.request.urlopen(request, timeout=60) as response:
            target.write_bytes(response.read())
        print(f"downloaded {target}")


if __name__ == "__main__":
    main()
