"""Check that every relative link in the READMEs and notebooks points at a file.

Run: uv run python scripts/check_links.py [project folder]

Web links (http, https, mailto) and in-page links (#...) are skipped.
Exits with 1 if any link is broken.
"""

import json
import re
import sys
from pathlib import Path

LINK = re.compile(r"\]\(([^)\s]+)\)")
SKIP = re.compile(r"(https?:|mailto:|#)")


def broken_links(source, text, root):
    """Yield (file, link) for every relative link in `text` that points nowhere."""
    for match in LINK.finditer(text):
        link = match.group(1)
        target = link.split("#")[0]
        if SKIP.match(link) or not target:
            continue
        if not (source.parent / target).exists():
            yield source.relative_to(root), link


def main():
    root = Path(sys.argv[1] if len(sys.argv) > 1 else ".").resolve()
    markdown = [root / "README.md", root / "CONTRIBUTING.md"]
    markdown += [root / "models" / "README.md", root / "data" / "homer" / "README.md"]
    markdown += [root / "data" / "homer" / "LICENSE.md"]
    markdown += [root / "data" / "eval" / "README.md"]
    markdown += [root / "data" / "domain" / "README.md"]
    markdown += sorted((root / "docs").glob("**/*.md"))
    markdown += sorted((root / "notebooks").glob("**/README.md"))

    problems = 0
    checked = 0
    for path in markdown:
        if not path.exists():
            print("MISSING FILE", path.relative_to(root))
            problems += 1
            continue
        text = path.read_text(encoding="utf-8")
        checked += len(LINK.findall(text))
        for where, link in broken_links(path, text, root):
            print("BROKEN", where, "->", link)
            problems += 1

    for path in sorted((root / "notebooks").glob("**/*.ipynb")):
        notebook = json.loads(path.read_text(encoding="utf-8"))
        for cell in notebook["cells"]:
            if cell["cell_type"] == "markdown":
                text = "".join(cell["source"])
                checked += len(LINK.findall(text))
                for where, link in broken_links(path, text, root):
                    print("BROKEN", where, "->", link)
                    problems += 1

    print(f"{checked} links checked, {problems} problems")
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
