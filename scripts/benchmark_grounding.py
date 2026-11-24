"""Ask Claude the labelled Homer questions twice: plain, and with search results.

Run (it is not part of CI, and it uses your Claude Code account):

    uv run python scripts/benchmark_grounding.py --limit 8 --model sonnet

For each question in data/eval/questions.jsonl it calls the Claude Code CLI in
headless mode (`claude -p`, no tools, a fixed system prompt) two times:

    plain      only the question
    grounded   the question, plus the k passages `search` finds for it

Both times Claude must end with a line `SOURCE: <poem> <book>` (plain) or
`SOURCE: <poem> <book>:<line>` (grounded). We score:

    right book      the poem and book are one of the labelled places
    right passage   grounded only: the cited line is inside a labelled place

Line numbers are the lines of our own files in data/homer/, which Claude has
never seen, so the plain run is only asked for the book. The answers are
printed too: read them, the scores do not judge the wording.

Notebook 09_04 explains the method and shows one recorded run.
"""

import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import date

from homer_ai.retrieval import load_questions, search

SYSTEM = (
    "You answer questions about Homer's Iliad and Odyssey in one or two plain "
    "sentences. You have no tools. Always finish with the SOURCE line you are "
    "asked for, and write nothing after it."
)
PLAIN = (
    "Question: {question}\n\n"
    "Answer from what you know. Then end with one line in exactly this form, "
    "naming the poem and the number of the book (1 to 24) where this happens:\n"
    "SOURCE: <odyssey or iliad> <book>"
)
GROUNDED = (
    "Question: {question}\n\n"
    "Here are passages from Samuel Butler's translation. Each starts with its "
    "reference in square brackets: poem book:line.\n\n{passages}\n\n"
    "Answer only from these passages. If they do not hold the answer, say so. "
    "Then end with one line in exactly this form, giving the reference of the "
    "passage your answer comes from:\n"
    "SOURCE: <odyssey or iliad> <book>:<line>"
)
# also accepts "iliad book 10" and "the Odyssey, 9:24"
SOURCE = re.compile(
    r"SOURCE:\W*(?:the\s+)?(odyssey|iliad)\W+(?:book\s+)?(\d+)(?::(\d+))?",
    re.IGNORECASE,
)


def pick(questions, limit):
    """`limit` questions spread evenly over the file (always the same ones)."""
    if not limit or limit >= len(questions):
        return questions
    step = len(questions) / limit
    return [questions[int(i * step)] for i in range(limit)]


def ask_claude(prompt, model, folder):
    """One headless call. Returns Claude's text ("" if the call failed)."""
    command = ["claude", "-p", "--model", model, "--tools", "", "--strict-mcp-config"]
    command += ["--no-session-persistence", "--system-prompt", SYSTEM]
    try:
        done = subprocess.run(
            command,
            input=prompt,
            capture_output=True,
            text=True,
            cwd=folder,  # an empty folder: no project files, no CLAUDE.md
            timeout=300,
        )
    except subprocess.TimeoutExpired:
        return ""
    if done.returncode != 0:
        print("claude failed:", done.stderr.strip()[:200], file=sys.stderr)
        return ""
    return done.stdout.strip()


def cited(answer):
    """(poem, book, line or None) from the last SOURCE line, or None."""
    found = SOURCE.findall(answer)
    if not found:
        return None
    poem, book, line = found[-1]
    return poem.lower(), int(book), int(line) if line else None


def right_book(source, question):
    if source is None or source[0] != question["poem"]:
        return False
    return any(source[1] == book for book, _first, _last in question["refs"])


def right_passage(source, question):
    if source is None or source[2] is None or source[0] != question["poem"]:
        return False
    return any(
        source[1] == book and first <= source[2] <= last
        for book, first, last in question["refs"]
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--limit", type=int, default=8, help="how many questions")
    parser.add_argument("--model", default="sonnet", help="model name for claude")
    parser.add_argument("-k", type=int, default=5, help="passages per question")
    parser.add_argument("--out", help="also write every answer to this JSON file")
    args = parser.parse_args()

    if shutil.which("claude") is None:
        sys.exit("The `claude` command was not found. Install Claude Code first.")

    questions = pick(load_questions(), args.limit)
    rows = []
    with tempfile.TemporaryDirectory() as folder:
        for number, question in enumerate(questions, start=1):
            found = search(question["question"], "both", "english", args.k)
            passages = "\n\n".join(f"[{p.ref}] {p.text}" for p, _score in found)
            plain = ask_claude(
                PLAIN.format(question=question["question"]), args.model, folder
            )
            grounded = ask_claude(
                GROUNDED.format(question=question["question"], passages=passages),
                args.model,
                folder,
            )
            row = {
                "id": question["id"],
                "question": question["question"],
                "expected": question["answer"],
                "refs": question["refs"],
                "retrieved": [p.ref for p, _score in found],
                "plain": plain,
                "grounded": grounded,
                "plain_right_book": right_book(cited(plain), question),
                "grounded_right_book": right_book(cited(grounded), question),
                "grounded_right_passage": right_passage(cited(grounded), question),
            }
            rows.append(row)
            print(f"\n[{number}/{len(questions)}] {question['question']}")
            print("  expected:", question["answer"], question["refs"])
            print("  plain:   ", plain.replace("\n", " | "))
            print("  grounded:", grounded.replace("\n", " | "))
            time.sleep(1)

    n = len(rows)
    print(f"\n{date.today()}  model={args.model}  questions={n}  k={args.k}")
    print(f"{'':16} {'plain':>8} {'grounded':>9}")
    plain_book = sum(r["plain_right_book"] for r in rows)
    grounded_book = sum(r["grounded_right_book"] for r in rows)
    grounded_passage = sum(r["grounded_right_passage"] for r in rows)
    print(f"{'right book':16} {plain_book:>5}/{n} {grounded_book:>6}/{n}")
    print(f"{'right passage':16} {'-':>8} {grounded_passage:>6}/{n}")
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False, indent=1)
        print("answers written to", args.out)


if __name__ == "__main__":
    main()
