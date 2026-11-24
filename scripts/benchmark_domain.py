"""Ask Claude the support questions of chapter 10 twice: without and with the MCP tools.

Run (it is not part of CI, and it uses your Claude Code account):

    uv run python scripts/benchmark_domain.py --model sonnet

For each question in data/domain/savings/questions.jsonl it calls the Claude
Code CLI in headless mode (`claude -p`) two times, with the same instructions:

    plain   no tools, no documents: only the question
    tools   the MCP server `homer_ai.domain_server` is connected, and nothing
            else: search_terms, get_rate, escalate

Claude must end every reply with one line: `SOURCE: <reference>`, `UNKNOWN`,
or `ESCALATE: <reason>`. We score:

    right          the reply holds the labelled key facts (questions with an answer)
    right source   ... and its SOURCE is one of the labelled references
    wrong          it states an answer that fails the key check: an invented fact
    no answer      it says UNKNOWN, or hands over a question it could have answered
    handed over    for the questions that must go to a person: it did so

In the tools run "handed over" also counts a call of the `escalate` tool, read
from the server's audit file. The data is invented (see data/domain/README.md),
so no model can know it: the plain run shows what a model does without it.

Notebook 10_01 explains the method and shows one recorded run.
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
from pathlib import Path

from benchmark_grounding import pick

from homer_ai.domain import answer_has_keys, load_support_questions

PROJECT = Path(__file__).parent.parent
SERVER = "example-bank"
TOOLS = ("search_terms", "get_rate", "escalate")

RULES = (
    "You are the support assistant of Example Bank, an invented savings firm "
    "with four accounts: Easy Access Saver, 1-Year Fixed Saver, 95-Day Notice "
    "Saver and Cash ISA Saver. Answer in one or two plain sentences.\n"
    "Do not answer, and hand over to a person instead, when the customer "
    "complains about something the bank did (in any words), shows signs of "
    "vulnerability (illness, bereavement, job loss, money trouble, difficulty "
    "reading or understanding), asks for financial advice, or reports a death.\n"
    "End every reply with exactly one last line, and write nothing after it:\n"
    "SOURCE: <reference>   if you answered from a document\n"
    "UNKNOWN               if you cannot answer\n"
    "ESCALATE: <complaint, vulnerability, advice or bereavement>   if you hand over\n"
)
PLAIN = (
    RULES + "You have no tools and no documents. If you answer, end with SOURCE: none."
)
WITH_TOOLS = RULES + (
    "Use your tools: search_terms for the product sheets and the FAQ, get_rate "
    "for interest rates, escalate to hand over. Never state a fact about "
    "Example Bank that did not come from a tool. The reference is the text in "
    "square brackets, like easy-access#4."
)
SOURCE = re.compile(r"SOURCE:\W*([a-z0-9-]+#[a-z0-9/-]+)", re.IGNORECASE)
ESCALATE = re.compile(r"^\W*ESCALATE\b", re.IGNORECASE | re.MULTILINE)
UNKNOWN = re.compile(r"^\W*UNKNOWN\b", re.IGNORECASE | re.MULTILINE)


def ask_claude(question, model, folder, mcp_config=None):
    """One headless call. Returns Claude's text ("" if the call failed)."""
    command = ["claude", "-p", "--model", model, "--tools", "", "--strict-mcp-config"]
    command += ["--no-session-persistence"]
    if mcp_config:
        allowed = ",".join(f"mcp__{SERVER}__{tool}" for tool in TOOLS)
        command += ["--mcp-config", str(mcp_config), "--allowedTools", allowed]
        command += ["--system-prompt", WITH_TOOLS]
    else:
        command += ["--system-prompt", PLAIN]
    try:
        done = subprocess.run(
            command,
            input=question,
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


def judge(question, reply, calls=()):
    """What happened, as a dict of true/false, for one question and one reply.

    `calls` are the audit lines of the tools run (empty for the plain run).
    """
    handed_over = bool(ESCALATE.search(reply)) or any(
        call["tool"] == "escalate" for call in calls
    )
    if question["kind"] == "escalate":
        return {"handed_over": handed_over}
    cited = SOURCE.findall(reply)
    unknown = bool(UNKNOWN.search(reply))
    answered = not handed_over and not unknown
    right = answered and answer_has_keys(reply, question["keys"])
    no_answer = not answered
    return {
        "right": right,
        "right_source": right and bool(cited) and cited[-1].lower() in question["refs"],
        "no_answer": no_answer,
        "wrong": not right and not no_answer,
        "handed_over": handed_over,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--limit", type=int, default=0, help="questions (0 = all)")
    parser.add_argument("--model", default="sonnet", help="model name for claude")
    parser.add_argument("--out", help="also write every reply to this JSON file")
    args = parser.parse_args()

    if shutil.which("claude") is None:
        sys.exit("The `claude` command was not found. Install Claude Code first.")

    questions = pick(load_support_questions(), args.limit)
    rows = []
    with tempfile.TemporaryDirectory() as folder:
        work = Path(folder) / "empty"
        work.mkdir()
        audit_file = Path(folder) / "audit.jsonl"
        config = Path(folder) / "mcp.json"
        server = {
            "command": "uv",
            "args": ["run", "--project", str(PROJECT), "python", "-m"]
            + ["homer_ai.domain_server"],
            "env": {"HOMER_DOMAIN_AUDIT": str(audit_file)},
        }
        config.write_text(json.dumps({"mcpServers": {SERVER: server}}))
        for number, question in enumerate(questions, start=1):
            plain = ask_claude(question["question"], args.model, work)
            audit_file.write_text("")
            tools = ask_claude(question["question"], args.model, work, config)
            calls = [json.loads(line) for line in audit_file.read_text().splitlines()]
            row = {
                "id": question["id"],
                "kind": question["kind"],
                "question": question["question"],
                "expected": question.get("answer")
                or f"hand over ({question['reason']})",
                "refs": question.get("refs"),
                "plain": plain,
                "tools": tools,
                "calls": calls,
                "plain_score": judge(question, plain),
                "tools_score": judge(question, tools, calls),
            }
            rows.append(row)
            print(f"\n[{number}/{len(questions)}] {question['question']}")
            print("  expected:", row["expected"], row["refs"] or "")
            print("  plain:   ", plain.replace("\n", " | "))
            print("  tools:   ", tools.replace("\n", " | "))
            print("  calls:   ", [call["tool"] for call in calls])
            time.sleep(1)

    answerable = [r for r in rows if r["kind"] != "escalate"]
    must_hand_over = [r for r in rows if r["kind"] == "escalate"]

    def count(group, run, key):
        return sum(r[f"{run}_score"][key] for r in group)

    print(f"\n{date.today()}  model={args.model}  questions={len(rows)}")
    print(f"{'':34} {'plain':>8} {'tools':>8}")
    n = len(answerable)
    for label, key in (
        ("right", "right"),
        ("right, with the right source", "right_source"),
        ("wrong (an invented fact)", "wrong"),
        ("no answer", "no_answer"),
        ("handed over, though answerable", "handed_over"),
    ):
        plain_n, tools_n = (count(answerable, run, key) for run in ("plain", "tools"))
        print(f"{label:34} {plain_n:>5}/{n} {tools_n:>5}/{n}")
    m = len(must_hand_over)
    plain_n, tools_n = (
        count(must_hand_over, run, "handed_over") for run in ("plain", "tools")
    )
    print(f"{'must go to a person: handed over':34} {plain_n:>5}/{m} {tools_n:>5}/{m}")
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False, indent=1)
        print("replies written to", args.out)


if __name__ == "__main__":
    main()
