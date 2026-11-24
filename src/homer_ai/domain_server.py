"""An MCP server for the support desk of an invented savings firm (chapter 10).

Three tools over the made-up data in `data/domain/savings/`:

    search_terms   find sections of the product sheets, each with its reference
    get_rate       the exact interest rate of a product on a date, from a table
    escalate       hand the conversation to a person, and stop answering

Start it with `uv run python -m homer_ai.domain_server`. It talks over
stdin/stdout, so logs go to stderr. If the environment variable
`HOMER_DOMAIN_AUDIT` names a file, one line per tool call is also added to that
file: an audit trail. It needs no model file.

Everything it serves is invented. "Example Bank" does not exist.
"""

import os
import sys
from datetime import date
from typing import Literal

from mcp.server.mcpserver import MCPServer

from homer_ai import domain

MIN_K, MAX_K = 1, 5  # how many sections one search may ask for

mcp = MCPServer("ExampleBankSupport")


def audit(tool, **fields):
    """Write one audit line to stderr, and to the file in $HOMER_DOMAIN_AUDIT."""
    line = domain.audit_line(tool, **fields)
    # stderr, never stdout: stdout is the channel to the MCP client
    print(f"domain: {line}", file=sys.stderr)
    path = os.environ.get("HOMER_DOMAIN_AUDIT")
    if path:
        with open(path, "a", encoding="utf-8") as f:
            f.write(line + "\n")


def clamp_k(k):
    """Keep the number of sections between MIN_K and MAX_K."""
    return max(MIN_K, min(MAX_K, k))


@mcp.tool()
def search_terms(query: str, k: int = 3) -> str:
    """
    Find sections of Example Bank's product sheets and FAQ that match a
    question. Example Bank is an invented firm with four savings accounts: Easy
    Access Saver, 1-Year Fixed Saver, 95-Day Notice Saver and Cash ISA Saver.
    Each result starts with its reference, like [easy-access#4], and the
    version and date of the sheet it comes from. Quote the reference in your
    answer.

    The search matches words (BM25), not meaning. Name the product in the
    query, and use the words a product sheet would use. The sheets hold no
    interest rates: use get_rate for those. If no section answers the question,
    say so; do not answer from memory.

    query: what to look for, in plain English.
    k: how many sections to return, from 1 to 5 (default 3); values outside
    that range are clamped.
    """
    k = clamp_k(k)
    found = domain.search_terms(query, k)
    audit("search_terms", query_chars=len(query), k=k, refs=[s.ref for s, _ in found])
    if not found:
        return "No section shares a word with this query."
    return "\n\n".join(
        f"[{s.ref}] ({s.stamp}) {s.doc_title}: {s.title}\n{s.text}" for s, _ in found
    )


@mcp.tool()
def get_rate(
    product: Literal["easy-access", "fixed-1-year", "notice-95", "cash-isa"],
    on: str = "",
) -> str:
    """
    The exact interest rate (AER, in percent a year) of one Example Bank
    product on one day, read from the rates table. Rates change, so always ask
    for the day the customer means: for a fixed saver that is the day the
    account was opened. The result ends with its reference, like
    [rates#easy-access/2026-09-01]. Quote it.

    Never state a rate that did not come from this tool.

    product: "easy-access", "fixed-1-year", "notice-95" or "cash-isa".
    on: the day, written YYYY-MM-DD. Leave it empty for today.
    """
    day = on.strip() or date.today().isoformat()
    try:
        rate = domain.get_rate(product, day)
    except (ValueError, LookupError) as error:
        audit("get_rate", product=product, on=day, found=None)
        return f"No rate: {error}."
    audit("get_rate", product=product, on=day, found=rate.ref)
    return (
        f"{product}: {rate.aer:.2f}% AER on {day} "
        f"(in force from {rate.in_force_from.isoformat()}) [{rate.ref}]"
    )


@mcp.tool()
def escalate(
    reason: Literal["complaint", "vulnerability", "advice", "bereavement"],
) -> str:
    """
    Hand the conversation to a member of staff. Call this BEFORE answering, and
    do not answer the question yourself, when the customer:

    - is unhappy with something Example Bank did or failed to do, in any words
      (complaint). They do not have to say "complaint".
    - mentions illness, a death in the family, losing a job, money trouble,
      disability, or trouble reading or understanding (vulnerability).
    - asks what they should do with their money or which account is best for
      them (advice).
    - reports that an account holder has died (bereavement).

    When in doubt, escalate. A person can hand an easy case back; nobody can
    un-send a wrong answer.

    reason: "complaint", "vulnerability", "advice" or "bereavement".
    """
    handover = domain.escalate(reason)
    audit("escalate", reason=reason, queue=handover["queue"])
    return (
        f"Handed to the {handover['queue']}. Do not answer the question. "
        f"Tell the customer: {handover['say']}"
    )


def main():
    """Serve the three tools over stdio. The first search builds the index."""
    print(
        f"domain: serving data/domain/savings, fingerprint {domain.data_fingerprint()}",
        file=sys.stderr,
    )
    mcp.run()


if __name__ == "__main__":
    main()
