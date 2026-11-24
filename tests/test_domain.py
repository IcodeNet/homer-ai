"""Tests of the chapter 10 code (domain.py, domain_server.py) and its invented data.

Nothing here downloads anything or loads a model.
"""

import asyncio
import hashlib
import importlib.util
import json
from datetime import date
from pathlib import Path

import pytest

from homer_ai import domain, domain_server

SCRIPT = Path(__file__).parent.parent / "scripts" / "make_domain_data.py"


@pytest.fixture(scope="module")
def generator():
    """scripts/make_domain_data.py, loaded as a module (scripts/ is not a package)."""
    spec = importlib.util.spec_from_file_location("make_domain_data", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------- savings: documents
def test_every_sheet_is_cut_into_numbered_sections_with_a_version():
    sections = domain.load_sections()
    assert len(sections) == 39
    assert {s.doc for s in sections} == {*domain.PRODUCTS, "faq"}
    for doc in {s.doc for s in sections}:
        numbers = [s.number for s in sections if s.doc == doc]
        assert numbers == list(range(1, len(numbers) + 1))
    first = sections[0]
    assert first.ref == "easy-access#1"
    assert first.stamp == "version 3, in force from 2026-09-01"
    assert all(date.fromisoformat(s.in_force_from) for s in sections)


def test_every_sheet_says_it_is_invented():
    for name in domain.SHEETS:
        text = (domain.SAVINGS / name).read_text(encoding="utf-8")
        assert "invented sample data" in text


def test_the_sheets_hold_no_rates():
    # rates live in rates.csv only, so there is one place to change them
    for section in domain.load_sections():
        assert "%" not in section.text


def test_search_terms_finds_the_section_and_respects_k():
    found = domain.search_terms("Easy Access withdrawal at 4pm, when does it arrive?")
    assert len(found) == 3
    assert found[0][0].ref == "easy-access#4"
    assert found[0][1] >= found[1][1] >= found[2][1]
    assert len(domain.search_terms("withdrawals", k=1)) == 1
    assert domain.search_terms("zzzz") == []


def test_sheet_titles_in_the_index_help_the_search():
    questions = [q for q in domain.load_support_questions() if q["kind"] == "terms"]

    def hits(with_titles, k):
        return sum(
            any(
                s.ref in q["refs"]
                for s, _ in domain.search_terms(q["question"], k, with_titles)
            )
            for q in questions
        )

    # the numbers of notebook 10_01
    assert (hits(True, 1), hits(True, 3)) == (15, 19)
    assert (hits(False, 1), hits(False, 3)) == (10, 15)


# ---------------------------------------------------------------- savings: rates
def test_get_rate_picks_the_row_in_force_on_that_day():
    assert domain.get_rate("easy-access", "2026-08-31").aer == 3.10
    assert (
        domain.get_rate("easy-access", "2026-09-01").aer == 2.85
    )  # the day it changes
    assert domain.get_rate("easy-access", date(2027, 1, 1)).aer == 2.85
    rate = domain.get_rate("fixed-1-year", "2026-08-20")
    assert (rate.aer, rate.ref) == (4.05, "rates#fixed-1-year/2026-08-15")


def test_get_rate_never_guesses():
    with pytest.raises(LookupError, match="no rate on file"):
        domain.get_rate("cash-isa", "2026-01-01")  # before the first row
    with pytest.raises(ValueError, match="unknown product"):
        domain.get_rate("current-account", "2026-10-01")
    with pytest.raises(ValueError, match="not a date"):
        domain.get_rate("easy-access", "1 October")


def test_every_rate_question_is_answered_by_the_table():
    for q in domain.load_support_questions():
        if q["kind"] == "rate":
            rate = domain.get_rate(q["product"], q["date"])
            assert [rate.ref] == q["refs"]
            assert domain.answer_has_keys(f"{rate.aer:.2f}%", q["keys"])


# ---------------------------------------------------------------- savings: handing over
def test_escalate_returns_a_handover_and_rejects_other_reasons():
    handover = domain.escalate("bereavement")
    assert handover["queue"] == "bereavement team"
    assert "colleague" in handover["say"]
    with pytest.raises(ValueError, match="unknown reason"):
        domain.escalate("boredom")


def test_needs_person_catches_the_plain_cases():
    assert domain.needs_person("I want to complain about my statement") == "complaint"
    assert domain.needs_person("My mother passed away last week") == "bereavement"
    assert domain.needs_person("Should I pick the ISA?") == "advice"
    assert domain.needs_person("I lost my job in May") == "vulnerability"
    assert domain.needs_person("How often is interest paid?") is None


def test_the_guard_on_the_labelled_questions():
    questions = domain.load_support_questions()
    must = [q for q in questions if q["kind"] == "escalate"]
    others = [q for q in questions if q["kind"] != "escalate"]
    caught = [q for q in must if domain.needs_person(q["question"])]
    # the numbers of notebook 10_01: 7 of 10 caught, no false alarm
    assert (len(must), len(caught)) == (10, 7)
    assert all(domain.needs_person(q["question"]) == q["reason"] for q in caught)
    assert not any(domain.needs_person(q["question"]) for q in others)


def test_support_questions_are_consistent():
    questions = domain.load_support_questions()
    assert [q["id"] for q in questions] == list(range(1, 40))
    refs = {s.ref for s in domain.load_sections()} | {
        r.ref for r in domain.load_rates()
    }
    kinds = {"terms": 0, "rate": 0, "escalate": 0}
    for q in questions:
        kinds[q["kind"]] += 1
        if q["kind"] == "escalate":
            assert q["reason"] in domain.QUEUES
        else:
            assert q["refs"] and set(q["refs"]) <= refs
            assert q["keys"] and all(isinstance(group, list) for group in q["keys"])
    assert kinds == {"terms": 22, "rate": 7, "escalate": 10}


def test_answer_has_keys():
    assert domain.answer_has_keys("The minimum is £1,000.", [["1000"]])
    assert not domain.answer_has_keys("The minimum is £100.", [["1000"]])
    assert domain.answer_has_keys("From 9 am until 1pm", [["9am", "9 am"], ["1pm"]])
    assert not domain.answer_has_keys("From 9am", [["9am", "9 am"], ["1pm"]])


def test_audit_line_and_fingerprint():
    record = json.loads(domain.audit_line("get_rate", product="cash-isa"))
    assert record["tool"] == "get_rate"
    assert record["product"] == "cash-isa"
    assert record["data"] == domain.data_fingerprint()
    assert len(record["data"]) == 12
    assert record["at"].endswith("+00:00")


# ---------------------------------------------------------------- the MCP server
def test_server_lists_three_tools_with_honest_descriptions():
    tools = {tool.name: tool for tool in asyncio.run(domain_server.mcp.list_tools())}
    assert list(tools) == ["search_terms", "get_rate", "escalate"]
    assert "not meaning" in tools["search_terms"].description
    assert "invented" in tools["search_terms"].description
    assert tools["get_rate"].input_schema["properties"]["product"]["enum"] == list(
        domain.PRODUCTS
    )
    assert tools["escalate"].input_schema["properties"]["reason"]["enum"] == list(
        domain.QUEUES
    )
    assert "When in doubt, escalate" in tools["escalate"].description


def test_server_tools_return_text_with_references(capsys):
    text = domain_server.search_terms("Easy Access withdrawal at 4pm", k=1)
    assert text.startswith("[easy-access#4] (version 3, in force from 2026-09-01)")
    assert domain_server.search_terms("withdrawals", k=99).count("\n\n") == 4  # 5
    rate = domain_server.get_rate("easy-access", "2026-08-01")
    assert rate == (
        "easy-access: 3.10% AER on 2026-08-01 (in force from 2026-03-01) "
        "[rates#easy-access/2026-03-01]"
    )
    assert domain_server.get_rate("cash-isa", "2026-01-01").startswith("No rate:")
    assert domain_server.get_rate("cash-isa", "soon").startswith("No rate:")
    assert "Do not answer" in domain_server.escalate("complaint")
    captured = capsys.readouterr()
    assert captured.out == ""  # stdout belongs to the MCP messages
    assert captured.err.count("domain: {") == 6


def test_server_writes_the_audit_file_without_the_customers_words(
    monkeypatch, tmp_path
):
    path = tmp_path / "audit.jsonl"
    monkeypatch.setenv("HOMER_DOMAIN_AUDIT", str(path))
    domain_server.search_terms("my secret question about the Cash ISA")
    domain_server.escalate("vulnerability")
    lines = [json.loads(line) for line in path.read_text().splitlines()]
    assert [line["tool"] for line in lines] == ["search_terms", "escalate"]
    assert lines[0]["query_chars"] == 37
    assert lines[0]["refs"][0].startswith("cash-isa#")
    assert "secret" not in path.read_text()


# ---------------------------------------------------------------- emails
def test_generator_is_deterministic_and_matches_the_committed_files(generator):
    def sha(text):
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    emails = generator.to_jsonl(generator.make_emails())
    tickets = generator.to_jsonl(generator.make_tickets())
    assert sha(emails) == sha(generator.to_jsonl(generator.make_emails()))
    assert sha(tickets) == sha(generator.to_jsonl(generator.make_tickets()))
    assert sha(emails) != sha(generator.to_jsonl(generator.make_emails(seed=11)))
    assert sha(emails) == hashlib.sha256(generator.EMAILS_FILE.read_bytes()).hexdigest()
    assert (
        sha(tickets) == hashlib.sha256(generator.TICKETS_FILE.read_bytes()).hexdigest()
    )


def test_emails_load_and_their_labels_are_consistent():
    emails = domain.load_emails()
    assert len(emails) == 400
    assert len({e["id"] for e in emails}) == 400
    assert len({e["text"] for e in emails}) == 400
    sizes = {s: len(domain.load_emails(split=s)) for s in ("train", "val", "test")}
    assert sizes == {"train": 280, "val": 40, "test": 80}
    for e in emails:
        assert isinstance(e["complaint"], bool) and isinstance(e["vulnerable"], bool)
        assert (e["styles"]["complaint"] != "none") == e["complaint"]
        assert (e["styles"]["vulnerable"] != "none") == e["vulnerable"]
        assert not (e["styles"]["tricky"] and (e["complaint"] or e["vulnerable"]))
    for split in sizes:  # every class is in every split
        classes = {domain.email_class(e) for e in domain.load_emails(split=split)}
        assert classes == {0, 1, 2, 3}


def test_test_emails_use_wordings_the_training_emails_never_saw(generator):
    train = " ".join(e["text"].casefold() for e in domain.load_emails(split="train"))
    test = " ".join(e["text"].casefold() for e in domain.load_emails(split="test"))
    for sentence in generator.VULNERABLE + generator.VULNERABLE_SOFT:
        if "{" not in sentence:
            assert not (sentence.casefold() in train and sentence.casefold() in test)


def test_handwritten_emails():
    emails = domain.load_emails("handwritten")
    assert len(emails) == 24
    counts = [0, 0, 0, 0]
    for e in emails:
        counts[domain.email_class(e)] += 1
    assert counts == [6, 6, 6, 6]


def test_email_class_round_trip():
    for number, name in enumerate(domain.EMAIL_CLASSES):
        complaint, vulnerable = domain.flags_of_class(number)
        email = {"complaint": complaint, "vulnerable": vulnerable}
        assert domain.email_class(email) == number
        assert (name in ("complaint", "both")) == complaint


def test_keyword_flags_and_what_they_miss():
    assert domain.keyword_flags("I want to COMPLAIN.") == (True, False)
    assert domain.keyword_flags("My husband passed away.") == (False, True)
    # dissatisfaction with no trigger word is missed ...
    assert domain.keyword_flags("I have now asked three times.") == (False, False)
    # ... and a trigger word in a happy email is a false alarm
    assert domain.keyword_flags("No complaints from me!") == (True, False)


def test_keyword_baseline_scores_on_the_test_emails():
    emails = domain.load_emails(split="test")
    guesses = [domain.keyword_flags(e["text"]) for e in emails]
    complaint = domain.recall([e["complaint"] for e in emails], [g[0] for g in guesses])
    vulnerable = domain.recall(
        [e["vulnerable"] for e in emails], [g[1] for g in guesses]
    )
    # the numbers of notebook 10_02
    assert round(complaint, 3) == 0.513
    assert round(vulnerable, 3) == 0.591


# ---------------------------------------------------------------- scores
def test_recall_precision_f1_by_hand():
    truth = [True, True, True, True, False, False, False, False]
    guess = [True, False, True, True, True, False, False, False]
    assert domain.recall(truth, guess) == 0.75  # caught 3 of the 4 real ones
    assert domain.precision(truth, guess) == 0.75  # 3 of the 4 flags were right
    assert domain.f1(truth, guess) == 0.75
    assert domain.recall(truth, [True] * 8) == 1.0  # flag everything: recall 1 ...
    assert domain.precision(truth, [True] * 8) == 0.5  # ... and precision pays
    assert domain.f1(truth, [False] * 8) == 0.0
    assert domain.recall([False, False], [False, True]) == 0.0  # nothing to catch


def test_macro_f1_gives_every_class_the_same_weight():
    truth = ["a"] * 8 + ["b"] * 2
    always_a = ["a"] * 10
    # accuracy would be 0.8; class b is never found, so its F1 is 0
    f1_a = 2 * 0.8 * 1.0 / (0.8 + 1.0)
    assert domain.macro_f1(truth, always_a, ["a", "b"]) == pytest.approx(f1_a / 2)
    assert domain.macro_f1(truth, truth, ["a", "b"]) == 1.0


def test_confusion_rows_are_the_truth():
    table = domain.confusion(["a", "a", "b", "b"], ["a", "b", "b", "b"], ["a", "b"])
    assert table == [[1, 1], [0, 2]]
    assert sum(map(sum, table)) == 4


# ---------------------------------------------------------------- service desk
def test_kb_and_tickets_load_and_agree():
    articles = domain.load_kb()
    assert [a.id for a in articles] == [f"kb-{n:02d}" for n in range(1, 11)]
    assert all(date.fromisoformat(a.reviewed) and a.text for a in articles)
    tickets = domain.load_tickets()
    assert len(tickets) == 200
    assert len(domain.load_tickets("train")) == 150
    assert len(domain.load_tickets("test")) == 50
    category_of = {a.id: a.category for a in articles}
    for ticket in tickets:
        assert ticket["category"] in set(category_of.values())
        if ticket["kb"] is not None:  # a ticket's article is in its own category
            assert category_of[ticket["kb"]] == ticket["category"]
    assert sum(t["kb"] is None for t in tickets) > 50  # many have no article


def test_suggest_articles():
    found = domain.suggest_articles("VPN won't connect from home", k=3)
    assert found[0][0].id == "kb-03"
    assert len(found) == 3
    assert domain.suggest_articles("zzzz") == []


def test_router_votes_and_admits_a_guess():
    solved = [
        {"text": "printer shows offline", "category": "hardware"},
        {"text": "printer queue is stuck", "category": "hardware"},
        {"text": "vpn will not connect", "category": "network"},
    ]
    router = domain.TicketRouter(solved, k=2)
    category, votes = router.route("the printer is offline again")
    assert category == "hardware"
    assert set(votes) == {"hardware"}
    assert router.route("zzzz") == ("hardware", {})  # no shared word: most common


def test_routing_scores_on_the_test_tickets():
    train, test = domain.load_tickets("train"), domain.load_tickets("test")

    def right(router):
        return sum(router.route(t["text"])[0] == t["category"] for t in test)

    # the numbers of notebook 10_03
    assert right(domain.TicketRouter(train)) == 17
    assert right(domain.ProfileRouter(train)) == 24
    assert right(domain.ProfileRouter(train, domain.load_kb())) == 30


def test_article_suggestions_on_the_test_tickets():
    with_article = [t for t in domain.load_tickets("test") if t["kb"]]
    found = [
        [a.id for a, _ in domain.suggest_articles(t["text"], k=3)] for t in with_article
    ]
    first = sum(
        ids[:1] == [t["kb"]] for ids, t in zip(found, with_article, strict=True)
    )
    top3 = sum(t["kb"] in ids for ids, t in zip(found, with_article, strict=True))
    assert (len(with_article), first, top3) == (26, 17, 22)  # notebook 10_03


# ---------------------------------------------------------------- benchmark scoring
@pytest.fixture(scope="module")
def benchmark(monkeypatch_module):
    """scripts/benchmark_domain.py as a module; importing it calls nothing."""
    monkeypatch_module.syspath_prepend(str(SCRIPT.parent))
    return importlib.import_module("benchmark_domain")


@pytest.fixture(scope="module")
def monkeypatch_module():
    patch = pytest.MonkeyPatch()
    yield patch
    patch.undo()


def test_benchmark_judges_an_answer(benchmark):
    question = domain.load_support_questions()[0]  # smallest deposit: £1,000
    right = benchmark.judge(question, "It is £1,000.\n\nSOURCE: fixed-1-year#3")
    assert right["right"] and right["right_source"] and not right["wrong"]
    other_source = benchmark.judge(question, "It is £1,000.\nSOURCE: [faq#1]")
    assert other_source["right"] and not other_source["right_source"]
    invented = benchmark.judge(question, "It is £500.\nSOURCE: none")
    assert invented["wrong"] and not invented["no_answer"]
    unknown = benchmark.judge(question, "I can't say; some banks ask £1,000.\nUNKNOWN")
    assert unknown["no_answer"] and not unknown["right"] and not unknown["wrong"]


def test_benchmark_judges_a_handover(benchmark):
    must = next(q for q in domain.load_support_questions() if q["kind"] == "escalate")
    assert benchmark.judge(must, "I'm passing this on.\nESCALATE: complaint") == {
        "handed_over": True
    }
    assert not benchmark.judge(must, "Sorry about that.\nSOURCE: faq#5")["handed_over"]
    # in the tools run, a call of the escalate tool counts too
    calls = [{"tool": "escalate", "reason": "complaint"}]
    assert benchmark.judge(must, "A colleague will reply.", calls)["handed_over"]
    answerable = domain.load_support_questions()[0]
    assert benchmark.judge(answerable, "ESCALATE: advice")["no_answer"]
