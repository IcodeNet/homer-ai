"""Tests of the chapter 9 search code (src/homer_ai/retrieval.py) and its data."""

import math

import pytest

from homer_ai import retrieval

BM25, Passage = retrieval.BM25, retrieval.Passage


def test_chunk_lines_counts_from_one_and_keeps_the_rest():
    lines = [f"line {n}" for n in range(1, 13)]
    chunks = retrieval.chunk_lines(lines, 5)
    assert [(first, last) for first, last, _ in chunks] == [(1, 5), (6, 10), (11, 12)]
    assert chunks[2][2] == "line 11\nline 12"


def test_passages_never_cross_a_book_and_cover_every_line():
    for poem in retrieval.POEMS:
        passages = retrieval.load_passages(poem, "greek")
        for book in range(1, 25):
            path = retrieval.HOMER / poem / "greek" / f"book_{book:02d}.txt"
            lines = path.read_text(encoding="utf-8").splitlines()
            mine = [p for p in passages if p.book == book]
            assert mine[0].first == 1
            assert mine[-1].last == len(lines)
            assert "\n".join(p.text for p in mine) == "\n".join(lines)
            assert all(p.last - p.first + 1 <= 5 for p in mine)


def test_english_passage_is_one_line_of_the_file():
    passages = retrieval.load_passages("odyssey", "english")
    assert len(passages) == 1027  # the paragraph count in data/homer/README.md
    first = passages[0]
    assert first.ref == "odyssey 1:1"
    assert first.text.startswith("Tell me, O Muse")


def test_ref_formats():
    assert Passage("iliad", "greek", 1, 1, 5, "").ref == "iliad 1:1-5"
    assert Passage("odyssey", "english", 9, 24, 24, "").ref == "odyssey 9:24"


def test_tokenize_lowercases_and_strips_accents():
    assert retrieval.tokenize("Μῆνιν ἄειδε, θεά!") == ["μηνιν", "αειδε", "θεα"]
    assert retrieval.tokenize("Noman’s comrades, 3 of them") == [
        "noman",
        "s",
        "comrades",
        "of",
        "them",
    ]
    assert retrieval.tokenize("δʼ ὅ γʼ") == ["δ", "ο", "γ"]


def test_expand_names_adds_butlers_name():
    assert retrieval.expand_names(["odysseus", "and", "zeus"]) == [
        "odysseus",
        "ulysses",
        "and",
        "zeus",
        "jove",
    ]


def test_bm25_matches_the_formula_by_hand():
    texts = ["noman noman noman eats", "noman is killing me", "pigs eat acorns here"]
    index = BM25(texts, k1=1.5, b=0.75)
    assert index.average == 4
    idf = math.log(1 + (3 - 2 + 0.5) / (2 + 0.5))  # "noman" is in 2 of 3 texts
    assert index.idf("noman") == pytest.approx(idf)
    # every text has 4 words, so the length part is exactly 1
    expected = idf * 3 * 2.5 / (3 + 1.5)
    assert index.scores(["noman"])[0] == pytest.approx(expected)
    assert index.scores(["zeus"]) == {}
    assert index.idf("pigs") > index.idf("noman")


def test_bm25_search_order_and_ties():
    index = BM25(["a b", "a b", "a a b", "c"])
    assert [n for n, _ in index.search(["a"], k=3)] == [2, 0, 1]  # tie: earlier first
    assert index.search(["x"], k=3) == []
    # a repeated question word counts once
    assert index.scores(["a", "a"]) == index.scores(["a"])


def test_search_is_deterministic_and_points_to_real_lines():
    first = retrieval.search("What false name does Odysseus give to the Cyclops?")
    again = retrieval.search("What false name does Odysseus give to the Cyclops?")
    assert first == again
    assert len(first) == 5
    best = first[0][0]
    assert best.ref == "odyssey 9:24"
    assert "my name is Noman" in best.text
    for passage, _score in first:
        path = (
            retrieval.HOMER / passage.poem / "english" / f"book_{passage.book:02d}.txt"
        )
        lines = path.read_text(encoding="utf-8").splitlines()
        assert lines[passage.first - 1] == passage.text
    scores = [score for _passage, score in first]
    assert scores == sorted(scores, reverse=True)


def test_search_one_poem_and_greek():
    only_iliad = retrieval.search("Achilles", poem="iliad", k=10)
    assert {p.poem for p, _ in only_iliad} == {"iliad"}
    greek = retrieval.search("μηνιν αειδε θεα", poem="iliad", language="greek", k=1)
    assert greek[0][0].ref == "iliad 1:1-5"
    assert greek[0][0].text.startswith("μῆνιν ἄειδε θεὰ")


def test_search_rejects_unknown_poem_or_language():
    with pytest.raises(ValueError, match="poem"):
        retrieval.search("x", poem="aeneid")
    with pytest.raises(ValueError, match="language"):
        retrieval.search("x", language="latin")


def test_index_is_built_once():
    assert retrieval.get_index("odyssey", "english") is retrieval.get_index(
        "odyssey", "english"
    )


def test_every_labelled_question_points_at_real_lines():
    questions = retrieval.load_questions()
    assert 25 <= len(questions) <= 40
    assert [q["id"] for q in questions] == list(range(1, len(questions) + 1))
    for q in questions:
        assert q["poem"] in retrieval.POEMS
        assert q["question"].endswith("?") and q["answer"]
        for book, first, last in q["refs"]:
            path = retrieval.HOMER / q["poem"] / "english" / f"book_{book:02d}.txt"
            n_lines = len(path.read_text(encoding="utf-8").splitlines())
            assert 1 <= first <= last <= n_lines, (q["id"], book, first, last)


def test_hits_and_scores():
    question = {"poem": "odyssey", "refs": [[9, 24, 25], [9, 28, 28]]}
    hit = Passage("odyssey", "english", 9, 25, 25, "")
    assert retrieval.is_hit(hit, question)
    assert not retrieval.is_hit(hit._replace(book=10), question)
    assert not retrieval.is_hit(hit._replace(poem="iliad"), question)
    assert not retrieval.is_hit(hit._replace(first=26, last=27), question)
    # a Greek passage of several lines hits if it overlaps the range
    assert retrieval.is_hit(hit._replace(first=21, last=25), question)
    miss = hit._replace(book=1)
    assert retrieval.first_hit_rank([miss, miss, hit], question) == 3
    assert retrieval.first_hit_rank([miss], question) is None

    scores = retrieval.retrieval_scores([1, 3, None, 2])
    assert scores["recall@1"] == 0.25
    assert scores["recall@5"] == 0.75
    assert scores["mrr"] == pytest.approx((1 + 1 / 3 + 0 + 1 / 2) / 4)


def test_bm25_score_on_the_labelled_questions_stays_put():
    """The numbers notebook 09_02 reports; a change to the search shows up here."""
    questions = retrieval.load_questions()
    ranks = []
    for q in questions:
        found = retrieval.search(q["question"], k=10)
        ranks.append(retrieval.first_hit_rank([p for p, _ in found], q))
    scores = retrieval.retrieval_scores(ranks)
    assert scores["recall@1"] == pytest.approx(0.35)
    assert scores["recall@5"] == pytest.approx(0.55)
    assert scores["recall@10"] == pytest.approx(0.575)
    assert scores["mrr"] == pytest.approx(0.420, abs=0.001)
