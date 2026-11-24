def test_round_trip(tokenizer):
    word = "ἄνδρα μοι ἔννεπε"
    assert tokenizer.decode(tokenizer.encode(word)) == word


def test_vocabulary_on_book_1(tokenizer):
    assert tokenizer.vocab_size == 137


def test_special_token_ids(tokenizer):
    assert tokenizer.unk_id == 135
    assert tokenizer.eot_id == 136


def test_unknown_character_becomes_unk(tokenizer):
    assert tokenizer.encode("!") == [135]
    assert tokenizer.encode("ἄ!") == [tokenizer.encode("ἄ")[0], 135]


def test_empty_text(tokenizer):
    assert tokenizer.encode("") == []
    assert tokenizer.decode([]) == ""
