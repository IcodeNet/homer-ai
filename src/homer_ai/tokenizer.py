from abc import ABC, abstractmethod


class Tokenizer(ABC):
    @abstractmethod
    def encode(self, text):
        raise NotImplementedError

    @abstractmethod
    def decode(self, ids):
        raise NotImplementedError

    @property
    @abstractmethod
    def vocab_size(self):
        raise NotImplementedError


class CharTokenizer(Tokenizer):
    """Cuts text into single characters and gives each one a number."""

    def __init__(self, text):
        # Every unique character, in a fixed order, plus two special
        # tokens at the end (so no existing ID changes):
        #   <|unk|>       = a character we have never seen
        #   <|endoftext|> = one book ends here, a new one starts
        self._set_chars(sorted(set(text)) + ["<|unk|>", "<|endoftext|>"])

    @classmethod
    def from_chars(cls, chars):
        """Rebuild a tokenizer from its saved vocabulary (`tokenizer.chars`).

        The list already holds the two special tokens at the end, so every ID
        is the same as in the tokenizer that was saved.
        """
        tokenizer = cls.__new__(cls)
        tokenizer._set_chars(list(chars))
        return tokenizer

    def _set_chars(self, chars):
        self.chars = chars
        self._vocab_size = len(self.chars)
        self.char_to_id = {ch: i for i, ch in enumerate(self.chars)}
        self.id_to_char = {i: ch for i, ch in enumerate(self.chars)}
        self.unk_id = self.char_to_id["<|unk|>"]
        self.eot_id = self.char_to_id["<|endoftext|>"]

    def encode(self, text):
        """Text -> list of numbers. Unseen characters become <|unk|>."""
        return [self.char_to_id.get(c, self.unk_id) for c in text]

    def decode(self, ids):
        """List of numbers -> text."""
        return "".join(self.id_to_char[i] for i in ids)

    @property
    def vocab_size(self):
        return self._vocab_size
