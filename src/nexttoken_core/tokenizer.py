"""Tokenization helpers.

Two tokenizers live here:

* :class:`EnumeratingTokenizer` -- the original text tokenizer. It enumerates
  the *sorted unique symbols* of a string/iterable and maps them to ids.
  Kept exactly as it was so the text example keeps working.

* :class:`Tokenizer` -- the tokenizer used by the universal Markov core. It
  maps *any* hashable Python object (a letter, a word, a MIDI note tuple, a
  special token, ...) to an integer id ``0..V-1`` and can map ids back. It
  preserves insertion order and never sorts, so it also works for tokens that
  are not mutually sortable (e.g. ``(60, 4)`` mixed with ``"BOS"``).
"""

from tqdm import tqdm


class EnumeratingTokenizer:
    def __init__(self, text):
        self.text = text

    def tokenize_text_by_enumerating(self):
        self.unique_symbols = sorted(set(self.text))
        word_to_tokode = dict([(y, x) for (x, y) in tqdm(enumerate(self.unique_symbols), total=len(self.unique_symbols))])
        text_digitized = [word_to_tokode[x] for x in tqdm(self.text)]
        return text_digitized


class Tokenizer:
    """Reversible ``hashable token <-> integer id`` mapping.

    Ids run from ``0`` to ``len(tokenizer) - 1``. Use :meth:`encode` /
    :meth:`decode` to go back and forth.
    """

    def __init__(self, tokens=()):
        self.token_to_id = {}
        self.id_to_token = []
        for token in tokens:
            self.add(token)

    def add(self, token):
        if token not in self.token_to_id:
            self.token_to_id[token] = len(self.id_to_token)
            self.id_to_token.append(token)
        return self.token_to_id[token]

    def encode(self, tokens):
        return [self.token_to_id[token] for token in tokens]

    def decode(self, ids):
        return [self.id_to_token[i] for i in ids]

    @property
    def vocabulary(self):
        return list(self.id_to_token)

    def __contains__(self, token):
        return token in self.token_to_id

    def __len__(self):
        return len(self.id_to_token)
