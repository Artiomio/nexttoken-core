"""Universal n-th order Markov chain over arbitrary hashable tokens.

This module deliberately knows nothing about text, words, letters, MIDI notes
or tuples. A token may be *any* hashable Python object and a training example
is just a sequence of such tokens. Text-specific loading/tokenization lives in
:mod:`nexttoken_core.tokenizer` and in ``examples/text_generation.py``.

The counting step and the ``np.random.choice`` sampling step are the same idea
that was originally inlined in ``markov_word_and_letterwise.py``.
"""

from __future__ import annotations

import time

import numpy as np

from .sparsematrix import SparseMatrix
from .tokenizer import Tokenizer


def calculate_transition_matrix(digitized_sequence, markov_dim, len_uniques=None):
    """Count ``(markov_dim + 1)``-grams of a single already-digitized sequence.

    This is the historical function from the text example, kept for backwards
    compatibility. Unlike the original code it defaults to ``max(...) + 1`` (or
    ``len(vocabulary)`` when given), so the last token of the vocabulary is not
    silently dropped from ``prob_matrix.shape``.
    """
    if len_uniques is None:
        len_uniques = max(digitized_sequence) + 1 if len(digitized_sequence) else 0

    prob_matrix = SparseMatrix([len_uniques] * (markov_dim + 1))
    for i in range(markov_dim, len(digitized_sequence)):
        n_gram_and_char = tuple(digitized_sequence[i - markov_dim : i + 1])
        prob_matrix[n_gram_and_char] += 1

    return prob_matrix


class MarkovChain:
    """An ``order``-th order Markov chain over hashable tokens.

    Example
    -------
    >>> model = MarkovChain(order=1, random_seed=42)
    >>> model.fit([["a", "b"], ["a", "c"]])
    >>> model.generate(initial_context=["a"], max_tokens=5)
    [...]

    Notes
    -----
    * Each element of ``sequences`` is treated as an independent work: the
      transition from the last token of one sequence to the first token of the
      next one is *never* counted.
    * ``next_distribution`` returns ``None`` for a *dead end* (a context that
      never occurred in the training data, or whose outgoing mass is zero).
    * ``dead_end_strategy="restart"`` jumps to a random context that is known
      to have a continuation in the training data. This is a restart, **not**
      a backoff: the model does not build statistics for shorter contexts.
    """

    def __init__(self, order=3, random_seed=None, distr_offset=0):
        if order < 1:
            raise ValueError("order must be >= 1, got %r" % (order,))
        self.order = int(order)
        self.random_seed = random_seed
        self.distr_offset = distr_offset
        self._rng = np.random.default_rng(random_seed)

        self.tokenizer_ = None
        self.prob_matrix = None
        self.vocabulary_size = 0
        self.unique_contexts = set()
        self.observed_token_contexts = []
        self.fit_time = 0.0

    # ------------------------------------------------------------------ fit
    def fit(self, sequences):
        """Learn counts from an iterable of independent token sequences."""
        start = time.time()
        sequences = [list(sequence) for sequence in sequences]

        tokenizer = Tokenizer()
        encoded_sequences = []
        for sequence in sequences:
            encoded_sequences.append([tokenizer.add(token) for token in sequence])

        self.tokenizer_ = tokenizer
        self.vocabulary_size = len(tokenizer)

        self.prob_matrix = SparseMatrix([self.vocabulary_size] * (self.order + 1))
        self.unique_contexts = set()
        for ids in encoded_sequences:
            for i in range(self.order, len(ids)):
                context = tuple(ids[i - self.order : i])
                self.prob_matrix[context + (ids[i],)] += 1
                self.unique_contexts.add(context)

        # Deterministic order -> identical random restarts for identical seeds.
        self.observed_token_contexts = [
            tuple(tokenizer.id_to_token[i] for i in context)
            for context in sorted(self.unique_contexts)
        ]

        self.fit_time = time.time() - start
        return self

    # -------------------------------------------------------------- helpers
    @property
    def vocabulary(self):
        return self.tokenizer_.vocabulary if self.tokenizer_ is not None else []

    def _context_ids(self, context):
        if self.tokenizer_ is None:
            raise RuntimeError("fit() must be called before using the model")
        ids = []
        for token in context:
            if token not in self.tokenizer_.token_to_id:
                return None
            ids.append(self.tokenizer_.token_to_id[token])
        return ids

    def next_distribution(self, context):
        """Probability vector over :attr:`vocabulary`, or ``None`` for a dead end.

        Only the last ``order`` tokens of ``context`` are used. The returned
        array is aligned with :attr:`vocabulary`, i.e. ``distribution[i]`` is
        ``P(vocabulary[i] | context)``.
        """
        ids = self._context_ids(context)
        if ids is None:
            return None
        if len(ids) < self.order:
            return None
        prefix = tuple(ids[-self.order :])

        distr = np.array(
            [self.prob_matrix[prefix + (x,)] for x in range(self.vocabulary_size)],
            dtype=float,
        )

        # `distr_offset` raises every *observed* count above zero and then zeroes
        # everything that is still <= the offset. It only ever redistributes mass
        # among transitions already seen in the training data, so it is a peak
        # flattening threshold, NOT add-k smoothing.
        distr += self.distr_offset
        distr[distr <= self.distr_offset] = 0

        total = distr.sum()
        if total <= 0:
            return None
        return distr / total

    def is_dead_end(self, context):
        return self.next_distribution(context) is None

    def _random_observed_context(self):
        if not self.observed_token_contexts:
            raise RuntimeError("no observed contexts; did you fit on empty sequences?")
        index = int(self._rng.integers(len(self.observed_token_contexts)))
        return list(self.observed_token_contexts[index])

    # ------------------------------------------------------------- generate
    def generate(
        self,
        initial_context=None,
        max_tokens=200,
        dead_end_strategy="restart",
        stop_tokens=None,
        min_tokens=0,
    ):
        """Generate up to ``max_tokens`` tokens.

        Parameters
        ----------
        initial_context:
            Sequence of tokens to start from (only the last ``order`` matter).
            If ``None`` a random context observed in training data is used.
        max_tokens:
            Hard upper bound on the number of generated tokens.
        dead_end_strategy:
            ``"restart"`` (jump to a known-good context), ``"stop"`` or
            ``"error"``.
        stop_tokens:
            Tokens that stop generation once ``min_tokens`` were produced.
        min_tokens:
            Minimum number of tokens to produce before ``stop_tokens`` is
            honoured (so e.g. ``EOS`` cannot end the piece too early).
        """
        if self.tokenizer_ is None:
            raise RuntimeError("fit() must be called before generate()")
        if dead_end_strategy not in ("restart", "stop", "error"):
            raise ValueError("unknown dead_end_strategy %r" % (dead_end_strategy,))
        if max_tokens < 0:
            raise ValueError("max_tokens must be >= 0")

        stop_set = set(stop_tokens or [])

        if initial_context is None:
            context = self._random_observed_context()
        else:
            context = list(initial_context)[-self.order :]
            if len(context) < self.order:
                # pad to the left with the first token so the context is valid
                context = [context[0]] * (self.order - len(context)) + context

        generated = []
        dead_ends = 0
        guard = 0
        max_guard = 1000 + 10 * max_tokens
        while len(generated) < max_tokens:
            guard += 1
            if guard > max_guard:
                raise RuntimeError("generation made no progress (dead-end loop)")

            distr = self.next_distribution(context)
            if distr is None:
                dead_ends += 1
                if dead_end_strategy == "stop":
                    break
                if dead_end_strategy == "error":
                    raise RuntimeError("dead end at context %r" % (context,))
                context = self._random_observed_context()
                continue

            index = int(self._rng.choice(self.vocabulary_size, p=distr))
            token = self.tokenizer_.id_to_token[index]
            context = (context + [token])[-self.order :]
            generated.append(token)
            self.last_dead_end_count = dead_ends

            if token in stop_set and len(generated) >= min_tokens:
                break

        self.last_dead_end_count = dead_ends
        return generated
