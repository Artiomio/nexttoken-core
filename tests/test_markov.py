"""Tests for the universal Markov core (:mod:`nexttoken_core.markov`)."""

import numpy as np
import pytest

from nexttoken_core import MarkovChain, calculate_transition_matrix


def test_last_vocabulary_token_is_not_lost():
    """Off-by-one regression: the last token of the vocabulary must be usable."""
    model = MarkovChain(order=1, random_seed=42)
    model.fit([["a", "c"]])

    # 'c' was added last -> highest id.
    assert model.vocabulary == ["a", "c"]
    assert model.vocabulary[-1] == "c"

    distr = model.next_distribution(["a"])
    last_index = model.tokenizer_.token_to_id[model.vocabulary[-1]]

    assert distr is not None
    assert distr[last_index] == pytest.approx(1.0)

    generated = model.generate(initial_context=["a"], max_tokens=5)
    assert generated == ["c"] * 5


def test_calculate_transition_matrix_includes_last_token():
    matrix = calculate_transition_matrix([0, 1, 2], markov_dim=1)
    # vocabulary is {0, 1, 2} -> size 3, not max(=2)
    assert list(matrix.shape) == [3, 3]
    assert matrix[(1, 2)] == 1


def test_no_transition_between_independent_sequences():
    model = MarkovChain(order=1, random_seed=0)
    model.fit([["a", "b"], ["c", "d"]])

    # 'b' only ends the first sequence: there is no b -> c transition.
    assert model.next_distribution(["b"]) is None
    assert model.next_distribution(["d"]) is None
    assert model.next_distribution(["a"]) is not None
    assert model.next_distribution(["c"]) is not None


def test_dead_end_context_does_not_loop_forever():
    model = MarkovChain(order=2, random_seed=7)
    model.fit([["a", "b", "c"], ["a", "b", "d"]])

    # ("c", "a") is not an observed context -> dead end, but restart must finish.
    generated = model.generate(
        initial_context=["c", "a"], max_tokens=50, dead_end_strategy="restart"
    )
    assert len(generated) == 50


def test_dead_end_stop_strategy_terminates():
    model = MarkovChain(order=2, random_seed=7)
    model.fit([["a", "b", "c"]])
    generated = model.generate(
        initial_context=["c", "c"], max_tokens=50, dead_end_strategy="stop"
    )
    assert generated == []


def test_same_seed_gives_same_sequence():
    sequences = [["a", "b", "a", "c", "a", "b", "c"]]
    first = MarkovChain(order=1, random_seed=42).fit(sequences)
    second = MarkovChain(order=1, random_seed=42).fit(sequences)

    assert first.generate(initial_context=["a"], max_tokens=100) == second.generate(
        initial_context=["a"], max_tokens=100
    )


def test_stop_token_respects_min_tokens():
    model = MarkovChain(order=1, random_seed=42)
    model.fit([["a", "EOS"], ["a", "b", "a"]])

    generated = model.generate(
        initial_context=["a"],
        restart_context=["a"],
        max_tokens=2,
        stop_tokens=["EOS"],
        min_tokens=2,
    )
    # EOS is masked, not emitted-and-ignored, before the minimum length.
    assert generated == ["b", "a"]
    assert "EOS" not in generated


def test_fixed_restart_context_is_used_after_dead_end():
    model = MarkovChain(order=1, random_seed=42)
    model.fit([["BOS", "a", "EOS"]])

    generated = model.generate(
        initial_context=["EOS"],
        restart_context=["BOS"],
        max_tokens=1,
        dead_end_strategy="restart",
    )

    assert generated == ["a"]
    assert model.last_dead_end_count == 1


def test_invalid_restart_context_is_rejected():
    model = MarkovChain(order=1, random_seed=42)
    model.fit([["BOS", "a"]])

    with pytest.raises(ValueError, match="restart_context"):
        model.generate(
            initial_context=["BOS"],
            restart_context=["missing"],
            max_tokens=1,
        )


def test_hashable_tuple_tokens():
    """Tokens may be arbitrary hashable objects such as MIDI-like tuples."""
    model = MarkovChain(order=1, random_seed=42)
    model.fit([[(60, 4), (62, 2), (64, 2)], [(60, 4), (64, 4)]])

    distr = model.next_distribution([(60, 4)])
    assert distr is not None
    assert set(model.tokenizer_.token_to_id) == {(60, 4), (62, 2), (64, 2), (64, 4)}

    generated = model.generate(initial_context=[(60, 4)], max_tokens=5)
    assert all(isinstance(token, tuple) for token in generated)
