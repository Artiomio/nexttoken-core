"""The text example must keep working on top of the universal core."""

from examples.text_generation import generate_text

SAMPLE_TEXT = (
    "the quick brown fox jumps over the lazy dog\n"
    "the quick brown fox jumps over the lazy cat\n"
    "the lazy dog sleeps and the quick fox runs\n"
    "a quick brown dog jumps over a lazy fox\n"
)


def test_generate_text_is_deterministic_and_reuses_symbols():
    first = generate_text(SAMPLE_TEXT, order=3, n_tokens=80, random_seed=42)
    second = generate_text(SAMPLE_TEXT, order=3, n_tokens=80, random_seed=42)

    assert first == second
    assert first  # non-empty
    assert set(first) <= set(SAMPLE_TEXT.replace("\n", ""))
