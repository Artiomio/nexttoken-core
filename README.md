# nexttoken-core

A tiny universal **n-th order Markov chain** over arbitrary hashable tokens,
extracted from [`nexttoken`](https://github.com/Artiomio/nexttoken) so that it
can be reused as a library.

The original `nexttoken` repository is kept unchanged as a pedagogical
reference (a single, inlined text example). This project is the same Markov
algorithm, split into a small importable package with a text example on top:

* the Markov core knows nothing about text, words, letters or MIDI notes — a
  token can be any hashable Python object;
* the same core powers both the text example here and the music experiment in
  [`nextnote`](https://github.com/Artiomio/nextnote).

| | |
| --- | --- |
| distribution name | `nexttoken-core` |
| import name | `nexttoken_core` |

## Layout

```text
src/nexttoken_core/
  __init__.py        # exports MarkovChain, SparseMatrix, Tokenizer, ...
  markov.py          # universal Markov core (counting + sampling)
  sparsematrix.py    # SparseMatrix (unchanged counting backend)
  tokenizer.py       # Tokenizer (hashable tokens) + EnumeratingTokenizer
  textbrushing.py    # text helpers (html cleaning, etc.)
examples/
  text_generation.py # text example using the core
tests/
```

## Install

```bash
pip install -e .
pip install pytest   # for the tests
```

Or directly from GitHub:

```bash
pip install "nexttoken-core @ git+https://github.com/Artiomio/nexttoken-core.git"
```

## The core

```python
from nexttoken_core import MarkovChain

model = MarkovChain(order=3, random_seed=42)
model.fit(sequences)                 # sequences: list[list[Hashable]]

distribution = model.next_distribution(context)   # None => dead end
generated = model.generate(
    initial_context=["BOS"] * 3,
    max_tokens=200,
    dead_end_strategy="restart",     # or "stop" / "error"
    restart_context=["BOS"] * 3,     # optional fixed restart point
    stop_tokens=["EOS"],
    min_tokens=150,                  # EOS is masked before this length
)
```

Each element of `sequences` is an independent work: the transition between the
last token of one sequence and the first token of the next one is never counted.

Notes:

* `next_distribution` returns a vector aligned with `model.vocabulary`, or
  `None` for a *dead end*. `dead_end_strategy="restart"` jumps to a context that
  has a continuation in the training data — this is a restart, not a backoff.
  By default that context is sampled from observed contexts; `restart_context`
  can pin it to a known boundary such as `BOS`.
* `stop_tokens` are masked out until `min_tokens` have been produced, then the
  remaining probability mass is renormalized. This prevents an early `EOS`
  from entering the generated stream and immediately causing a dead end.
* `distr_offset` is a peak-flattening threshold applied to already-observed
  transitions. It is **not** add-k smoothing.
* The old off-by-one (`len_uniques = max(...)`) is fixed; the vocabulary size is
  `max(...) + 1` / `len(vocabulary)`, so the last token can be generated.

## Text example

```bash
python examples/text_generation.py --file my_text.txt --order 3 --seed 42
python examples/text_generation.py --url <url> --order 3
```

## Tests

```bash
pytest
```
