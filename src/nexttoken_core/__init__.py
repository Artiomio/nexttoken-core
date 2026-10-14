"""nexttoken_core: a universal n-th order Markov chain plus a text example.

The reusable Markov core lives in :mod:`nexttoken_core.markov` and knows nothing
about text. ``nextnote`` imports :class:`nexttoken_core.MarkovChain` to generate
music, exactly like ``examples/text_generation.py`` uses it for text.
"""

from .markov import MarkovChain, calculate_transition_matrix
from .sparsematrix import SparseMatrix
from .tokenizer import EnumeratingTokenizer, Tokenizer

__all__ = [
    "MarkovChain",
    "calculate_transition_matrix",
    "SparseMatrix",
    "Tokenizer",
    "EnumeratingTokenizer",
]

__version__ = "0.2.0"
