"""Text-generation example built on the extracted universal Markov core.

This mirrors the original ``markov_word_and_letterwise.py`` (same idea, same
defaults) but uses :class:`nexttoken_core.MarkovChain` instead of inlining the
counting and sampling loops. It is the reference "text" consumer of the core.

Usage::

    python examples/text_generation.py --file my_text.txt --order 3 --seed 42
    python examples/text_generation.py --url <url> --order 3
"""

from __future__ import annotations

import argparse

import requests

from nexttoken_core import EnumeratingTokenizer, MarkovChain
from nexttoken_core.textbrushing import clean_html

DEFAULT_URL = (
    "http://az.lib.ru/t/twen_m/"
    "text_1884_the_adventures_of_huckleberry_finn.shtml"
)


def digitize_letterwise(text):
    """Return ``(digitized_ids, unique_symbols)`` for letter-wise modelling."""
    tokenizer = EnumeratingTokenizer(text)
    digitized = tokenizer.tokenize_text_by_enumerating()
    return digitized, tokenizer.unique_symbols


def generate_text(text, order=3, n_tokens=1500, random_seed=42, distr_offset=0):
    """Generate text letter-wise and return it as a string."""
    text = text.replace("\n", "")
    digitized, unique_symbols = digitize_letterwise(text)

    model = MarkovChain(order=order, random_seed=random_seed, distr_offset=distr_offset)
    model.fit([digitized])

    generated_ids = model.generate(
        initial_context=digitized[:order],
        max_tokens=n_tokens,
        dead_end_strategy="restart",
    )
    return "".join(unique_symbols[i] for i in generated_ids)


def build_arg_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--file", help="Path to a local text file")
    source.add_argument("--url", help="URL to fetch the text from")
    parser.add_argument("--order", type=int, default=3, help="Markov order")
    parser.add_argument("--tokens", type=int, default=1500, help="Tokens to generate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--distr-offset", type=int, default=0,
        help="Peak-flattening threshold for observed transitions",
    )
    return parser


def main(argv=None):
    args = build_arg_parser().parse_args(argv)

    if args.file:
        with open(args.file, "r", encoding="utf-8", errors="replace") as handle:
            text = handle.read(500 * 10 ** 6)
    else:
        url = args.url or DEFAULT_URL
        text = requests.get(url).text
        text = clean_html(text)

    print(f"Next word predicted based on {args.order} previous tokens")
    generated = generate_text(
        text,
        order=args.order,
        n_tokens=args.tokens,
        random_seed=args.seed,
        distr_offset=args.distr_offset,
    )
    print(generated)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
