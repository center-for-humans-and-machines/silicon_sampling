"""Tokenise a growing prompt without re-tokenising what has not changed.

A respondent's prompt grows by one question and one answer per step and is
handed to the engine ~78 times before the session is done.  Passing it as text
costs about 1 µs per prompt token *per step* — measured at 0.99 µs/token on the
4090 box — which over the Pfänder run is 2.5 hours of CPU spent re-deriving token
ids that were identical the step before.  On a run whose GPU term is 15 minutes
that is the whole schedule.

The cache holds ids for a prefix and re-encodes only the tail.  **Where the cut
falls is the entire correctness question**: `encode(a) + encode(b)` equals
`encode(a + b)` only when the join is a boundary the pre-tokeniser would have
split at anyway.  A byte-level BPE tokeniser groups a run of newlines into its
own pre-token, so cutting *inside* such a run — between the two newlines of a
``"\\n\\n"`` paragraph break — can merge differently and silently change what the
model conditions on.  So the cut is placed at the *end of a complete newline
run*, and :func:`verify` asserts equality against whole-string tokenisation for
every step of every condition rather than trusting the argument.
"""

from __future__ import annotations

import functools
from typing import Callable, Sequence


@functools.lru_cache(maxsize=4)
def load_tokenizer(model: str):
    """The model's tokenizer, shared between the token budgets and the sampler."""
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model)


def newline_boundary(text: str, floor: int) -> int:
    """The largest cut above ``floor`` that ends a complete run of newlines.

    Returns ``floor`` when there is none, which just means nothing new can be
    cached this step.
    """
    for index in range(len(text) - 1, floor, -1):
        if text[index - 1] == "\n" and text[index] != "\n":
            return index
    return floor


class PrefixTokens:
    """Token ids for one session's monotonically growing prompt."""

    def __init__(self, encode: Callable[[str], list[int]]) -> None:
        self._encode = encode
        #: Characters of the prompt covered by :attr:`_ids`.  The invariant is
        #: that ``prompt[:_chars]`` ends a complete newline run, or is empty.
        self._chars = 0
        self._ids: list[int] = []

    def ids(self, text: str) -> list[int]:
        """Ids for ``text``, reusing the cached prefix where it still applies."""
        cut = newline_boundary(text, self._chars)
        if cut > self._chars:
            self._ids = self._ids + self._encode(text[self._chars : cut])
            self._chars = cut
        tail = text[self._chars :]
        return self._ids + self._encode(tail) if tail else list(self._ids)


def encoder(tokenizer) -> Callable[[str], list[int]]:
    """A plain ``str -> ids`` callable, with no special tokens added.

    Special tokens are the other way this can go wrong: a BOS prepended to every
    tail would land in the middle of the transcript.
    """

    def encode(text: str) -> list[int]:
        return tokenizer(text, add_special_tokens=False)["input_ids"]

    return encode


def verify(tokenizer, prompts: Sequence[str]) -> None:
    """Assert incremental tokenisation of a growing prompt is byte-exact.

    ``prompts`` is the sequence one session goes through, shortest first.  Raises
    ``AssertionError`` naming the first step that disagrees, with the surrounding
    text, because a silent disagreement here changes the sampled distribution.
    """
    encode = encoder(tokenizer)
    cache = PrefixTokens(encode)
    for step, prompt in enumerate(prompts):
        got = cache.ids(prompt)
        want = encode(prompt)
        if got != want:
            index = next(
                (i for i, (a, b) in enumerate(zip(got, want)) if a != b),
                min(len(got), len(want)),
            )
            raise AssertionError(
                f"incremental tokenisation diverged at step {step}, token {index}: "
                f"got {got[max(0, index - 4) : index + 4]} "
                f"want {want[max(0, index - 4) : index + 4]} "
                f"near {prompt[max(0, index - 40) : index + 40]!r}"
            )
