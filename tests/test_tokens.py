"""Checks for incremental prompt tokenisation.

The sampler submits token ids it built up step by step instead of handing vLLM
the whole transcript to re-tokenise.  That is only sound if the incremental ids
are *identical* to tokenising the whole prompt: a one-token difference is a
different conditioning context, so it would quietly change the distribution this
project exists to measure.  These checks are the gate on that.

Runs under plain ``python tests/test_tokens.py`` as well as pytest.  The
instrument walks need a tokenizer; they skip themselves when none is cached
locally, which is the case for DeepSeek-V4-Flash-Base outside DAIS.
"""

from __future__ import annotations

import sys

from silicon_sampling.sampling.tokens import PrefixTokens, newline_boundary, verify


#: A tokenizer stand-in with the property that actually matters: runs of
#: newlines merge into one token, so cutting inside a run tokenises differently.
#: Every id is derived from the piece of text it covers, so a mismatch shows up.
def _fake_encode(text: str) -> list[int]:
    ids = []
    index = 0
    while index < len(text):
        if text[index] == "\n":
            rest = text[index:]
            run = len(rest) - len(rest.lstrip("\n"))
            ids.append(1000 + run)  # "\n" and "\n\n" are *different* tokens
            index += run
        else:
            end = text.find("\n", index)
            end = len(text) if end < 0 else end
            for word in text[index:end].split(" "):
                ids.append(hash(word) % 997)
            index = end
    return ids


def test_newline_boundary_never_splits_a_newline_run():
    # "a\n\nb": the only safe cut is at index 3, after both newlines.
    assert newline_boundary("a\n\nb", 0) == 3
    # A cut already past the last boundary cannot move.
    assert newline_boundary("a\n\nb", 3) == 3
    # No newline at all: nothing can be cached.
    assert newline_boundary("abc", 0) == 0
    # A trailing newline run is not a boundary — the next step may extend it.
    assert newline_boundary("a\nb\n", 0) == 2


def test_incremental_matches_whole_string_on_a_growing_prompt():
    cache = PrefixTokens(_fake_encode)
    text = ""
    for question in range(20):
        text += f"Q{question}. how are you\n      pick one\nResponse: "
        assert cache.ids(text) == _fake_encode(text)
        text += f"answer {question}\n\n"
        assert cache.ids(text) == _fake_encode(text)


def test_resubmitting_the_same_prompt_is_stable():
    """Rejection sampling re-issues the identical prompt; it must not drift."""
    cache = PrefixTokens(_fake_encode)
    text = "Q1. why\n\nResponse: "
    first = cache.ids(text)
    assert cache.ids(text) == first
    assert cache.ids(text) == _fake_encode(text)


def test_a_cut_inside_a_newline_run_would_have_been_wrong():
    """Why `newline_boundary` looks for the *end* of a run, not just any newline."""
    text = "a\n\nb"
    inside = 2  # between the two newlines: each becomes its own token
    assert _fake_encode(text[:inside]) + _fake_encode(text[inside:]) != _fake_encode(
        text
    )
    assert newline_boundary(text, 0) != inside


def test_a_newline_run_split_across_two_steps_stays_exact():
    """The hazard a plain rfind would walk into.

    After a prompt ending ``"a\\n"``, caching up to that newline looks safe — but
    the next step turns it into ``"a\\n\\n"``, and the run has been split.  Nothing
    may be cached until the run is known to be complete.
    """
    cache = PrefixTokens(_fake_encode)
    assert cache.ids("a\n") == _fake_encode("a\n")
    grown = "a\n\nb\nResponse: "
    assert cache.ids(grown) == _fake_encode(grown)


def _tokenizer(model: str):
    try:
        from silicon_sampling.sampling.tokens import load_tokenizer

        return load_tokenizer(model)
    except Exception as error:  # pragma: no cover - depends on the local cache
        print(f"  (skipped: no tokenizer for {model}: {type(error).__name__})")
        return None


def _walk_prompts(session, widest) -> list[str]:
    """Every prompt one respondent is shown, in order."""
    prompts = []
    while (step := session.next_prompt()) is not None:
        prompts.append(step[0])
        session.submit(step[1], widest(step[1]))
    return prompts


def test_pfander_transcripts_tokenise_incrementally(model="Qwen/Qwen2.5-7B"):
    tokenizer = _tokenizer(model)
    if tokenizer is None:
        return
    from silicon_sampling.pfander.run import widest_answer, worst_case_sessions

    for session in worst_case_sessions():
        verify(tokenizer, _walk_prompts(session, widest_answer))


def test_voelkel_transcripts_tokenise_incrementally(model="Qwen/Qwen2.5-7B"):
    tokenizer = _tokenizer(model)
    if tokenizer is None:
        return
    from silicon_sampling.voelkel.run import _widest, worst_case_sessions

    for session in worst_case_sessions():
        verify(tokenizer, _walk_prompts(session, _widest))


def main() -> int:
    failures = 0
    for name, test in sorted(globals().items()):
        if name.startswith("test_") and callable(test):
            try:
                test()
                print(f"ok   {name}")
            except Exception as error:
                failures += 1
                print(f"FAIL {name}: {error}")
    print(f"\n{failures} failure(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
