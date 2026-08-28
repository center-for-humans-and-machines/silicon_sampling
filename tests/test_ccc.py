"""The Climate Change Challenge study package."""

from __future__ import annotations

import json

from silicon_sampling.ccc import export

#: U+2028 LINE SEPARATOR.  Named rather than inlined, because it is invisible in
#: an editor and a reader has to be told it is there.
LINE_SEPARATOR = "\u2028"


def test_records_split_on_newlines_and_nothing_else(tmp_path):
    """Model free text contains Unicode line separators, and they are not newlines.

    ``str.splitlines()`` breaks on U+2028, U+2029, VT, FF and NEL as well as
    ``\n``.  One real run held six U+2028s inside free-text answers, which cut six
    records in half and made ``json.loads`` fail with "Unterminated string" on a
    file whose 12,757 lines were every one of them valid JSON.
    """
    path = tmp_path / "answers.jsonl"
    note = f"first{LINE_SEPARATOR}second"
    payload = {"profile_id": "c1", "answers": {"note": note}}
    # ensure_ascii=False is what the sampler uses, and it is what puts a literal
    # U+2028 in the file rather than the six-character escape.
    path.write_text(json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8")

    records = list(export.read_records(path))
    assert len(records) == 1
    assert records[0]["answers"]["note"] == note

    # The bug this guards against, spelled out: one line by any sane reading, two
    # according to splitlines().
    raw = path.read_text(encoding="utf-8")
    assert raw.count("\n") == 1
    assert len(raw.splitlines()) == 2
