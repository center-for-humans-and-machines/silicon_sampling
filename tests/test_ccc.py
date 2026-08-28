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


def test_build_csv_survives_a_unicode_line_separator(tmp_path, monkeypatch):
    """The regression has to be tested through ``build_csv``, not the helper.

    An earlier version of this file tested ``read_records`` directly.  It passed
    while ``build_csv`` still called ``splitlines()`` — so the fix looked done and
    the cluster kept failing.  Test the entry point, not the piece underneath it.
    """
    from silicon_sampling.ccc import export as ex

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    note = f"first{LINE_SEPARATOR}second"
    rows = [
        {
            "profile_id": f"c{i}",
            "condition": "Control Neckties",
            "answers": {"Concern_Post_1": 50, "note": note},
        }
        for i in range(3)
    ]
    (run_dir / "answers.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
        encoding="utf-8",
    )
    monkeypatch.setattr(ex, "samples_dir", lambda run: run_dir)

    report = ex.build_csv("whatever")
    assert report["rows"] == 3
