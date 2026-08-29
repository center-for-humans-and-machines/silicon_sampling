"""The Zenodo deposit record, and the one field that can break a deposit.

``.zenodo.json`` controls a **permanent** record.  The template's own warning is
that a malformed ORCID makes Zenodo reject the deposit with an opaque HTTP 500,
and that depositing without the file at all produces a record with no description
or license attached to a DOI that cannot be withdrawn.  Both failure modes are
silent at build time, so they are tested here.
"""

from __future__ import annotations

import json

import pytest

from silicon_sampling.submission import zenodo as Z


def _metadata(tmp_path, **overrides):
    payload = {
        "team_id": "mpib",
        "team_name": "Example",
        "tier": 1,
        "license": "CC-BY-4.0",
        "approach_family": "per-respondent simulation, multi-model component hybrid",
        "models": ["Qwen/Qwen2.5-7B"],
        "disclosure_class": "A",
        "code_repository": "https://github.com/example/repo",
        "coverage": {"interventions": 16, "outcomes": 13},
        "creators": [{"name": "Doe, Jane", "affiliation": "Somewhere", "orcid": ""}],
        "abstract": "",
    }
    payload.update(overrides)
    (tmp_path / "metadata.json").write_text(json.dumps(payload), encoding="utf-8")
    return tmp_path


@pytest.mark.parametrize(
    "orcid, kept",
    [
        # Real, checksum-valid ORCIDs, including the X check digit.
        ("0000-0002-1825-0097", True),
        ("0000-0002-1694-233X", True),
        # Right shape, wrong check digit — this is the case that 500s.
        ("0000-0002-1825-0098", False),
        # The all-zero dummy the template explicitly warns about.
        ("0000-0000-0000-0000", False),
        ("not-an-orcid", False),
        ("", False),
    ],
)
def test_only_checksum_valid_orcids_reach_the_deposit(tmp_path, orcid, kept):
    root = _metadata(
        tmp_path, creators=[{"name": "Doe, Jane", "affiliation": "X", "orcid": orcid}]
    )
    payload = Z.build_zenodo(root)
    assert ("orcid" in payload["creators"][0]) is kept
    # A rejected ORCID must never be silently dropped: it is reported.
    assert bool(payload["_omitted_orcids"]) is (bool(orcid) and not kept)


def test_the_record_carries_what_zenodo_would_otherwise_invent(tmp_path):
    root = _metadata(tmp_path)
    payload = Z.write_zenodo(root)
    written = json.loads((root / ".zenodo.json").read_text(encoding="utf-8"))
    assert written == payload
    # The four fields the template says a missing .zenodo.json leaves empty.
    assert written["description"]
    assert written["license"] == "CC-BY-4.0"
    assert written["creators"][0]["affiliation"] == "Somewhere"
    assert written["keywords"]
    assert "Tier 1 (individual simulation)" in written["title"]
    assert written["version"] == "1.0-t1"
    # The internal reporting key must not leak into the deposited file.
    assert "_omitted_orcids" not in written


def test_an_abstract_becomes_paragraphs_and_replaces_the_scaffold(tmp_path):
    root = _metadata(tmp_path, abstract="First para.\n\nSecond para.")
    payload = Z.build_zenodo(root)
    assert payload["description"] == "<p>First para.</p><p>Second para.</p>"
    assert "Approach family" not in payload["description"]


def test_the_code_repository_is_linked_only_when_it_is_real(tmp_path):
    def relations(root):
        return {r["relation"] for r in Z.build_zenodo(root)["related_identifiers"]}

    assert "isCompiledBy" in relations(_metadata(tmp_path))
    placeholder = _metadata(
        tmp_path, code_repository="https://github.com/your-team/your-repo"
    )
    assert "isCompiledBy" not in relations(placeholder)


def test_a_missing_team_id_stops_rather_than_deposits_a_nameless_record(tmp_path):
    root = _metadata(tmp_path, team_id="")
    with pytest.raises(ValueError, match="team_id"):
        Z.build_zenodo(root)


def test_every_built_entry_carries_a_well_formed_record():
    """The three entries on disk, if they have been built."""
    from pathlib import Path

    root = Path("data/pfander/submission")
    entries = [d for d in (root.iterdir() if root.is_dir() else []) if d.is_dir()]
    if not entries:  # pragma: no cover - data/ is gitignored
        pytest.skip("no built entries on disk")
    for entry in entries:
        target = entry / ".zenodo.json"
        if not target.exists():
            continue
        written = json.loads(target.read_text(encoding="utf-8"))
        assert written["upload_type"] == "software"
        assert written["access_right"] == "open"
        assert written["description"].startswith("<p>")
        assert "_omitted_orcids" not in written
