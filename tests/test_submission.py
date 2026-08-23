"""Checks for the submission packaging and format gate.

The point of :mod:`silicon_sampling.submission.check` is that it says the same
thing the benchmark's R script would say, so the tests that matter here are the
ones pinning *which* defects are FAILs and which are WARNs — a WARN misread as a
FAIL would make us discard a perfectly acceptable submission.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from silicon_sampling.submission import build, check, spec

TEMPLATE = spec.default_template_root()
needs_template = pytest.mark.skipif(
    TEMPLATE is None, reason="no submission-template checkout available"
)


def frame(per_condition: int = 2) -> pd.DataFrame:
    """A minimal, entirely valid Tier-1 frame."""
    rows = []
    counter = 0
    for condition in spec.CONDITIONS:
        for index in range(per_condition):
            counter += 1
            row = {
                "profile_id": f"p{counter:05d}",
                "condition": condition,
                **{
                    name: levels[index % len(levels)]
                    for name, levels in spec.MODERATORS.items()
                },
            }
            for item in spec.TRUST_ITEMS:
                row[item] = 50
            row["trust_multidimensional"] = 50.0
            for outcome in spec.SCALE_0_100:
                row.setdefault(outcome, 40.0)
            row["donation_ams"] = 5
            row["newsletter_signup"] = index % 2
            rows.append(row)
    return pd.DataFrame(rows)[list(spec.TIER1_COLUMNS)]


def build_repo(
    tmp_path: Path, data: pd.DataFrame | Path, **kwargs
) -> build.BuildResult:
    return build.build_submission(
        data, tmp_path / "sub", template_root=TEMPLATE, **kwargs
    )


def status_of(result: check.CheckResult, prefix: str) -> list[str]:
    return [row.status for row in result.rows if row.check.startswith(prefix)]


def test_spec_agrees_with_the_shipped_materials():
    assert spec.verify_against_codebook() == []


def test_tier1_columns_are_the_33_the_example_uses():
    assert len(spec.TIER1_COLUMNS) == 33
    candidates = [
        TEMPLATE / "predictions" / "example_T1_primary_v1.csv" if TEMPLATE else None,
        spec._SNAPSHOT / "example_T1_primary_v1.csv",
    ]
    example = next((path for path in candidates if path and path.is_file()), None)
    if example is None:
        pytest.skip("no example Tier-1 file available")
    header = example.read_text(encoding="utf-8").splitlines()[0]
    assert tuple(header.split(",")) == spec.TIER1_COLUMNS


def test_reference_level_is_the_first_level():
    assert spec.REFERENCE_LEVELS == {
        "gender": "Male",
        "age_band": "18-29",
        "race": "White / Caucasian",
        "education": "Less than high school",
        "income": "Less than $30,000",
        "party": "Republican",
    }
    for name, levels in spec.MODERATORS.items():
        assert spec.REFERENCE_LEVELS[name] == levels[0]


def test_filename_grammar():
    pattern = spec.filename_pattern("mpib", 1)
    assert pattern.match("mpib_T1_primary_v1.csv")
    assert pattern.match("mpib_T1_secondary-2_v12.csv")
    assert not pattern.match("mpib_T1_primary_v1.CSV")
    assert not pattern.match("mpib_T1_secondary_v1.csv")
    assert not pattern.match("mpib_T2_primary_v1.csv")
    assert not pattern.match("other_T1_primary_v1.csv")
    assert (
        spec.prediction_filename("mpib", "secondary-1", 3)
        == "mpib_T1_secondary-1_v3.csv"
    )
    with pytest.raises(ValueError):
        spec.prediction_filename("mpib", "tertiary")


def test_orcid_checksum():
    assert check.orcid_is_valid("0000-0002-1825-0097")
    assert not check.orcid_is_valid("0000-0000-0000-0000")
    assert not check.orcid_is_valid("not-an-orcid")


@needs_template
def test_a_clean_build_passes_with_only_expected_warnings(tmp_path):
    result = build_repo(tmp_path, frame())
    assert result.predictions.name == "mpib_T1_primary_v1.csv"
    verdict = check.check_repo(result.root)
    assert verdict.failures == []
    assert verdict.verdict == "PASS WITH WARNINGS"
    warned = {row.check.split(":")[0] for row in verdict.warnings}
    # The only data warning a 34-row pilot earns is the precision floor.
    assert "precision floor (500/intervention, 1,000 control)" in warned
    assert "registration.md filled in" in warned


@needs_template
def test_full_size_frame_clears_the_precision_floor(tmp_path):
    data = pd.concat(
        [frame(500), frame(500).assign(condition=spec.CONTROL)], ignore_index=True
    )
    data["profile_id"] = [f"p{index:05d}" for index in range(len(data))]
    result = build_repo(tmp_path, data)
    verdict = check.check_repo(result.root)
    assert verdict.failures == []
    assert status_of(verdict, "precision floor") == [check.PASS]


@needs_template
def test_verbatim_copy_keeps_the_bytes(tmp_path):
    source = tmp_path / "tier1_submission.csv"
    frame().to_csv(source, index=False)
    result = build_repo(tmp_path, source)
    assert result.copied_verbatim
    assert result.predictions.read_bytes() == source.read_bytes()
    assert result.sha256 == check.sha256_file(source)
    meta = json.loads(result.metadata.read_text())
    assert meta["prediction_files"] == [
        {"file": "predictions/mpib_T1_primary_v1.csv", "sha256": result.sha256}
    ]
    assert check.check_repo(result.root).failures == []


@needs_template
def test_extra_columns_are_kept_after_the_schema(tmp_path):
    data = frame()
    data["scratch"] = 1
    result = build_repo(tmp_path, data[["scratch"] + list(spec.TIER1_COLUMNS)])
    written = pd.read_csv(result.predictions)
    assert tuple(written.columns) == spec.TIER1_COLUMNS + ("scratch",)
    assert check.check_repo(result.root).failures == []


@needs_template
def test_nothing_is_overwritten_by_default(tmp_path):
    build_repo(tmp_path, frame())
    with pytest.raises(FileExistsError):
        build_repo(tmp_path, frame())
    build_repo(tmp_path, frame(), overwrite=True)


@needs_template
def test_metadata_patch_keeps_hand_edited_fields(tmp_path):
    result = build_repo(tmp_path, frame())
    document = json.loads(result.metadata.read_text())
    document["contact"] = "someone@mpib-berlin.mpg.de"
    result.metadata.write_text(json.dumps(document, indent=2))
    build_repo(tmp_path, frame(), overwrite=True)
    assert (
        json.loads(result.metadata.read_text())["contact"]
        == "someone@mpib-berlin.mpg.de"
    )


@needs_template
@pytest.mark.parametrize(
    "prefix, status, mutate",
    [
        ("condition labels valid", check.FAIL, lambda d: d.assign(condition="Control")),
        ("gender levels valid", check.FAIL, lambda d: d.assign(gender="male")),
        ("gender has data", check.FAIL, lambda d: d.assign(gender=None)),
        ("Tier-1 required columns", check.FAIL, lambda d: d.drop(columns=["party"])),
        ("profile_id unique", check.WARN, lambda d: d.assign(profile_id="p1")),
        ("trust_post in [0,100]", check.WARN, lambda d: d.assign(trust_post=101)),
        ("donation_ams in [0,10]", check.WARN, lambda d: d.assign(donation_ams=11)),
        (
            "newsletter_signup binary",
            check.WARN,
            lambda d: d.assign(newsletter_signup=2),
        ),
        (
            "trust_multidimensional consistent",
            check.WARN,
            lambda d: d.assign(trust_multidimensional=90.0),
        ),
        (
            "all 17 conditions present",
            check.WARN,
            lambda d: d[d["condition"] != "Consensus"],
        ),
    ],
)
def test_defects_keep_the_R_s_fail_warn_split(tmp_path, prefix, status, mutate):
    result = build_repo(tmp_path, mutate(frame()))
    verdict = check.check_repo(result.root)
    assert status_of(verdict, prefix) == [status], verdict.report_lines()
    if status == check.FAIL:
        assert verdict.verdict == check.FAIL
        assert not verdict.passed
    else:
        assert verdict.passed


@needs_template
def test_partly_missing_moderator_warns_but_all_missing_fails(tmp_path):
    data = frame(10)
    data.loc[data.index[:100], "party"] = None
    result = build_repo(tmp_path, data)
    verdict = check.check_repo(result.root)
    assert status_of(verdict, "party mostly present") == [check.WARN]
    assert verdict.passed


@needs_template
@pytest.mark.parametrize(
    "prefix, patch",
    [
        ("coverage is full", {"coverage": {"interventions": 15, "outcomes": 13}}),
        ("coverage declared", {"coverage": {}}),
        ("blinding_attestation == true", {"blinding_attestation": False}),
        ("entry is primary|secondary-k", {"entry": "best"}),
        ("disclosure_class in {A,B,C}", {"disclosure_class": "D"}),
        ("tier in {1,2,3}", {"tier": 4}),
        ("contact present", {"contact": ""}),
        ("models listed", {"models": []}),
    ],
)
def test_metadata_defects_fail(tmp_path, prefix, patch):
    result = build_repo(tmp_path, frame())
    document = json.loads(result.metadata.read_text())
    document.update(patch)
    result.metadata.write_text(json.dumps(document, indent=2))
    verdict = check.check_repo(result.root)
    assert status_of(verdict, prefix) == [check.FAIL], verdict.report_lines()


@needs_template
def test_fingerprint_mismatch_and_missing_file_fail(tmp_path):
    result = build_repo(tmp_path, frame())
    document = json.loads(result.metadata.read_text())
    document["prediction_files"][0]["sha256"] = "0" * 64
    result.metadata.write_text(json.dumps(document, indent=2))
    verdict = check.check_repo(result.root)
    assert status_of(verdict, "sha256 matches") == [check.FAIL]

    result.predictions.unlink()
    verdict = check.check_repo(result.root)
    assert status_of(verdict, "file present") == [check.FAIL]


@needs_template
def test_tier_2_is_not_ported_rather_than_silently_passing(tmp_path):
    result = build_repo(tmp_path, frame())
    document = json.loads(result.metadata.read_text())
    document["tier"] = 2
    result.metadata.write_text(json.dumps(document, indent=2))
    with pytest.raises(NotImplementedError):
        check.check_repo(result.root)


@needs_template
def test_missing_repo_files_fail(tmp_path):
    result = build.build_submission(frame(), tmp_path / "sub")  # no template staging
    verdict = check.check_repo(result.root)
    failed = {row.check for row in verdict.failures}
    assert {
        "registration.md present",
        "codebook.csv present",
        "survey/ present",
    } <= failed


def test_report_renders_like_the_r_script():
    result = check.CheckResult(
        [
            check.CheckRow("a", check.PASS),
            check.CheckRow("b", check.WARN, "why"),
        ]
    )
    lines = result.report_lines()
    assert lines[0] == "Silicon Sample Benchmark — submission self-check"
    assert lines[2] == "[ok]   a"
    assert lines[3] == "[warn] b  — why"
    assert lines[-1] == "OVERALL: PASS WITH WARNINGS   (1 pass, 1 warn, 0 fail)"
