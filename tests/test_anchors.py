"""External level anchors: the crosswalk's integrity, the conversions, the levels.

The failure mode this file exists to catch is not an exception, it is a plausible
wrong number.  An anchor is a control-arm level borrowed from a survey that asked
a slightly different question on a different scale, and every step between the raw
file and the level handed to ``calibrate`` can produce something that looks
reasonable and is off by ten points — which the Voelkel validation shows is enough
to make the anchoring actively harmful.  So the tests here check the arithmetic
against hand-computed values, check that the graded filtering cannot let a weak
mapping through, and check that no outcome can end up with two anchors, which is
the one bookkeeping error that would let a later change silently swap which source
a level came from.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from silicon_sampling.anchors import crosswalk as cw
from silicon_sampling.anchors import ccam, levels, scales, tisp, validate
from silicon_sampling.calibration import tier1
from silicon_sampling.pfander.outcomes import (
    DIRECT,
    MEANS,
    MODERATORS,
    OUTCOMES,
    SUBSCALES,
)

pytest.importorskip("pyreadstat")

needs_tisp = pytest.mark.skipif(
    not tisp.CSV.exists(), reason=f"{tisp.CSV} is not in this checkout"
)
needs_ccam = pytest.mark.skipif(
    not ccam.SAV.exists(), reason=f"{ccam.SAV} is not in this checkout"
)


# --------------------------------------------------------------------------- #
# scale conversion
# --------------------------------------------------------------------------- #


def test_linear_conversion_puts_the_options_where_the_docstring_says():
    values = pd.Series([1, 2, 3, 4, 5])
    assert scales.to_slider(values, 5).tolist() == [0.0, 25.0, 50.0, 75.0, 100.0]
    assert scales.to_slider(pd.Series([1, 2, 3]), 3).tolist() == [0.0, 50.0, 100.0]


def test_bin_midpoint_conversion_contracts_toward_the_scale_midpoint():
    values = pd.Series([1, 2, 3, 4, 5])
    assert scales.to_slider_bin_midpoint(values, 5).tolist() == [
        10.0,
        30.0,
        50.0,
        70.0,
        90.0,
    ]


def test_the_two_conversions_disagree_by_the_closed_form():
    """``(50 - mean) / k``, which is the anchor's irreducible error."""
    values = pd.Series([4, 5, 4, 3, 5, 4])
    linear = scales.to_slider(values, 5).mean()
    midpoint = scales.to_slider_bin_midpoint(values, 5).mean()
    assert midpoint - linear == pytest.approx(scales.conversion_gap(linear, 5))


def test_out_of_range_and_missing_codes_drop_rather_than_convert():
    values = pd.Series([1.0, 4.0, 6.0, -1.0, np.nan])
    converted = scales.to_slider(values, 5, missing_codes=(-1.0,))
    assert converted.notna().tolist() == [True, True, False, False, False]


def test_sheppard_correction_removes_the_grouping_variance():
    """A 1-5 scale converted to 0-100 has options 25 apart, so h^2/12 = 52.1."""
    assert scales.sheppard_sd(30.0, 5) == pytest.approx(np.sqrt(900 - 625 / 12))
    assert scales.sheppard_sd(1.0, 5) == 0.0


def test_weighted_moments_match_a_hand_computation():
    values = pd.Series([0.0, 50.0, 100.0])
    weights = pd.Series([1.0, 1.0, 2.0])
    result = scales.weighted_moments(values, weights)
    assert result.mean == pytest.approx(62.5)
    assert result.n == 3
    # Kish: (sum w)^2 / sum w^2 = 16 / 6.
    assert result.n_effective == pytest.approx(16 / 6)


def test_weights_that_are_missing_or_zero_drop_the_row():
    values = pd.Series([0.0, 100.0, 100.0, 100.0])
    weights = pd.Series([1.0, 1.0, 0.0, np.nan])
    assert scales.weighted_moments(values, weights).mean == pytest.approx(50.0)


def test_composite_requires_every_item_unless_told_otherwise():
    frame = pd.DataFrame({"a": [1, 5, 5], "b": [1, 5, np.nan]})
    strict = scales.composite(frame, ("a", "b"), options=5)
    assert strict.notna().tolist() == [True, True, False]
    loose = scales.composite(frame, ("a", "b"), options=5, min_answered=1)
    assert loose.tolist() == [0.0, 100.0, 100.0]


# --------------------------------------------------------------------------- #
# the crosswalk
# --------------------------------------------------------------------------- #


def test_no_outcome_is_anchored_twice():
    """The bookkeeping invariant: one level per outcome, from one named group.

    Several groups may *compete* for an outcome — TISP offers both the trust
    battery and a rival single item for ``trust_post`` — but the assembled table
    has to choose, or a later edit could change which source a level came from
    without changing anything visible.
    """
    for grade in cw.GRADES:
        anchors = levels.build(grade)
        assert len(anchors) == len(set(anchors))
        for outcome, anchor in anchors.items():
            assert anchor.outcome == outcome
            assert len({entry for entry in anchor.items}) == len(anchor.items)


def test_no_crosswalk_row_is_duplicated_and_no_source_item_is_reused():
    keys = [
        (e.pfander_outcome, e.pfander_item, e.source, e.source_item)
        for e in cw.CROSSWALK
    ]
    assert len(keys) == len(set(keys))
    # A source item standing in for two different outcomes would be double-dipping:
    # the same 2,559 answers would be sold as two independent anchors.
    by_item: dict[tuple[str, str], set[str]] = {}
    for entry in cw.CROSSWALK:
        by_item.setdefault((entry.source, entry.source_item), set()).add(
            entry.pfander_outcome
        )
    assert all(len(outcomes) == 1 for outcomes in by_item.values())


def test_graded_filtering_is_monotone_and_respects_the_order():
    sizes = [len(cw.at_least(grade)) for grade in cw.GRADES]
    assert sizes == sorted(sizes)
    assert sizes[-1] == len(cw.CROSSWALK)
    for grade in cw.GRADES:
        limit = cw.grade_rank(grade)
        assert all(cw.grade_rank(e.grade) <= limit for e in cw.at_least(grade))
    with pytest.raises(ValueError):
        cw.at_least("pretty good")


def test_graded_filtering_of_assembled_anchors_only_ever_adds():
    seen: set[str] = set()
    for grade in cw.GRADES:
        anchors = set(levels.build(grade))
        assert seen <= anchors
        seen = anchors
    assert not levels.build("verbatim"), "nothing in either source is verbatim"
    assert set(levels.build("near")) == {
        "trust_multidimensional",
        "trust_post",
        "policy_role_mean",
    }


def test_every_scored_outcome_is_either_crosswalked_or_explained():
    """A search that found nothing has to be recorded as such, not omitted."""
    crosswalked = {entry.pfander_outcome for entry in cw.CROSSWALK}
    assert crosswalked | set(cw.UNMATCHED) == set(OUTCOMES)
    assert not crosswalked & set(cw.UNMATCHED)


def test_crosswalk_items_are_real_pfander_items():
    known = (
        set(DIRECT)
        | {item for items in MEANS.values() for item in items}
        | {item for items in SUBSCALES.values() for item in items}
        | {f"trust_{facet}_{i}" for facet in SUBSCALES for i in (1, 2, 3)}
    )
    known |= {name.replace("trust_", "trust_") for name in known}
    for entry in cw.CROSSWALK:
        stem = entry.pfander_item.split(" ")[0]
        assert stem in known or stem.startswith(
            ("trust_", "policy_", "funding_", "concern_", "individual_", "belief_")
        ), entry.pfander_item


def test_a_group_is_only_as_good_as_its_worst_item():
    mixed = (
        cw.CROSSWALK[0],
        cw.Entry(
            pfander_outcome="trust_post",
            pfander_item="trust_post_1",
            pfander_text="x",
            pfander_scale="x",
            source="TISP",
            source_item="x",
            source_text="x",
            source_scale="x",
            source_options=5,
            grade="unusable",
            note="x",
            group="test",
        ),
    )
    assert cw.group_grade(mixed) == "unusable"


def test_an_unknown_grade_or_outcome_is_rejected_at_construction():
    with pytest.raises(ValueError):
        cw.Entry(
            pfander_outcome="trust_post",
            pfander_item="a",
            pfander_text="a",
            pfander_scale="a",
            source="TISP",
            source_item="a",
            source_text="a",
            source_scale="a",
            source_options=5,
            grade="excellent",
            note="a",
            group="test",
        )
    with pytest.raises(ValueError):
        cw.Entry(
            pfander_outcome="trust_in_vibes",
            pfander_item="a",
            pfander_text="a",
            pfander_scale="a",
            source="TISP",
            source_item="a",
            source_text="a",
            source_scale="a",
            source_options=5,
            grade="near",
            note="a",
            group="test",
        )


# --------------------------------------------------------------------------- #
# the sources
# --------------------------------------------------------------------------- #


@needs_tisp
def test_tisp_us_subsample_is_the_one_the_report_describes():
    frame = tisp.load()
    assert len(frame) == 2559
    assert set(frame["UserLanguage"]) == {"EN"}
    assert frame["weight"].notna().all()
    # The country weight averages to 1 within the country, which is what makes it
    # the right one here; reading it as a string would silently unweight everything.
    assert frame["weight"].mean() == pytest.approx(1.0, abs=0.01)
    assert frame["weight"].dtype.kind == "f"


@needs_tisp
def test_tisp_trust_items_all_use_the_full_five_point_scale():
    frame = tisp.load()
    for items in tisp.FACETS.values():
        for item in items:
            values = set(pd.to_numeric(frame[item], errors="coerce").dropna().unique())
            assert values == {1.0, 2.0, 3.0, 4.0, 5.0}, item


@needs_tisp
def test_the_referent_gap_is_positive_and_the_size_the_report_claims():
    gap = tisp.referent_gap()
    assert gap.mean == pytest.approx(3.92, abs=0.05)
    assert gap.se < 1.0
    assert gap.n > 2500


@needs_tisp
def test_the_trust_facets_are_ordered_competence_down_to_openness():
    facets = tisp.facet_levels()
    assert (
        facets["competence"] > facets["integrity"] > facets["benevolence"]
    ) and facets["benevolence"] > facets["openness"]
    # A 12-point spread is far too wide to treat as noise, which is why
    # ``levels.facet_levels`` exists at all.
    assert max(facets.values()) - min(facets.values()) > 10


@needs_ccam
def test_ccam_defaults_to_the_most_recent_complete_wave():
    frame = ccam.load()
    assert len(frame) == 1013
    assert set(frame["wave"]) == {ccam.DEFAULT_WAVE}
    assert ccam.wave_label() == "Dec 2024"
    for entry in cw.CROSSWALK:
        if entry.source == ccam.SOURCE and entry.source_options is not None:
            assert frame[entry.source_item].notna().sum() > 950, entry.source_item


@needs_ccam
def test_ccam_refusals_are_dropped_rather_than_scored():
    """Scoring the ``-1`` refusals would pull every CCAM level down by their share."""
    frame = ccam.load()
    entry = next(e for e in cw.CROSSWALK if e.source_item == "reduce_tax")
    raw = pd.to_numeric(frame[entry.source_item], errors="coerce")
    assert (raw <= 0).any(), "the wave should contain refusals to drop"
    assert -1.0 in entry.missing_codes
    assert ccam.measure((entry,), frame).n == int((raw > 0).sum())


@needs_ccam
@needs_tisp
def test_the_cross_source_check_finds_a_gap_above_break_even():
    """The two sources disagree by more than the tolerance, which is the verdict."""
    check = ccam.cross_source_check()
    assert len(check) == 2
    assert check["abs_gap"].min() > 5.0


# --------------------------------------------------------------------------- #
# the assembled levels
# --------------------------------------------------------------------------- #


@needs_tisp
def test_the_offered_anchors_are_in_range_and_carry_their_provenance():
    for outcome, anchor in levels.build().items():
        assert 0.0 <= anchor.mean <= 100.0
        assert 0.0 < anchor.sd < 50.0
        assert anchor.sd_slider < anchor.sd
        assert anchor.se < 1.0
        assert anchor.n > 2000
        assert anchor.grade in ("verbatim", "near")
        assert anchor.provenance and anchor.notes
        assert anchor.source in levels.SOURCES
        assert outcome in OUTCOMES


@needs_tisp
def test_the_referent_shift_is_applied_to_the_trust_battery_and_nothing_else():
    adjusted = levels.build(referent_adjust=True)
    plain = levels.build(referent_adjust=False)
    assert (
        adjusted["trust_multidimensional"].mean < plain["trust_multidimensional"].mean
    )
    assert adjusted["trust_multidimensional"].referent_adjustment == pytest.approx(
        tisp.referent_gap().mean
    )
    for outcome in ("trust_post", "policy_role_mean"):
        assert adjusted[outcome].mean == plain[outcome].mean
        assert adjusted[outcome].referent_adjustment == 0.0


@needs_tisp
def test_levels_are_a_plain_mapping_with_no_missing_values():
    table = levels.levels()
    assert table and all(np.isfinite(value) for value in table.values())
    # ``belief_post``'s only candidate has no ordered scale to convert, so it must
    # not appear even when the caller asks for unusable grades.
    assert "belief_post" not in levels.levels("unusable")


@needs_tisp
def test_the_report_table_explains_every_outcome_it_cannot_anchor():
    frame = levels.to_frame("near")
    assert len(frame) == len(OUTCOMES)
    missing = frame[frame["mean"].isna()]
    assert len(missing) == len(OUTCOMES) - 3
    assert missing["grade"].str.len().gt(0).all()


def pfander_like(n: int = 2400, seed: int = 7) -> pd.DataFrame:
    """A minimal Tier-1 frame carrying only what the anchored outcomes need."""
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame(
        {"condition": rng.choice(["control", "message_a", "message_b"], size=n)}
    )
    for moderator, values in MODERATORS.items():
        frame[moderator] = rng.choice(values[:2], size=n)
    for item in tier1.TRUST_ITEMS:
        frame[item] = np.clip(rng.normal(55, 18, size=n), 0, 100)
    frame["trust_multidimensional"] = frame[list(tier1.TRUST_ITEMS)].mean(axis=1)
    for outcome in ("trust_post", "policy_role_mean"):
        frame[outcome] = np.clip(rng.normal(58, 20, size=n), 0, 100)
    return frame


@needs_tisp
def test_the_anchors_drop_into_tier1_calibrate_without_moving_an_effect():
    """The whole point: a level anchor must not touch the leaderboard's metrics."""
    frame = pfander_like()
    before = frame.groupby("condition")["trust_multidimensional"].mean()
    rebuilt, audit = tier1.calibrate(frame, levels=levels.levels())
    after = rebuilt.groupby("condition")["trust_multidimensional"].mean()
    assert audit["max_abs_effect_drift"].max() < 1e-9
    assert (after - after["control"]).sub(before - before["control"]).abs().max() < 1e-9
    for outcome, level in levels.levels().items():
        realised = rebuilt.loc[rebuilt["condition"] == "control", outcome].mean()
        assert realised == pytest.approx(level, abs=1e-6)


# --------------------------------------------------------------------------- #
# the Voelkel validation
# --------------------------------------------------------------------------- #


def voelkel_like(n: int = 1200, seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame(
        {
            "condition": rng.choice([validate.CONTROL, "Party_Overlap"], size=n),
            "gender": rng.choice(["Male", "Female"], size=n),
            "race": rng.choice(["White", "Black"], size=n),
            "party_gen": rng.choice(["Republican", "Democrat"], size=n),
            "age_band": rng.choice(["18-29", "60+"], size=n),
            "education": rng.choice(["HS", "BA"], size=n),
        }
    )
    for outcome in ("PA", "ADA", "SPV", "SUC", "OppBip", "SocDistrust", "SocDis"):
        frame[outcome] = np.clip(rng.normal(60, 15, size=n), 0, 100)
    frame["BEPF"] = np.clip(rng.normal(40, 12, size=n), 0, 100)
    frame["Composite"] = frame[["PA", "ADA", "SPV"]].mean(axis=1)
    return frame


def test_relevelling_moves_the_level_and_holds_every_effect_fixed():
    frame = voelkel_like()
    wanted = {"PA": 20.0, "ADA": 80.0}
    before = frame.groupby("condition")["PA"].mean()
    rebuilt, drift = validate.relevel(frame, wanted)
    after = rebuilt.groupby("condition")["PA"].mean()
    assert drift["max_abs_effect_drift"].max() < 1e-9
    assert rebuilt.loc[rebuilt["condition"] == validate.CONTROL, "PA"].mean() == (
        pytest.approx(20.0)
    )
    assert (after - after[validate.CONTROL]).sub(
        before - before[validate.CONTROL]
    ).abs().max() < 1e-9
    # An outcome absent from the anchor keeps its own level untouched.
    assert rebuilt.loc[rebuilt["condition"] == validate.CONTROL, "SPV"].mean() == (
        pytest.approx(frame.loc[frame["condition"] == validate.CONTROL, "SPV"].mean())
    )


def test_break_even_interpolates_and_reports_no_crossing_as_infinite():
    sweep = pd.DataFrame(
        [
            {
                "run": "r",
                "anchor": "none (raw sample)",
                "nominal_error": np.nan,
                "realised_error": 9.0,
                "w1": 10.0,
                "ovl": 0.7,
                "ks": 0.2,
                "baseline_r": 0.8,
                "baseline_rmse": 12.0,
            },
            {
                "run": "r",
                "anchor": "truth +0",
                "nominal_error": 0.0,
                "realised_error": 0.0,
                "w1": 4.0,
                "ovl": 0.8,
                "ks": 0.1,
                "baseline_r": 0.9,
                "baseline_rmse": 4.0,
            },
            {
                "run": "r",
                "anchor": "truth +10",
                "nominal_error": 10.0,
                "realised_error": 10.0,
                "w1": 14.0,
                "ovl": 0.6,
                "ks": 0.3,
                "baseline_r": 0.95,
                "baseline_rmse": 8.0,
            },
        ]
    )
    result = validate.break_even(sweep).set_index("metric")["break_even_error"]
    assert result["w1"] == pytest.approx(6.0)
    assert result["ovl"] == pytest.approx(5.0)
    assert result["ks"] == pytest.approx(5.0)
    assert np.isinf(result["baseline_r"])
    assert np.isinf(result["baseline_rmse"])


def test_a_metric_that_the_truth_already_worsens_reports_no_break_even():
    sweep = pd.DataFrame(
        [
            {
                "run": "r",
                "anchor": "none (raw sample)",
                "nominal_error": np.nan,
                "realised_error": 9.0,
                "w1": 1.0,
                "ovl": 0.9,
                "ks": 0.1,
                "baseline_r": 0.99,
                "baseline_rmse": 1.0,
            },
            {
                "run": "r",
                "anchor": "truth +0",
                "nominal_error": 0.0,
                "realised_error": 0.0,
                "w1": 4.0,
                "ovl": 0.8,
                "ks": 0.2,
                "baseline_r": 0.9,
                "baseline_rmse": 4.0,
            },
        ]
    )
    result = validate.break_even(sweep).set_index("metric")["break_even_error"]
    assert result.isna().all()


def test_the_sign_patterns_are_what_they_claim_to_be():
    outcomes = ("PA", "ADA", "SPV", "SUC")
    assert set(validate._signs("+", outcomes).values()) == {1}
    assert set(validate._signs("-", outcomes).values()) == {-1}
    mixed = validate._signs("mixed:1", outcomes)
    assert set(mixed.values()) <= {-1, 1}
    assert validate._signs("mixed:1", outcomes) == mixed, "must be seeded"
