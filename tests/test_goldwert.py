"""Checks for the Goldwert calibration study.

Written to run under plain ``python tests/test_goldwert.py`` as well as pytest,
matching ``test_pfander.py``.

Three of these are the ones that matter.  ``test_derived_columns_reproduce``
recomputes every column the authors derived from the raw items in the published
file and demands an exact match, which is what makes an effect estimate here
comparable to theirs.  ``test_slot_ids_are_published_columns`` demands that every
scored response position in every template names a real column of
``goldwert_etal2026.csv``, which is what makes a sampled answer comparable to a
real one without a translation table.  ``test_modality_audit_call`` pins the
eleven-of-eighteen count, because that number is the whole value of the package
and a silent change to it would go unnoticed.
"""

from __future__ import annotations

import json
import random
import sys

import pandas as pd

from silicon_sampling.goldwert import convert, instrument, outcomes, paths, profiles
from silicon_sampling.goldwert import score, templates
from silicon_sampling.survey.render import MARKER_RE, render_template, slot_manifest
from silicon_sampling.survey.slots import ChoiceSlot, IntSlot, Slot


def _published_columns() -> set[str]:
    return set(pd.read_csv(paths.RESPONSES_CSV, nrows=2, low_memory=False).columns)


def _responses() -> pd.DataFrame:
    return pd.read_csv(paths.RESPONSES_CSV, low_memory=False)


# --------------------------------------------------------------------------- #
# materials
# --------------------------------------------------------------------------- #


def test_osf_materials_are_present_and_verified():
    manifest = json.loads(paths.OSF_MANIFEST.read_text())
    assert manifest["osf_project"] == "wv7c3"
    assert len(manifest["files"]) == 52
    assert all(entry["ok"] for entry in manifest["files"])
    assert len(list(paths.ARM_QSF_DIR.glob("*.qsf"))) == 18
    assert len(list(paths.ARM_DOCX_DIR.glob("*.docx"))) == 18


def test_every_arm_maps_to_an_export_and_to_a_condition_name():
    published = set(_responses()["condName"].dropna().unique())
    assert len(instrument.ARMS) == 18
    assert {arm.cond for arm in instrument.ARMS} == set(range(18))
    assert {arm.name for arm in instrument.ARMS} == published
    for arm in instrument.ARMS:
        assert paths.arm_qsf(arm.qsf).exists(), arm.qsf


# --------------------------------------------------------------------------- #
# the modality audit — the decision-relevant number
# --------------------------------------------------------------------------- #


def test_modality_audit_call():
    rows = instrument.modality_audit()
    assert len(rows) == 18
    usable = [row for row in rows if row["usable"] == "yes"]
    assert len(usable) == 11
    dropped = [row for row in rows if row["usable"] == "no"]
    assert sum(1 for row in dropped if row["modality"] == "video") == 5
    assert sum(1 for row in dropped if row["modality"] == "image_of_text") == 2
    # Every dropped arm has media to justify the call.
    for row in dropped:
        assert row["video"] or row["iframe"] or row["graphic"], row["condName"]
    # The control is kept despite its video, because that video is a knot-tying
    # clip; every other kept arm has no video at all bar one illustrative clip.
    assert instrument.BY_NAME["Control"].usable
    assert instrument.CONDITIONS[0] == instrument.CONTROL


def test_graphic_questions_are_counted_as_media():
    # Co-Benefits' whole stimulus is one Qualtrics graphic question, which carries
    # no <img> tag at all. Missing it would read the arm as 73 words of text.
    row = next(r for r in instrument.modality_audit() if r["condName"] == "CoBenefits")
    assert row["graphic"] == 1
    assert row["image"] == 0
    assert not instrument.BY_NAME["CoBenefits"].usable


def test_piped_media_is_counted():
    # Misperception Correction pipes six charts in from the survey flow, so a scan
    # of its question text alone finds no media whatsoever.
    row = next(
        r for r in instrument.modality_audit() if r["condName"] == "MispCorrectionRisks"
    )
    assert row["piped_image"] == 6


# --------------------------------------------------------------------------- #
# the instrument
# --------------------------------------------------------------------------- #


def test_battery_questions_all_exist_in_the_master_export():
    survey = instrument.master().survey
    for name, qids in instrument.BATTERY.items():
        for qid in qids:
            assert qid in survey.questions, (name, qid)


def test_dv_block_order_matches_the_published_column_order():
    # The nine block names come from DV_order in the published file; the canonical
    # order is the file's own column order, which is Qualtrics export order.
    order = list(instrument.DV_BLOCK_ORDER)
    seen = set()
    for value in _responses()["DV_order"].dropna().head(200):
        seen |= {part.strip() for part in str(value).split("|")}
    assert seen == set(order)
    columns = list(_responses().columns)
    assert columns.index("belief_1") < columns.index("petition")
    assert columns.index("petition") < columns.index("donation")
    assert columns.index("donation") < columns.index("conversation")
    assert order[0] == "BeliefandPolicySupport"
    assert order[-1] == "Commitment"


def test_embedded_data_resolves_the_misperception_content():
    arm = instrument.BY_NAME["MispCorrectionRisks"]
    embedded = instrument.flow_embedded(paths.arm_qsf(arm.qsf))
    assert "employment_text" in embedded
    text = render_template("", instrument.arm_elements(arm, random.Random(0)))
    # The correction paragraph is only reachable through the flow's embedded data.
    assert "lead to lower income and benefits" in text
    # Fields the flow declares without a value stay as echoes: they are filled in
    # at runtime from the respondent's own answer.
    assert "<<=img>>" in text


def test_display_logic_becomes_a_conditional_branch():
    arm = instrument.BY_NAME["CollEfficacyEmoBenefit"]
    text = render_template("", instrument.arm_elements(arm, random.Random(0)))
    assert text.count("<<?if") == 6
    assert text.count("<<?endif>>") == 6
    # Both halves of one branch pair are present, gated on the same question.
    assert "Wrong!" in text and "You are right!" in text


def test_page_breaks_inside_an_arm_come_from_the_export():
    arm = instrument.BY_NAME["DynamicAngerNorm"]
    state = instrument.arm_survey(arm.qsf)
    bid = instrument.arm_blocks(arm)[0][0]
    items = convert.block_items(state.blocks, bid)
    stated = sum(1 for kind, _ in items if kind == "Page Break")
    assert stated == 6
    text = render_template("", instrument.arm_elements(arm, random.Random(0)))
    assert text.count("- - - [ page") == stated + 1


# --------------------------------------------------------------------------- #
# templates
# --------------------------------------------------------------------------- #


def test_templates_render_for_every_usable_arm():
    for arm_name in instrument.CONDITIONS:
        elements = instrument.elements_for(
            arm_name, battery=list(instrument.DV_BLOCK_ORDER), rng=random.Random(0)
        )
        text = render_template(instrument.header("p", arm_name), elements)
        assert len(text) > 15_000, arm_name
        # No Qualtrics pipe survives into a transcript: a literal ${...} is a
        # token sequence no respondent ever saw.
        assert "${" not in text, arm_name
        # Every response position is three lines ending in "Response: ".
        assert text.count("Response: ") == len(slot_manifest(elements))


def test_slot_ids_are_unique_and_prompts_carry_no_markers():
    for arm_name in instrument.CONDITIONS:
        elements = instrument.elements_for(
            arm_name, battery=list(instrument.DV_BLOCK_ORDER), rng=random.Random(0)
        )
        slots = slot_manifest(elements)
        ids = [slot["id"] for slot in slots]
        assert len(ids) == len(set(ids)), arm_name
        for slot in slots:
            assert not MARKER_RE.search(slot["prompt"]) or "<<=" in slot["prompt"]


def test_slot_ids_are_published_columns():
    published = _published_columns()
    # De-identified out of the published file: the letter text and the zip code.
    allowed_gaps = {"letter_content", "zipcode_1"}
    for arm_name in instrument.CONDITIONS:
        arm = instrument.BY_NAME[arm_name]
        elements = instrument.elements_for(
            arm_name, battery=list(instrument.DV_BLOCK_ORDER), rng=random.Random(0)
        )
        for slot in slot_manifest(elements):
            if slot["id"].startswith(f"{arm.slug}__"):
                continue  # intervention-internal: never scored, never published
            assert slot["id"] in published | allowed_gaps, (arm_name, slot["id"])


def test_intervention_slots_cannot_shadow_an_outcome():
    published = _published_columns()
    for arm_name in instrument.CONDITIONS:
        arm = instrument.BY_NAME[arm_name]
        elements = instrument.elements_for(arm_name, rng=random.Random(0))
        for element in instrument.arm_elements(arm, random.Random(0)):
            for inner in element.elements:
                if isinstance(inner, Slot):
                    assert inner.id.startswith(f"{arm.slug}__")
                    assert inner.id not in published
        assert elements


def test_render_all_writes_the_expected_file_set():
    manifest = templates.render_all()
    files = sorted(p.name for p in paths.TEMPLATES.glob("*.txt"))
    assert len(files) == 11
    assert files[0] == "00_control.txt"
    assert manifest["n_arms"] == 18
    assert manifest["n_arms_usable"] == 11
    assert paths.FORMAT_DOC.exists()
    assert paths.MODALITY_AUDIT.exists()
    written = json.loads(paths.MANIFEST.read_text())
    assert set(written["arms"]) == {arm.name for arm in instrument.ARMS}
    for name, entry in written["arms"].items():
        assert entry["usable"] == (name in instrument.CONDITIONS)
        if entry["usable"]:
            assert (paths.TEMPLATES / entry["file"]).exists()


# --------------------------------------------------------------------------- #
# outcomes
# --------------------------------------------------------------------------- #


def test_derived_columns_reproduce():
    table = outcomes.verify_against_published(_responses())
    assert len(table) == 18
    assert bool(table["matches"].all()), table[~table["matches"]].to_string()
    # Nothing gained or lost a value in the recomputation.
    assert int(table["n_missing_mismatch"].sum()) == 0


def test_derive_items_is_idempotent_on_the_published_file():
    frame = _responses()
    once = outcomes.derive_items(frame)
    twice = outcomes.derive_items(once)
    for column in ("petition", "newsletter1", "newsletter2", "video", "newsletter"):
        left = pd.to_numeric(once[column], errors="coerce")
        right = pd.to_numeric(twice[column], errors="coerce")
        assert left.equals(right), column


def test_unreliable_items_are_excluded_from_the_scored_set():
    for column in ("belief_1", "policy_1", "bankscore"):
        assert column in outcomes.BY_COLUMN
        assert column not in outcomes.SCORED
    assert set(outcomes.COMPOSITES) <= set(outcomes.SCORED)


def test_coverage_reports_the_zero_spike_on_start_at_zero_sliders():
    frame = outcomes.compute(_responses())
    table = outcomes.coverage(frame).set_index("column")
    # A slider that starts at 0 records every non-mover as an exact 0.
    assert table.loc["march", "at_scale_floor"] > 2000
    # A binary item has no floor to report.
    assert pd.isna(table.loc["petition", "at_scale_floor"])
    # Three columns were zero-filled to every row by their construction.
    for column in ("newsletter", "donation_bin", "letter"):
        assert table.loc[column, "share_observed"] > 0.99


# --------------------------------------------------------------------------- #
# scoring
# --------------------------------------------------------------------------- #


def test_human_reference_covers_every_usable_arm():
    humans = score.load_humans()
    assert humans["condition"].nunique() == 11
    reference, human1, human2 = score.human_reference(humans)
    assert len(human1) + len(human2) == len(humans)
    assert set(human1.index).isdisjoint(set(human2.index))
    # Ten interventions against the control, on eleven outcomes.
    assert reference["condition"].nunique() == 10
    assert reference["outcome"].nunique() == len(outcomes.SCORED)
    assert len(reference) == 110


def test_human_replication_beats_the_null_baseline():
    board, _ = score.leaderboard()
    row = board.set_index("submission")
    human = row.loc["Human replication (Human 2)"]
    null = row.loc["Baseline: no effect"]
    assert human["directional_pct"] > 80
    assert human["rmse"] < null["rmse"]
    assert human["pearson_r"] > 0.5


def test_position_effects_reproduce_the_papers_reported_swings():
    table = score.position_effects().set_index("outcome")
    # The paper reports march falling from about 47 to about 39 and pol_campaign
    # from 56 to 48 between the first and last display position.
    assert 46 < table.loc["march", "mean_first_position"] < 49
    assert 37 < table.loc["march", "mean_last_position"] < 41
    assert table.loc["pol_campaign", "slope_per_position"] < -0.8
    # And the two reverse-labelled items moving the other way, which is one of the
    # three reasons they are not scored.
    assert table.loc["belief_1", "slope_per_position"] > 0.5


def test_control_levels_pin_the_two_pfander_anchors():
    table = score.control_levels().set_index("outcome")
    donation = table.loc["donation"]
    assert 4.5 < donation["mean"] < 5.0
    assert donation["scale_max"] == 10.0
    assert 0.15 < table.loc["newsletter", "mean"] < 0.30


# --------------------------------------------------------------------------- #
# profiles and slot grammars
# --------------------------------------------------------------------------- #


def test_profiles_are_deterministic_and_balanced():
    first = profiles.build(seed=7, per_arm=12)
    second = profiles.build(seed=7, per_arm=12)
    assert [p.profile_id for p in first] == [p.profile_id for p in second]
    assert [p.condition for p in first] == [p.condition for p in second]
    assert len(first) == 12 * len(instrument.CONDITIONS)
    counts = {name: 0 for name in instrument.CONDITIONS}
    for profile in first:
        counts[profile.condition] += 1
    assert set(counts.values()) == {12}


def test_prefilled_answers_are_legal_for_their_slots():
    profile = profiles.build(seed=7, per_arm=1)[0]
    elements = instrument.elements_for(profile.condition, rng=random.Random(0))
    by_id = {}
    for event, payload in _walk(elements):
        if isinstance(payload, Slot):
            by_id[payload.id] = payload
    for slot_id, answer in profile.prefilled.items():
        slot = by_id.get(slot_id)
        assert slot is not None, slot_id
        assert slot.parse(str(answer)) is not None, (slot_id, answer)


def _walk(elements):
    from silicon_sampling.survey.render import walk

    return walk(elements)


def test_donation_slot_accepts_dollars_and_rejects_out_of_range():
    elements = instrument.battery_elements("Donation")
    slot = next(e for e in elements if isinstance(e, IntSlot))
    assert slot.id == "donation"
    assert slot.parse("$7") == 7
    assert slot.parse("0") == 0
    assert slot.parse("10") == 10
    assert slot.parse("11") is None


def test_newsletter_slots_are_the_two_organisations():
    elements = instrument.battery_elements("Newsletter")
    slots = [e for e in elements if isinstance(e, ChoiceSlot)]
    assert [slot.id for slot in slots] == ["newsletter1", "newsletter2"]
    for slot in slots:
        assert slot.options == ("Yes", "No")
    # The signup forms themselves were live third-party panels, and the transcript
    # says so rather than pretending they were pictures.
    text = render_template("", elements)
    assert "live web panel embedded here from 350.org" in text
    assert "citizensclimatelobby.org" in text


def test_pfander_anchors_name_real_columns_on_both_sides():
    from silicon_sampling.pfander import outcomes as pf

    published = _published_columns()
    for anchor in outcomes.PFANDER_ANCHORS:
        assert anchor.goldwert_column in published, anchor.goldwert_column
        assert anchor.pfander_outcome in pf.OUTCOMES, anchor.pfander_outcome
        assert anchor.closeness in {
            "near_identical",
            "adjacent",
            "conceptual",
            "unusable",
        }
    usable = {
        a.goldwert_column for a in outcomes.PFANDER_ANCHORS if a.closeness != "unusable"
    }
    # Every usable anchor is something a silicon sample is actually scored on.
    assert usable <= set(outcomes.SCORED)
    # The two near-identical rows are the donation and the newsletter, and both
    # match their Pfänder counterpart's scale exactly.
    near = {
        a.goldwert_column: a
        for a in outcomes.PFANDER_ANCHORS
        if a.closeness == "near_identical"
    }
    assert set(near) == {"donation", "newsletter"}
    assert pf.SCALE_RANGE["donation_ams"] == outcomes.SCORED["donation"] == 10.0
    assert pf.SCALE_RANGE["newsletter_signup"] == outcomes.SCORED["newsletter"] == 1.0
    # Six Pfänder outcomes have nothing here, and none of them is claimed.
    assert len(outcomes.PFANDER_UNCOVERED) == 6
    claimed = {a.pfander_outcome for a in outcomes.PFANDER_ANCHORS}
    assert claimed.isdisjoint(outcomes.PFANDER_UNCOVERED)


def test_anchor_levels_separate_reached_from_zero_filled():
    table = outcomes.anchor_levels(_responses()).set_index("condName")
    assert len(table) == 18
    control = table.loc["Control"]
    # The zero-filled newsletter column understates the signup rate by about ten
    # points, because it counts everyone who never reached the page as a refusal.
    assert control["newsletter_rate_reached"] > control["newsletter_rate_zero_filled"]
    assert 0.30 < control["newsletter_rate_reached"] < 0.33
    assert 0.20 < control["newsletter_rate_zero_filled"] < 0.23
    # The best arm's donation beats the control by well under a dollar.
    assert 0.5 < table["donation_mean"].max() - control["donation_mean"] < 1.0


def test_us_coverage_shows_there_is_nothing_to_filter_on():
    assert score.US_FILTER is None
    table = score.us_coverage()
    assert len(table) == 18
    assert int(table["n_assigned"].sum()) == 31324
    assert table["n_assigned"].between(1733, 1745).all()
    # Every column that mentions the United States reaches less of the sample
    # than the sample itself, which is why none of them is used as a filter.
    assert int(table["n_country_is_us"].sum()) == 10568
    assert int(table["n_region_known"].sum()) == 20668


def main() -> int:
    tests = [
        value
        for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    ]
    failures = 0
    for test in tests:
        try:
            test()
        except Exception as error:  # noqa: BLE001 - a test runner reports everything
            failures += 1
            print(f"FAIL  {test.__name__}: {error}")
        else:
            print(f"ok    {test.__name__}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
