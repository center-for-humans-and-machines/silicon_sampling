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

import functools
import json
import pathlib
import random
import re
import sys
import tempfile
from types import SimpleNamespace

import pandas as pd

from silicon_sampling.goldwert import cli as goldwert_cli
from silicon_sampling.goldwert import convert, instrument, outcomes, paths, profiles
from silicon_sampling.goldwert import export, run, score, templates, validate
from silicon_sampling.sampling.tokens import verify
from silicon_sampling.survey.render import MARKER_RE, render_template, slot_manifest
from silicon_sampling.survey.slots import ChoiceSlot, IntSlot, Slot

#: Model whose tokenizer the sampling checks below measure with.  The weights are
#: never loaded -- only the tokenizer -- and the checks skip themselves when even
#: that is not cached locally, which is how the suite stays runnable in a checkout
#: without a model.
TOKENIZER_MODEL = "Qwen/Qwen2.5-7B"


def _tokenizer(model: str = TOKENIZER_MODEL):
    try:
        from silicon_sampling.sampling.tokens import load_tokenizer

        return load_tokenizer(model)
    except Exception as error:  # pragma: no cover - depends on the local cache
        print(f"  (skipped: no tokenizer for {model}: {type(error).__name__})")
        return None


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


# --------------------------------------------------------------------------- #
# the sampling run
# --------------------------------------------------------------------------- #
#
# These drive the *real* runner with a stand-in generator, and they exist because
# every break this package has had was invisible to a template render.  A template
# prints `<<=slot>>` where a session has to *resolve* it, and prints an option list
# where a session has to *parse* one, so an echo naming a slot that does not exist,
# a label the parser cannot match, and a numeric box typed as free text all render
# perfectly and then fail -- or worse, quietly distort -- on the GPU.


def test_every_usable_arm_drives_to_the_end():
    """Every echo resolves, no marker escapes, every slot takes its own answer."""
    runs = validate.dry_run_all()
    assert len(runs) == len(instrument.CONDITIONS)
    assert {r.condition for r in runs} == set(instrument.CONDITIONS)
    for one in runs:
        assert MARKER_RE.search(one.transcript) is None, one.condition
        assert one.n_asked > 25, (one.condition, one.n_asked)


def test_no_choice_option_rejects_itself():
    """An option carrying a newline can never be matched, so the scale dies.

    ``ThreatInjustEfficacy``'s seven-point fairness item arrived from Qualtrics as
    ``"Completely unfair <br> 1"`` and so on.  ``ChoiceSlot`` truncates a draw at
    the first newline before matching it, so all seven options were unreachable and
    rejection sampling would have recorded the forced default for every respondent.
    """
    for slot_id, slot in run.all_slots().items():
        if isinstance(slot, ChoiceSlot):
            for option in slot.options:
                assert "\n" not in option, (slot_id, option)
                assert slot.parse(option) == option, (slot_id, option)


def test_every_echo_names_a_slot_that_exists():
    """The check a template render cannot make, because it prints the marker."""
    echo = re.compile(r"<<=([A-Za-z0-9_]+)>>")
    for name in instrument.CONDITIONS:
        elements = instrument.elements_for(
            name, battery=list(instrument.DV_BLOCK_ORDER), rng=random.Random(0)
        )
        ids = {entry["id"] for entry in slot_manifest(elements)}
        for echoed in set(echo.findall(render_template("", elements))):
            assert echoed in ids or echoed in run.PIPED_STAND_INS, (name, echoed)


def test_the_percentage_boxes_are_numbers_and_not_prose():
    """Three boxes are echoed back as "You guessed X%", so X has to be a number."""
    slots = run.all_slots()
    for tail in ("genderQ_6", "CCworryQ1_1", "CCdiscussQ2_1"):
        slot_id = f"linking_individual_structural__{tail}"
        slot = slots[slot_id]
        assert isinstance(slot, IntSlot), slot_id
        assert (slot.lo, slot.hi) == (0, 100), slot_id


def test_the_one_bipolar_slider_accepts_negative_answers():
    """A -100..100 scale whose negative half the parser refuses is half a scale."""
    signed = [
        slot
        for slot in run.all_slots().values()
        if isinstance(slot, IntSlot) and slot.lo < 0
    ]
    assert signed, "the hope/anger valence slider should be here"
    for slot in signed:
        assert slot.parse("-40") == -40
        assert slot.parse(str(slot.lo)) == slot.lo
        assert slot.parse(str(slot.lo - 1)) is None
        assert str(slot.lo) in slot.describe()


class _StubEngine:
    """Answers whatever the pending slot accepts, with no model behind it.

    Prompts arrive as token ids, so the pending slot is recovered by *position*:
    :func:`~silicon_sampling.sampling.driver.run_group` builds its pending list in
    session order and hands ``generate`` the prompts in that same order.  The
    positional assumption holds only while every slot resolves in the first round,
    which is exactly the property worth asserting -- a slot that rejects its own
    legal answer trips the length check below instead of passing quietly.  That is
    how the bipolar slider above was found.
    """

    def __init__(self, config):
        self.config = config
        self.sessions = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return None

    def kv_cache_tokens(self):
        return 500_000

    def group_size_for(self, tokens_per_session, safety=0.95, cap=128):
        return max(1, min(cap, int(500_000 * safety // tokens_per_session)))

    def params(self, *, max_tokens, n, seed, structured=None):
        return SimpleNamespace(max_tokens=max_tokens, n=n, seed=seed)

    def generate(self, prompts, params):
        pending = [s for s in self.sessions if s.next_prompt() is not None]
        assert len(pending) == len(prompts), (
            "a slot did not resolve in the first round, which means it rejects "
            "its own legal answers"
        )
        out = []
        for session, param in zip(pending, params):
            slot = session.next_prompt()[1]
            draws = [str(validate.answer(slot, random.Random(param.seed)))] * param.n
            if param.seed % 7 == 0:
                draws[0] = "Yes | No | I would rather not say"
            out.append(draws)
        return out


def _stub_sample(out_dir, profile_list, group_size=4):
    """Run the study's own Runner over ``profile_list`` with the stub engine."""
    from silicon_sampling.goldwert.run import Runner
    from silicon_sampling.sampling import runner as runner_mod
    from silicon_sampling.sampling.driver import SamplerConfig
    from silicon_sampling.sampling.driver import run_group as real_run_group
    from silicon_sampling.sampling.engine import EngineConfig

    def wrapped(engine, sessions, *args, **kwargs):
        engine.sessions = sessions
        return real_run_group(engine, sessions, *args, **kwargs)

    tokenizer = _tokenizer()
    budgets = (
        run.fit_token_budgets(TOKENIZER_MODEL)
        if tokenizer is not None
        else {slot_id: slot.max_tokens for slot_id, slot in run.all_slots().items()}
    )
    saved = runner_mod.VLLMEngine, runner_mod.run_group
    runner_mod.VLLMEngine, runner_mod.run_group = _StubEngine, wrapped
    try:
        runner = Runner(
            out_dir,
            EngineConfig(model=TOKENIZER_MODEL),
            SamplerConfig(
                group_size=group_size,
                draws_per_call=4,
                token_id_prompts=tokenizer is not None,
                max_tokens_by_slot=budgets,
            ),
        )
        return runner.run(profile_list)
    finally:
        runner_mod.VLLMEngine, runner_mod.run_group = saved


def test_every_arm_can_be_sampled_and_exported(tmp_path):
    built = profiles.build(seed=7, per_arm=1)
    meta = _stub_sample(tmp_path, built)
    assert meta["sampled"] == len(instrument.CONDITIONS)
    assert meta["draws"]["forced"] == 0
    assert meta["draws"]["structured_fallbacks"] == 0
    assert 0 < meta["draws"]["rejected"] < meta["draws"]["draws"]

    for name in ("answers.jsonl", "draws.jsonl", "run_meta.json"):
        assert (tmp_path / name).exists(), name
    written = sorted((tmp_path / "raw").rglob("*.txt"))
    assert len(written) == len(instrument.CONDITIONS)
    assert {p.parent.name for p in written} == {
        instrument.BY_NAME[name].slug for name in instrument.CONDITIONS
    }
    for path in written:
        text = path.read_text(encoding="utf-8")
        assert MARKER_RE.search(text) is None, path
        assert not text.endswith("Response: \n"), path

    summary = export.build_csvs(tmp_path)
    assert summary["rows"] == len(instrument.CONDITIONS)
    assert summary["arms"] == len(instrument.CONDITIONS)
    frame = pd.read_csv(tmp_path / "samples.csv")
    for name in outcomes.SCORED:
        assert name in frame.columns, name
        assert frame[name].notna().any(), name


def test_the_yes_no_outcomes_survive_the_trip_through_the_published_coding(tmp_path):
    """Answering "Yes" has to end up as 1, not as a silent zero.

    The published file stores these as Qualtrics codes 4 and 5, and the study's
    cleaning step recodes 4 to 1.  A frame still holding the on-screen text goes
    through ``pd.to_numeric`` as ``NaN``, the recode never fires, and every
    newsletter signup is recorded as a refusal.
    """
    _stub_sample(tmp_path, profiles.build(seed=23, per_arm=2))
    export.build_csvs(tmp_path)
    frame = pd.read_csv(tmp_path / "samples.csv")
    for column in ("petition", "newsletter1", "newsletter2", "newsletter"):
        values = set(pd.to_numeric(frame[column], errors="coerce").dropna())
        assert values <= {0.0, 1.0}, (column, values)
        assert values, column
    # `newsletter` is the OR of the two, so it cannot sit below either.
    assert frame["newsletter"].mean() >= frame["newsletter1"].mean()
    assert frame["newsletter"].mean() >= frame["newsletter2"].mean()


def test_the_letter_column_exists_so_political_advocacy_is_not_all_missing(tmp_path):
    """Without it the composite is NaN for everyone; with it, it is scoreable."""
    _stub_sample(tmp_path, profiles.build(seed=29, per_arm=1))
    summary = export.build_csvs(tmp_path)
    frame = pd.read_csv(tmp_path / "samples.csv")
    assert frame["letter"].notna().all()
    assert frame["political_advocacy"].notna().all()
    assert 0.0 <= summary["letter_rate"] <= 1.0
    assert export.letter_code("I care about climate change and want action now.") == 1.0
    assert export.letter_code("no") == 0.0
    assert export.letter_code("I would rather not answer this one, thank you.") == 0.0
    assert export.letter_code(None) == 0.0


def test_the_moderator_columns_speak_the_human_frames_vocabulary(tmp_path):
    """A subgroup table intersects levels; two vocabularies would intersect to nothing."""
    _stub_sample(tmp_path, profiles.build(seed=31, per_arm=2))
    export.build_csvs(tmp_path)
    frame = pd.read_csv(tmp_path / "samples.csv")
    for moderator in score.VISIBLE_MODERATORS:
        assert moderator in frame.columns, moderator
        assert set(frame[moderator].dropna()), moderator
    assert set(frame["education"].dropna()) <= set(score.EDUCATION.values())
    assert set(frame["age_band"].dropna()) <= set(score.AGE_BANDS[1])
    assert set(frame["party"].dropna()) <= set(profiles.PARTY_ONSCREEN.values())


def test_the_run_resumes_from_the_answer_log(tmp_path):
    built = profiles.build(seed=37, per_arm=1)
    first = _stub_sample(tmp_path, built[:4])
    assert first["sampled"] == 4
    again = _stub_sample(tmp_path, built)
    assert (again["skipped"], again["sampled"]) == (4, len(built) - 4)
    lines = (tmp_path / "answers.jsonl").read_text().strip().splitlines()
    assert len(lines) == len(built)


def test_a_torn_final_record_is_truncated_before_anything_is_appended(tmp_path):
    built = profiles.build(seed=41, per_arm=1)
    _stub_sample(tmp_path, built[:2])
    with (tmp_path / "answers.jsonl").open("a", encoding="utf-8") as handle:
        handle.write('{"profile_id": "gTORN", "cond')
    _stub_sample(tmp_path, built[:4])
    ids = [
        json.loads(line)["profile_id"]
        for line in (tmp_path / "answers.jsonl").read_text().splitlines()
        if line.strip()
    ]
    assert ids == [p.profile_id for p in built[:4]]


def test_the_piped_fields_never_reach_the_analysis_frame():
    """Page furniture the survey's own script set is not a response."""
    columns = export.item_columns()
    assert len(columns) == len(set(columns))
    for piped in run.PIPED_STAND_INS:
        assert piped not in columns


def test_the_worst_case_transcript_fits_the_context_the_cli_asks_for():
    """A cap below the longest arm does not slow the run down, it breaks it."""
    tokenizer = _tokenizer()
    if tokenizer is None:
        return
    longest = run.max_transcript_tokens(TOKENIZER_MODEL)
    assert longest < goldwert_cli.MAX_MODEL_LEN, (longest, goldwert_cli.MAX_MODEL_LEN)
    # And 8192 -- the value that looks safe -- would not have held it.
    assert longest > 7000


def test_transcripts_tokenise_incrementally():
    """Submitting ids instead of text must be byte-identical, or the run is a lie."""
    tokenizer = _tokenizer()
    if tokenizer is None:
        return
    for session in run.worst_case_sessions():
        prompts = []
        while (step := session.next_prompt()) is not None:
            prompts.append(step[0])
            session.submit(step[1], run.widest_answer(step[1]))
        verify(tokenizer, prompts)


def main() -> int:
    # The sampling checks take pytest's tmp_path fixture; under the plain runner
    # they are each handed a scratch directory of their own instead.
    scratch = pathlib.Path(tempfile.mkdtemp())
    tests = []
    for name, value in sorted(globals().items()):
        if not (name.startswith("test_") and callable(value)):
            continue
        wants_tmp = (
            "tmp_path" in value.__code__.co_varnames[: value.__code__.co_argcount]
        )
        tests.append(functools.partial(value, scratch / name) if wants_tmp else value)
        setattr(tests[-1], "__name__", name)
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
