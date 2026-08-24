"""Tests for the ICPC study package.

Five things are worth testing here and they are all cheap.  The instrument has to
*drive* — every echo resolvable before it is shown, no marker reaching a model,
every slot accepting its own legal answers.  The slot ids have to name real
columns in the published export, because that is what makes a sampled answer
comparable.  And the four outcomes have to reproduce the second publication's
cleaned columns exactly, because the whole calibration rests on them.

The other two were added after an audit found the first three insufficient, and
both are about the *binding* between a transcript and the published data rather
than about whether anything runs:

* **Which item goes with which column.**  Naming a real column is not the
  requirement; naming the *right* one is.  A battery transcribed in the wrong
  order names 24 real columns and files every answer under a neighbour, and no
  battery-level check can see it because a mean is invariant to permuting its
  items.  So the binding is re-derived here from the ``.qsf``, cross-checked
  against ``codebook.xlsx``, and sanity-checked against the published per-item
  means, for every battery rather than for a spot check.
* **Which screens a respondent is shown.**  Sixteen questions carry Qualtrics
  ``DisplayLogic``, and a transcript that renders them unconditionally produces
  answer patterns that occur zero times in 8,253 real respondents — including in
  the effort outcome, which is a count of exactly those answers.  So what the
  transcript gates is checked against what the ``.qsf`` gates, and the click
  patterns synthetic respondents actually produce are checked for the monotonicity
  the survey enforced.

The data-dependent tests skip rather than fail when the 168 MB export is absent,
so the package stays testable in a checkout without ``data/``.
"""

from __future__ import annotations

import csv
import json
import random
import re
from types import SimpleNamespace

import pandas as pd
import pytest

from silicon_sampling.icpc import convert
from silicon_sampling.icpc import export

# Aliased: an existing test below binds ``run`` as a loop variable.
from silicon_sampling.icpc import run as icpc_run
from silicon_sampling.icpc import instrument as inst
from silicon_sampling.icpc import outcomes as oc
from silicon_sampling.icpc import paths, profiles
from silicon_sampling.icpc import templates as tpl
from silicon_sampling.icpc import validate
from silicon_sampling.icpc.images import IMAGE_ALT
from silicon_sampling.icpc import cli as icpc_cli
from silicon_sampling.sampling.tokens import verify
from silicon_sampling.survey.render import MARKER_RE, render_template, slot_manifest

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


requires_qsf = pytest.mark.skipif(
    not paths.QSF.exists(), reason="instrument .qsf not vendored"
)
requires_data = pytest.mark.skipif(
    not paths.DOELL_CSV.exists(), reason="published export not present"
)
requires_codebook = pytest.mark.skipif(
    not paths.CODEBOOK.exists(), reason="published codebook not vendored"
)
requires_stimuli = pytest.mark.skipif(
    not (paths.STIMULI / "index.json").exists(), reason="stimuli not vendored"
)


# --------------------------------------------------------------------------- #
# the arm table
# --------------------------------------------------------------------------- #


def test_twelve_arms_with_unique_codes_and_slugs():
    assert len(inst.ARMS) == 12
    assert [arm.code for arm in inst.ARMS] == list(range(1, 13))
    assert len({arm.key for arm in inst.ARMS}) == 12
    assert len({arm.slug for arm in inst.ARMS}) == 12
    assert inst.CONDITIONS[0] == inst.CONTROL


def test_the_two_renamed_arms_carry_both_names():
    assert inst.BY_KEY["Letter2Future"].alias == "LetterFutureGen"
    assert inst.BY_KEY["Identity-Social-Norms-Intervention"].alias == "WorkTogetherNorm"
    # Every other arm is named the same in both publications.
    same = [arm for arm in inst.ARMS if arm.key == arm.alias]
    assert len(same) == 10


# --------------------------------------------------------------------------- #
# the instrument
# --------------------------------------------------------------------------- #


@requires_qsf
def test_every_arm_assembles_and_renders():
    for arm in inst.ARMS:
        converted = inst.assemble(arm)
        text = render_template(inst.header("p00001", arm), converted.elements)
        assert text.count("\nResponse:") == len(slot_manifest(converted.elements))
        assert len(text) > 20_000


@requires_qsf
def test_control_reads_its_filler_before_the_shared_definition():
    keys = [
        block.key
        for block in inst.assemble("Control").elements
        if hasattr(block, "key")
    ]
    filler = inst._key(inst.BY_KEY["Control"].block)
    assert keys.index(filler) < keys.index("intro")


@requires_qsf
def test_only_the_control_arm_gets_the_extra_measure_blocks():
    control = {
        block.key
        for block in inst.assemble("Control").elements
        if hasattr(block, "key")
    }
    treated = {
        block.key
        for block in inst.assemble("SciConsens").elements
        if hasattr(block, "key")
    }
    assert {"control_ivs", "terms_probing"} <= control
    assert not {"control_ivs", "terms_probing"} & treated


@requires_qsf
def test_every_response_position_has_a_question_stem():
    # Three lines per response position: a numbered stem (which may itself run to
    # several lines), an indented statement of the legal answers, then Response:.
    for arm in inst.ARMS:
        text = render_template(inst.header("p", arm), inst.assemble(arm).elements)
        lines = text.splitlines()
        responses = [i for i, line in enumerate(lines) if line.startswith("Response:")]
        assert responses
        for index in responses:
            assert lines[index - 1].startswith("      "), lines[index - 1]
        numbered = [line for line in lines if re.match(r"Q\d+\. ", line)]
        assert len(numbered) == len(responses), arm.key


@requires_qsf
def test_slot_ids_are_unique_within_an_arm():
    for arm in inst.ARMS:
        slots = slot_manifest(inst.assemble(arm).elements)
        ids = [slot["id"] for slot in slots]
        assert len(ids) == len(set(ids)), arm.key


@requires_qsf
def test_echo_targets_are_answered_before_they_are_displayed():
    # `<<=id>>` is matched by word characters only, so any echo naming an id with
    # an R-mangled dot in it would survive rendering and never resolve.
    for arm in inst.ARMS:
        text = render_template(inst.header("p", arm), inst.assemble(arm).elements)
        for marker in MARKER_RE.findall(text):
            if marker.startswith("<<="):
                assert marker[3:-2].replace("_", "").isalnum(), marker


@requires_qsf
def test_every_stimulus_image_is_described():
    state = inst.loaded()
    for question in state.survey.questions.values():
        for key in _image_keys(question.raw_text):
            assert key in IMAGE_ALT, key


def _image_keys(raw: str) -> list[str]:
    keys = []
    for tag in re.findall(r"<img\b[^>]*>", raw or "", re.I):
        source = re.search(r'src="([^"]+)"', tag)
        url = source.group(1) if source else ""
        found = re.search(r"IM=(IM_\w+)", url)
        keys.append(found.group(1) if found else url)
    return keys


@requires_qsf
def test_modality_audit_keeps_all_twelve_arms():
    rows = inst.modality_rows()
    assert len(rows) == 12
    assert {row["decision"] for row in rows} == {"keep"}
    # The claim the audit exists to support: nothing but static images.
    for row in rows:
        assert row["video"] == 0 and row["audio"] == 0
        assert row["iframe"] == 0 and row["script"] == 0


# --------------------------------------------------------------------------- #
# driving it
# --------------------------------------------------------------------------- #


@requires_qsf
def test_every_arm_can_be_driven_to_the_end():
    runs = validate.dry_run_all()
    assert len(runs) == 12
    for run in runs:
        assert MARKER_RE.search(run.transcript) is None
        # Not "asked more than n questions": how many a respondent is asked now
        # depends on their own answers, because the gated screens really are
        # gated.  What has to hold is that nothing a scored outcome is built from
        # was skipped.
        for item in oc.BELIEF_ITEMS + oc.POLICY_ITEMS + ("Share", "WEPT1confirm"):
            assert item in run.answers, (run.condition, item)


@requires_qsf
def test_the_panel_record_can_be_switched_off():
    with_panel = validate.dry_run(profiles.build(per_arm=1)[0], panel_header=True)
    without = validate.dry_run(profiles.build(per_arm=1)[0], panel_header=False)
    assert "PARTICIPANT PANEL RECORD" in with_panel.transcript
    assert "PARTICIPANT PANEL RECORD" not in without.transcript


# --------------------------------------------------------------------------- #
# profiles
# --------------------------------------------------------------------------- #


def test_profiles_are_balanced_and_deterministic():
    first = profiles.build(per_arm=20)
    second = profiles.build(per_arm=20)
    assert [p.profile_id for p in first] == [p.profile_id for p in second]
    assert [p.condition for p in first] == [p.condition for p in second]
    counts = profiles.sanity(first)["per_arm"]
    assert set(counts.values()) == {20}


def test_profile_round_trips_through_csv(tmp_path):
    built = profiles.build(per_arm=3)
    path = tmp_path / "profiles.csv"
    profiles.write_csv(built, path)
    back = profiles.read_csv(path)
    assert [p.profile_id for p in back] == [p.profile_id for p in built]
    assert back[0].prefilled == built[0].prefilled


def test_prefilled_answers_cover_the_screened_items():
    profile = profiles.build(per_arm=1)[0]
    answers = profile.prefilled
    assert answers["AttentionCheck_purp"] == "Purple"
    assert answers["Attn_60"] == "sixty"
    assert answers["WEPTdemo1_1"] == "67, 85"
    assert answers["WEPTdemo2_1"] == "23, 81"
    assert answers["cond"] == profile.cond


def test_the_wept_demonstration_answers_are_actually_the_target_numbers():
    from silicon_sampling.vlasceanu.content_shared import _WEPT_DEMO_ROWS

    for row, expected in zip(_WEPT_DEMO_ROWS, ("67, 85", "23, 81")):
        targets = [
            number
            for number in row
            if (number // 10) % 2 == 0 and (number % 10) % 2 == 1
        ]
        assert ", ".join(str(number) for number in targets) == expected


# --------------------------------------------------------------------------- #
# outcomes
# --------------------------------------------------------------------------- #


def test_outcome_scales_are_the_published_ones():
    assert oc.OUTCOMES == {
        "belief": 100.0,
        "policy": 100.0,
        "sharing": 1.0,
        "wept": 8.0,
    }
    assert len(oc.BELIEF_ITEMS) == 4
    assert len(oc.POLICY_ITEMS) == 9
    assert len(oc.WEPT_ITEMS) == 8


def test_outcomes_are_computed_from_text_answers_too():
    frame = pd.DataFrame(
        {
            "Belief.in.CC_1": [100, 0],
            "Belief.in.CC_2": [80, 20],
            "Belief.in.CC_4": [60, 40],
            "Belief.in.CC_5": [40, 60],
            "Share": [
                "Yes, I am willing to share this information.",
                "I do not use social media.",
            ],
            **{f"WEPT{i}confirm": ["yes", "no"] for i in range(1, 9)},
        }
    )
    computed = oc.compute(frame)
    assert computed["belief"].tolist() == [70.0, 30.0]
    assert computed["sharing"].tolist()[0] == 1.0
    assert pd.isna(computed["sharing"].tolist()[1])
    assert computed["wept"].tolist() == [8.0, 0.0]


@requires_data
def test_slot_ids_map_onto_real_published_columns():
    header = pd.read_csv(paths.DOELL_CSV, encoding="latin-1", nrows=1, low_memory=False)
    columns = set(header.columns)
    mapped = inst.data_columns()
    assert len(mapped) > 140
    missing = {slot: column for slot, column in mapped.items() if column not in columns}
    assert missing == {}


@requires_data
def test_the_us_filter_selects_the_quota_subsample():
    from silicon_sampling.icpc import score

    frame = score.load_raw(columns=("ResponseId", "country", "condName", "teams"))
    us = score.us_subsample(frame)
    assert len(us) == 8253
    assert set(us["teams"]) == {"usa_1", "usa_2", "usa_3"}
    assert set(us["condName"]) == set(inst.CONDITIONS)


@requires_data
def test_outcomes_reproduce_the_cleaned_publication():
    from silicon_sampling.icpc.score import verify_outcomes

    table = verify_outcomes()
    assert set(table["outcome"]) == set(oc.OUTCOMES)
    assert table["matches"].all()
    assert (table["max_abs_diff"] == 0).all()


# --------------------------------------------------------------------------- #
# rendered artefacts
# --------------------------------------------------------------------------- #


@requires_qsf
def test_render_all_writes_the_expected_file_set(tmp_path):
    manifest = tpl.render_all(out_dir=tmp_path)
    assert len(manifest["arms"]) == 12
    assert (tmp_path / "00_FORMAT.md").exists()
    assert (tmp_path / "manifest.json").exists()
    assert (tmp_path / "modality_audit.csv").exists()
    assert len(list(tmp_path.glob("*.txt"))) == 12
    written = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    for entry in written["arms"].values():
        assert (tmp_path / entry["file"]).exists()
        assert entry["n_slots"] == len(entry["slots"])
        assert all(slot["data_column"] for slot in entry["slots"])
    with (tmp_path / "modality_audit.csv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 12


@requires_qsf
def test_the_three_us_instruments_agree_up_to_two_spellings():
    """usa_1/2/3 are the same survey; this pins down exactly how they differ."""
    from silicon_sampling.icpc import convert

    def stimuli(path):
        return {
            arm.key: re.sub(
                r"\s+",
                " ",
                render_template(
                    "H", convert.convert_qsf_block(arm.block, path=path).elements
                ),
            )
            for arm in inst.ARMS
            if arm.block
        }

    reference = stimuli(paths.QSF_ALL[-1])
    for path in paths.QSF_ALL[:-1]:
        other = stimuli(path)
        assert set(other) == set(reference)
        differing = {key for key in other if other[key] != reference[key]}
        # usa_1 alone spells "gray", "labor" and "droughts" where the other two
        # have "grey", "labour" and the master's typo "draughts"; usa_2 differs
        # from usa_3 only in invisible characters, which the squeeze removes.
        allowed = (
            {"Control", "PsychDistance", "FutureSelfCont"}
            if path.name == "usa_1.qsf"
            else set()
        )
        assert differing == allowed, (path.name, differing)


# --------------------------------------------------------------------------- #
# the item-to-column binding
# --------------------------------------------------------------------------- #


def _bound_wording() -> dict[str, str]:
    """Published column -> the statement the transcripts print beside it.

    Read off the assembled arms rather than out of the source, so what is checked
    is the artefact a sampler consumes.
    """
    bound: dict[str, str] = {}
    for arm in inst.ARMS:
        converted = inst.assemble(arm)
        for slot in slot_manifest(converted.elements):
            column = converted.data_columns.get(slot["id"], "")
            if column:
                bound[column] = slot["prompt"]
    return bound


def _squeeze(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", re.sub(r"\s+", " ", text.strip().lower()))


@requires_qsf
def test_every_battery_item_is_bound_to_the_column_the_qsf_gives_it():
    """The regression test for 24 items bound to a neighbouring column.

    The batteries were transcribed visually from a PDF whose text layer is
    corrupt, and the order came out permuted — so ``CC_policy_1`` asked about
    public transport while the column holds support for a carbon tax.  Nothing
    downstream could notice: the ids are real columns and the composites are
    means.  This holds every battery item to the ``.qsf``, where the export suffix
    *is* the choice code because none of these questions defines
    ``RecodeValues``, ``ChoiceDataExportTags`` or ``VariableNaming``.
    """
    authority = convert.qsf_item_wording()
    assert len(authority) > 40
    bound = _bound_wording()
    checked = 0
    for column, wording in authority.items():
        if column not in bound:
            continue
        checked += 1
        assert _squeeze(bound[column]) == _squeeze(wording), column
    assert checked > 40


@requires_qsf
@requires_codebook
def test_every_battery_item_matches_the_published_codebook():
    """The same binding against the second, independent authority.

    ``codebook.xlsx`` was produced from the live survey rather than from the
    ``.qsf``, so agreement with it rules out a misreading of the ``.qsf`` as
    readily as it rules out a misreading of the PDF.
    """
    codebook = convert.codebook_item_wording()
    bound = _bound_wording()
    checked = 0
    for column, wording in codebook.items():
        if column not in bound:
            continue
        checked += 1
        assert _squeeze(bound[column]) == _squeeze(wording), column
    assert checked > 40


#: Enviro_motiv splits into internally and externally motivated statements, and
#: the US sample endorses the internal ones far more strongly.  Keyed off the
#: *wording* so the classification follows whatever the binding claims; the
#: reverse-keyed item ("acting non-environmental is OK") belongs to neither pole
#: and is left out.
_INTERNAL = ("personally", "my personal values", "my self-concept")
_EXTERNAL = ("others", "politically correct")


@requires_qsf
@requires_data
def test_published_item_means_agree_with_the_wording_each_column_is_bound_to():
    """The published numbers, used as a third authority on the binding.

    This is the check the delivered verification lacked.  Comparing battery means
    cannot see a permutation; comparing *items* can, because an item's mean is a
    fact about its content.  Two facts suffice, and the permuted binding violated
    both: the least-supported of the nine policies is a tax on meat and dairy and
    the best-supported is protecting forests, and every internally motivated
    environmental-motivation item outscores every externally motivated one.
    """
    columns = list(oc.POLICY_ITEMS) + [
        f"Enviro_motiv_{i}" for i in (1, 11, 12, 13, 14, 15, 16, 17, 18, 20)
    ]
    frame = pd.read_csv(
        paths.DOELL_CSV,
        encoding="latin-1",
        low_memory=False,
        usecols=["country"] + columns,
    )
    us = frame[frame["country"] == "usa"]
    means = {name: pd.to_numeric(us[name], errors="coerce").mean() for name in columns}
    bound = _bound_wording()

    policy = {name: means[name] for name in oc.POLICY_ITEMS}
    assert "meat" in bound[min(policy, key=policy.get)]
    assert "forested" in bound[max(policy, key=policy.get)]

    internal, external = [], []
    for name in columns:
        if not name.startswith("Enviro_motiv"):
            continue
        wording = bound[name].lower()
        if " ok" in wording:
            continue
        if any(mark in wording for mark in _INTERNAL):
            internal.append(means[name])
        elif any(mark in wording for mark in _EXTERNAL):
            external.append(means[name])
    assert len(internal) == 4 and len(external) == 5
    assert min(internal) > max(external)


@requires_data
def test_the_per_item_check_covers_every_belief_and_policy_item():
    from silicon_sampling.icpc.score import verify_items

    table = verify_items()
    assert len(table) == len(oc.BELIEF_ITEMS) + len(oc.POLICY_ITEMS)
    assert set(table["item"]) == set(oc.BELIEF_ITEMS + oc.POLICY_ITEMS)
    assert table["matches"].all()
    assert (table["max_abs_diff"] == 0).all()


# --------------------------------------------------------------------------- #
# what a response position tells a model about the answer it wants
# --------------------------------------------------------------------------- #


def _int_slots(elements):
    """Every :class:`IntSlot` in an assembled arm, conditionals included."""
    from silicon_sampling.survey.slots import IntSlot

    for element in elements:
        inner = getattr(element, "elements", None)
        if inner is not None:
            yield from _int_slots(inner)
        elif isinstance(element, IntSlot):
            yield element


@requires_qsf
def test_every_slider_states_the_range_a_legal_answer_falls_in():
    """The regression test for the defect that cost this study its first run.

    The project's slider convention printed the endpoint labels the respondent saw
    and nothing else, which is faithful to the screen and useless to a model: the
    screen also showed a 0-100 track.  Asked for a number with no range given, the
    models answered on a small scale -- 80% to 94% of every 0-100 answer came back
    as an integer of ten or less against 8% to 31% for real participants -- and
    mean control-arm level error ran 20 to 47 points.  Nothing in the templates
    looked wrong, because a missing sentence looks like a shorter sentence.  So the
    range is asserted on every response position that wants an integer, not on a
    sample of them, and both bounds are checked against the slot's own.
    """
    total = 0
    for arm in inst.ARMS:
        for slot in _int_slots(inst.assemble(arm).elements):
            total += 1
            assert f"from {slot.lo} to {slot.hi}." in (slot.anchors or ""), (
                arm.key,
                slot.id,
                slot.anchors,
            )
    # 245 slider positions across the twelve arms, plus the 12 `Age` boxes (one
    # per arm, prefilled but still an integer position) and the two text boxes
    # Qualtrics validated as numbers -- see the number-box test below. The earlier
    # comment here read "257 sliders ... plus the two text boxes", which reached
    # the right total by counting `Age` as a slider.
    assert total == 259, total


@requires_qsf
def test_a_number_box_asks_for_a_number_and_says_which_numbers():
    """A Qualtrics text box carries its type in ``Validation``, not in its type.

    ``negEmo_cliThreshTime`` and ``1.5 Threshold`` are ``TE``/``SL``, the same
    question type as a comment box, and only
    ``Validation.Settings.ContentType == "ValidNumber"`` says the screen refused
    anything but a number.  Rendered as free text they let a synthetic respondent
    answer "about four years, I think" into a column whose 726 and 692 human
    answers are numeric without a single exception.  So every validated number box
    in a ``.qsf``-rendered block is asserted to be an integer position, and to
    state its range -- the same sentence the sliders now carry.
    """
    from silicon_sampling.survey.slots import IntSlot

    state = inst.loaded()
    found = 0
    for arm in inst.ARMS:
        if arm.block is None:
            continue
        block = state.by_description[arm.block]
        for entry in state.block_elements[block.bid]:
            question = state.survey.questions.get(entry.get("QuestionID"))
            if question is None or question.kind != "TE":
                continue
            payload = state.payloads.get(question.qid, {})
            if convert.numeric_bounds(question, payload) is None:
                continue
            found += 1
            produced, _ = convert.convert_question(question, payload, {})
            assert len(produced) == 1, question.export_tag
            slot = produced[0]
            assert isinstance(slot, IntSlot), (question.export_tag, type(slot))
            assert f"from {slot.lo} to {slot.hi}." in slot.anchors, question.export_tag
    assert found == 2, found


@requires_qsf
@requires_data
def test_no_human_answered_a_validated_number_box_with_prose():
    """The premise of the test above, taken from the data rather than assumed."""
    import pandas as pd

    columns = ["country", "negEmo_cliThreshTime", "X1.5.Threshold"]
    frame = pd.read_csv(
        paths.DOELL_CSV,
        encoding="latin-1",
        usecols=lambda name: name in columns,
        low_memory=False,
    )
    united_states = frame[frame["country"].astype(str).str.lower() == "usa"]
    for column in ("negEmo_cliThreshTime", "X1.5.Threshold"):
        answered = united_states[column].dropna()
        assert len(answered) > 500, (column, len(answered))
        assert pd.to_numeric(answered, errors="coerce").notna().all(), column


@requires_qsf
def test_a_not_applicable_box_is_stated_exactly_where_qualtrics_showed_one():
    """An ``NA`` *label* is not an ``NA`` *box*, and only the box was on screen.

    Qualtrics keeps the escape option's wording in ``Labels["NA"]`` whether or not
    the option is switched on; switching it on is ``Configuration.NotApplicable``.
    Sixteen control-arm sliders here carry the label with the box off, so a
    transcript that read the label would offer every one of them an escape the
    respondent never had -- and a transcript that ignored the flag entirely would
    drop the escape from the nine policy items, which is a scored outcome.  The
    hand transcription tracks the flag; this is what holds it there.
    """
    from silicon_sampling.vlasceanu import content_shared as shared
    from silicon_sampling.vlasceanu import elements as V
    from silicon_sampling.vlasceanu.country import UNITED_STATES

    state = inst.loaded()
    shown: dict[str, str | None] = {}
    for question in state.survey.questions.values():
        if question.kind != "Slider":
            continue
        payload = state.payloads.get(question.qid, {})
        labels = payload.get("Labels") or {}
        configuration = payload.get("Configuration") or {}
        escape = None
        if isinstance(labels, dict) and configuration.get("NotApplicable"):
            escape = (labels.get("NA") or {}).get("Display")
        rows = list(zip(question.choices, question.codes or question.choices))
        for _, code in rows:
            shown[convert.data_column(question, code)] = escape

    blocks = [
        shared.BELIEF,
        shared.POLICY,
        shared.CONTROL_EXTRA_IVS,
        shared.terms_probing_block(0),
        shared.demographics_block(UNITED_STATES),
    ]
    checked = 0
    for block in blocks:
        for screen in block.screens:
            for element in screen.elements:
                if isinstance(element, V.Slider):
                    items = [element.slot]
                elif isinstance(element, V.Matrix):
                    items = [slot for slot, _ in element.items]
                else:
                    continue
                for slot in items:
                    if slot not in shown:
                        continue
                    checked += 1
                    assert (element.extra or None) == shown[slot], (
                        slot,
                        element.extra,
                        shown[slot],
                    )
    assert checked == 37, checked


def _flat(text: str) -> str:
    """Normalised for comparison: one authority uses curly quotes, one straight."""
    for pair in (
        "\u2019'",
        "\u2018'",
        '\u201c"',
        '\u201d"',
        "\u2013-",
        "\u2014-",
        "\xa0 ",
    ):
        text = text.replace(pair[0], pair[1])
    return re.sub(r"\s+", " ", text).strip().lower()


@requires_qsf
def test_every_line_of_every_stimulus_reaches_the_transcript():
    """No screen, paragraph or option of a manipulation is silently dropped.

    Every other check here is about a response *position* -- does it exist, does it
    gate, does it name the right column.  None of them would notice a stimulus
    screen going missing, because a screen with no question on it has no slot and
    contributes nothing countable; and eleven of the twelve arms are nothing but
    such screens.  So the ``.qsf``'s own text is walked chunk by chunk and looked
    for in the rendered template, which is the one assertion that fails if a page
    break, an ``<img>`` or a paragraph stops being emitted.

    The two Qualtrics pipes are excluded on purpose: they are the only text that is
    *supposed* to differ, having become an echo of this respondent's own answer.
    """
    state = inst.loaded()
    for arm in inst.ARMS:
        if arm.block is None:
            continue
        rendered = _flat(
            render_template(inst.header("p", arm), inst.assemble(arm).elements)
        )
        block = state.by_description[arm.block]
        for entry in state.block_elements[block.bid]:
            question = state.survey.questions.get(entry.get("QuestionID"))
            if question is None or question.kind in {"Timing", "Meta"}:
                continue
            visible = convert.strip_html(convert.with_images(question.raw_text))
            chunks = [c for c in visible.split("\n") if len(c.strip()) > 25]
            for chunk in chunks + [c for c in question.choices if c]:
                if "${" in chunk:
                    continue
                assert _flat(chunk) in rendered, (arm.key, question.export_tag, chunk)


#: ``.qsf`` block -> the hand-transcribed blocks that stand in for it, for the
#: wording check below.  ``DEMOGRAPHICS`` maps to two of them because the
#: debriefing form is the tail of that block rather than a block of its own, and
#: ``WEPTdemo`` to one because the eight ``WEPTpage`` blocks are checked with it.
SHARED_AGAINST_QSF: tuple[tuple[tuple[str, ...], tuple[str, ...]], ...] = (
    (("CONSENT FORM",), ("consent",)),
    (("Climate Change Information Overview for all",), ("intro",)),
    (("Belief in AnthrCC",), ("belief",)),
    (("Policy Support",), ("policy",)),
    (("Social media sharing Piped Text",), ("sharing",)),
    (
        ("WEPTdemo", *(f"WEPTpage{page}" for page in range(1, 9))),
        ("wept",),
    ),
    (("1. Control Condition IVs",), ("control_ivs",)),
    (("AttentionCheck_60", "DEMOGRAPHICS"), ("demographics", "debrief")),
)


@requires_qsf
def test_the_shared_screens_are_worded_the_way_the_qsf_words_them():
    """The transcription is a *transcription*, and the ``.qsf`` is what it copies.

    The shared screens were read visually off a PDF whose text layer is corrupt,
    and every other check here is blind to a wrong word: the slot ids are right,
    the columns are right, the batteries bind correctly, the sessions drive.  A
    transcription that says "a greenhouse gases" where the survey said "a
    greenhouse gas" passes all of it.  So every sentence the ``.qsf`` holds is
    looked for in the assembled screens, which is the only assertion that fails
    when a hand edit drifts.

    Two kinds of text are excluded because they are *meant* to differ: the
    ``[IMAGE: ...]`` lines, whose words are ours and not the survey's, and the one
    sentence carrying a Qualtrics pipe, which becomes this respondent's own
    condition code.
    """
    state = inst.loaded()
    assembled: dict[str, str] = {}
    for arm in inst.ARMS:
        for block in inst.assemble(arm).elements:
            key = getattr(block, "key", None)
            if key is None or key in assembled:
                continue
            assembled[key] = _flat(render_template("", [block]))

    for descriptions, keys in SHARED_AGAINST_QSF:
        rendered = " ".join(assembled[key] for key in keys)
        for description in descriptions:
            block = state.by_description[description]
            for entry in state.block_elements[block.bid]:
                question = state.survey.questions.get(entry.get("QuestionID"))
                if question is None or question.kind in {"Timing", "Meta"}:
                    continue
                visible = convert.strip_html(question.raw_text)
                for chunk in visible.split("\n"):
                    if len(chunk.strip()) <= 25 or "${" in chunk:
                        continue
                    flat = _flat(chunk.lstrip("\u2022 "))
                    assert flat in rendered, (
                        description,
                        question.export_tag,
                        chunk[:120],
                    )


# --------------------------------------------------------------------------- #
# the effortful task
# --------------------------------------------------------------------------- #


@requires_qsf
def test_the_wept_grids_are_the_numbers_the_qsf_holds():
    """The sixty numbers on each effort page, checked against their own source.

    A Qualtrics Profile matrix stores its rows as empty ``Choices`` and its cells
    as nested ``Answers``, so ``question.choices`` on a WEPT grid is six empty
    strings and the grid looks absent from the ``.qsf``.  It is not, which means the
    one part of this instrument that was believed to be unverifiable by hand is
    verifiable -- and a grid transcribed from a PDF is exactly the kind of thing
    that acquires a wrong digit silently, because nothing downstream reads it.

    Rows are compared as a *set*: every grid sets ``Randomization`` to ``All``, so
    the row order a respondent saw was their own and no fixed order can be right.
    """
    from silicon_sampling.vlasceanu import content_shared as shared

    state = inst.loaded()

    def grid(tag: str) -> set[tuple[int, ...]]:
        question = next(
            q for q in state.survey.questions.values() if q.export_tag == tag
        )
        answers = state.payloads[question.qid]["Answers"]
        rows = set()
        for row in sorted(answers, key=int):
            cells = answers[row]
            rows.add(tuple(int(cells[c]["Display"]) for c in sorted(cells, key=int)))
        return rows

    assert set(shared._WEPT_DEMO_ROWS) == grid("WEPTdemo1") | grid("WEPTdemo2")
    for page in range(1, 9):
        transcribed = set(shared._WEPT_PAGE_ROWS[page])
        assert len(transcribed) == 6, page
        assert transcribed == grid(f"WEPT{page}nums"), page


# --------------------------------------------------------------------------- #
# display logic
# --------------------------------------------------------------------------- #


@requires_qsf
def test_the_transcript_gates_exactly_what_qualtrics_gated():
    """The regression test for sixteen ungated conditional questions.

    The display logic used to live only in an English sentence on the screen, and
    the one consumer that tried to read it back out matched seven of the fifteen
    WEPT screens and neither of the sharing ones.  A missed gate fails silently:
    the screen simply renders for everyone.  So the set of gated response
    positions is compared against ``DisplayLogic`` in the ``.qsf`` itself.
    """
    stems = convert.qsf_gated_stems()
    assert len(stems) == 16
    for arm in inst.ARMS:
        converted = inst.assemble(arm)
        for slot in slot_manifest(converted.elements):
            column = converted.data_columns.get(slot["id"], "")
            expected = convert.is_gated(column, stems)
            assert expected == ("shown_if" in slot), (arm.key, slot["id"])


@requires_qsf
def test_no_synthetic_respondent_produces_a_click_pattern_the_survey_forbids():
    """Sixty respondents, none of them impossible.

    The WEPT confirmations are a chain — page n is only offered to someone who
    accepted page n-1 — and the effort outcome is their sum, so an ungated chain
    does not merely add noise, it inflates the one outcome whose treatment effects
    run negative.  In the 8,253 US respondents there is not a single answer after
    a decline.
    """
    for profile in profiles.build(per_arm=5):
        run = validate.dry_run(profile)
        asked = [i for i in range(1, 9) if f"WEPT{i}confirm" in run.answers]
        assert asked == list(range(1, len(asked) + 1)), profile.profile_id
        declined = [i for i in asked if run.answers[f"WEPT{i}confirm"] == "no"]
        assert declined in ([], [asked[-1]]), profile.profile_id
        grids = [i for i in range(1, 9) if f"WEPT{i}nums_1" in run.answers]
        assert grids == [i for i in asked if run.answers[f"WEPT{i}confirm"] == "yes"]


@requires_qsf
def test_the_platform_question_is_only_asked_of_willing_sharers():
    """Nobody is asked which platform they posted on after refusing to post."""
    willing = "Yes, I am willing to share this information."
    seen_both = set()
    for profile in profiles.build(per_arm=5):
        run = validate.dry_run(profile)
        answered = "Share2" in run.answers
        assert answered == (run.answers.get("Share") == willing), profile.profile_id
        seen_both.add(answered)
    # Both branches have to occur, or the assertion above proves nothing.
    assert seen_both == {True, False}


# --------------------------------------------------------------------------- #
# what the verification reports
# --------------------------------------------------------------------------- #


def _toy_published(wept):
    """Two respondents, in the shape the cleaned extract has."""
    return pd.DataFrame(
        {
            "ResponseId": ["a", "b"],
            **{f"Belief{i}": [50, 50] for i in range(1, 5)},
            **{f"Policy{i}": [50, 50] for i in range(1, 10)},
            "SHAREcc": [1, 1],
            "WEPTcc": wept,
        }
    )


def _toy_raw(confirms):
    return pd.DataFrame(
        {
            "ResponseId": ["a", "b"],
            **{name: [50, 50] for name in oc.BELIEF_ITEMS + oc.POLICY_ITEMS},
            "Share": [1, 1],
            **confirms,
        }
    )


def test_the_outcome_check_counts_rows_only_one_side_has():
    """A row the study scored and we did not must not vanish from the comparison.

    Reporting only the intersection is how 18 respondents published as
    ``WEPTcc = 0`` hid behind a maximum absolute difference of zero and a
    ``matches`` of ``True``.
    """
    published = _toy_published([8, 0])
    scored = oc.verify_against_published(
        _toy_raw({f"WEPT{i}confirm": [1, None] for i in range(1, 9)}), published
    ).set_index("outcome")
    assert scored.loc["wept", "n_compared"] == 2
    assert scored.loc["wept", "only_published"] == 0
    assert bool(scored.loc["wept", "matches"])

    # A frame with no WEPT column at all: ours is missing for both rows, the
    # study scored both, and that has to be visible rather than masked away.
    hidden = oc.verify_against_published(_toy_raw({}), published).set_index("outcome")
    assert hidden.loc["wept", "n_compared"] == 0
    assert hidden.loc["wept", "only_published"] == 2
    assert not bool(hidden.loc["wept", "matches"])


def test_a_missing_wept_confirmation_counts_as_a_refusal():
    """What the study does with a chain that stopped early, and with no chain."""
    stopped = oc.compute(
        pd.DataFrame(
            {
                "WEPT1confirm": ["yes"],
                "WEPT2confirm": ["yes"],
                "WEPT3confirm": ["no"],
            }
        )
    )
    assert stopped["wept"].tolist() == [2.0]
    # Never reached the block: every flag present and missing.  The study
    # publishes 0 for these 18 respondents, so summing with ``min_count=1`` and
    # calling it missing disagreed with the publication and hid inside the
    # intersection of the comparison.
    never_reached = oc.compute(pd.DataFrame({name: [None] for name in oc.WEPT_ITEMS}))
    assert never_reached["wept"].tolist() == [0.0]
    # No such column at all, though, is a frame that says nothing about effort.
    absent = oc.compute(pd.DataFrame({"Share": ["I do not use social media."]}))
    assert pd.isna(absent["wept"]).all()


@requires_data
def test_the_polarity_the_module_docstring_claims_is_the_one_in_the_data():
    """The docstring is a delivered claim; hold it to the number it states."""
    from silicon_sampling.icpc import score

    effects = score.effects(score.load_humans())
    wept = effects[effects["outcome"] == "wept"]
    assert int((wept["estimate"] < 0).sum()) == 9
    assert set(wept.loc[wept["estimate"] > 0, "condition"]) == {
        "BindingMoral",
        "DynamicNorm",
    }
    assert "nine of them" in score.__doc__
    assert "eleven of eleven" not in score.__doc__


# --------------------------------------------------------------------------- #
# vendored stimuli
# --------------------------------------------------------------------------- #


@requires_stimuli
def test_the_stimulus_index_names_a_file_that_exists_for_everything_it_kept():
    """The index is the only record of which pictures survived; it has to be true.

    It used to list six CDN failures and none of the five recoveries, so a reader
    concluded six images were lost when only one is.
    """
    index = json.loads((paths.STIMULI / "index.json").read_text(encoding="utf-8"))
    assert isinstance(index, dict), "the index has to state its own counts"
    on_disk = {
        path.name for path in paths.STIMULI.iterdir() if path.name != "index.json"
    }
    named = set()
    counts = {"ok": 0, "recovered": 0, "lost": 0}
    for entry in index["images"]:
        counts[entry["status"]] += 1
        if entry["status"] == "lost":
            assert entry["file"] is None
            continue
        assert entry["file"] in on_disk, entry
        named.add(entry["file"])
    assert named == on_disk
    assert counts == {"ok": 51, "recovered": 5, "lost": 1}
    assert index["counts"] == {
        "referenced": sum(counts.values()),
        **counts,
        "files_on_disk": len(on_disk),
    }


# --------------------------------------------------------------------------- #
# profiles, continued
# --------------------------------------------------------------------------- #


def test_the_ideology_weights_are_ten_point_bins_not_deciles():
    """Deciles hold a tenth each; these hold between 5% and 22%."""
    assert not hasattr(profiles, "IDEOLOGY_DECILES")
    assert len(profiles.IDEOLOGY_BINS) == 10
    assert abs(sum(profiles.IDEOLOGY_BINS) - 1) < 1e-3
    assert max(profiles.IDEOLOGY_BINS) > 0.2


# --------------------------------------------------------------------------- #
# the sampling run
# --------------------------------------------------------------------------- #
#
# These drive the *real* runner with a stand-in generator.  What they are for is
# the class of break that only a session can show: a rendered template prints a
# marker where a session has to resolve it, so an echo naming a slot that does
# not exist, or an option whose own label the parser will not accept, renders
# perfectly and then fails on the GPU after the time has been spent.


class _StubEngine:
    """Answers whatever the pending slot accepts, with no model behind it.

    Prompts arrive as token ids, so the pending slot is recovered by *position*:
    :func:`~silicon_sampling.sampling.driver.run_group` builds its pending list in
    session order and hands ``generate`` the prompts in that same order.  The
    positional assumption holds only while every slot resolves in the first round,
    which is exactly the property being asserted — a slot that rejects its own
    legal answer trips the length assertion below rather than passing quietly.
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
            legal = str(validate.answer(slot, random.Random(param.seed)))
            draws = [legal] * param.n
            # One call in seven gets an illegal first draw, so the rejection
            # accounting and the draw log are exercised too.
            if param.seed % 7 == 0:
                draws[0] = "Yes | No | Not applicable"
            out.append(draws)
        return out


def _stub_sample(out_dir, profile_list, group_size=4):
    """Run the study's own Runner over ``profile_list`` with the stub engine."""
    from silicon_sampling.icpc.run import Runner
    from silicon_sampling.sampling import runner as runner_mod
    from silicon_sampling.sampling.driver import SamplerConfig
    from silicon_sampling.sampling.driver import run_group as real_run_group
    from silicon_sampling.sampling.engine import EngineConfig

    def wrapped(engine, sessions, *args, **kwargs):
        engine.sessions = sessions
        return real_run_group(engine, sessions, *args, **kwargs)

    # Budgets are supplied so the run needs no tokenizer when none is cached; with
    # one, prompts go as ids and the incremental tokeniser is exercised too.
    tokenizer = _tokenizer()
    budgets = (
        icpc_run.fit_token_budgets(TOKENIZER_MODEL)
        if tokenizer is not None
        else {
            slot_id: slot.max_tokens for slot_id, slot in icpc_run.all_slots().items()
        }
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
    """One respondent per arm, all the way to samples.csv."""
    built = profiles.build(seed=7, per_arm=1)
    meta = _stub_sample(tmp_path, built)
    assert meta["sampled"] == len(inst.ARMS)
    # Nothing was forced and nothing needed a grammar: every slot took a legal
    # answer on the first round.
    assert meta["draws"]["forced"] == 0
    assert meta["draws"]["structured_fallbacks"] == 0
    assert 0 < meta["draws"]["rejected"] < meta["draws"]["draws"]

    for path in ("answers.jsonl", "draws.jsonl", "run_meta.json"):
        assert (tmp_path / path).exists(), path
    # One transcript per respondent, filed under its arm's own slug.
    written = sorted(
        p.relative_to(tmp_path / "raw") for p in (tmp_path / "raw").rglob("*.txt")
    )
    assert len(written) == len(inst.ARMS)
    assert {p.parent.name for p in written} == {arm.slug for arm in inst.ARMS}

    summary = export.build_csvs(tmp_path)
    assert summary["rows"] == len(inst.ARMS)
    assert summary["arms"] == len(inst.ARMS)
    frame = pd.read_csv(tmp_path / "samples.csv")
    assert list(frame.columns)[-len(oc.OUTCOMES) :] == list(oc.OUTCOMES)
    for name in oc.OUTCOMES:
        assert frame[name].notna().any(), name


def test_no_transcript_carries_a_marker_or_an_unanswered_response(tmp_path):
    _stub_sample(tmp_path, profiles.build(seed=11, per_arm=1))
    for path in (tmp_path / "raw").rglob("*.txt"):
        text = path.read_text(encoding="utf-8")
        assert MARKER_RE.search(text) is None, path
        assert "\nResponse: \n" not in text and not text.endswith("Response: \n"), path


def test_the_run_resumes_from_the_answer_log(tmp_path):
    built = profiles.build(seed=13, per_arm=1)
    first = _stub_sample(tmp_path, built[:4])
    assert first["sampled"] == 4
    again = _stub_sample(tmp_path, built)
    assert (again["skipped"], again["sampled"]) == (4, len(built) - 4)
    assert len((tmp_path / "answers.jsonl").read_text().strip().splitlines()) == len(
        built
    )


def test_a_torn_final_record_is_truncated_before_anything_is_appended(tmp_path):
    built = profiles.build(seed=17, per_arm=1)
    _stub_sample(tmp_path, built[:2])
    with (tmp_path / "answers.jsonl").open("a", encoding="utf-8") as handle:
        handle.write('{"profile_id": "iTORN", "cond')
    _stub_sample(tmp_path, built[:4])
    ids = [
        json.loads(line)["profile_id"]
        for line in (tmp_path / "answers.jsonl").read_text().splitlines()
        if line.strip()
    ]
    assert ids == [p.profile_id for p in built[:4]]


def test_the_moderator_columns_speak_the_human_frames_vocabulary(tmp_path):
    """A subgroup table intersects levels; two vocabularies would intersect to nothing."""
    _stub_sample(tmp_path, profiles.build(seed=19, per_arm=1))
    export.build_csvs(tmp_path)
    frame = pd.read_csv(tmp_path / "samples.csv")
    from silicon_sampling.icpc import score as sc

    for moderator in sc.VISIBLE_MODERATORS:
        assert moderator in frame.columns, moderator
        levels = set(frame[moderator].dropna())
        assert levels, moderator
    assert set(frame["education"].dropna()) <= set(sc.EDUCATION_LABELS.values())
    assert set(frame["income_band"].dropna()) <= set(sc.INCOME_BANDS.values())
    assert set(frame["ideology_band"].dropna()) <= set(sc.IDEOLOGY_BREAKS[1])
    assert set(frame["age_band"].dropna()) <= set(sc.AGE_BREAKS[1])


def test_every_item_column_is_a_slot_and_no_echo_leaks_into_the_frame():
    """Echo-only keys are inputs, not responses, and must not become columns."""
    columns = export.item_columns()
    assert len(columns) == len(set(columns))
    assert set(columns) == set(icpc_run.all_slots())
    for leaked in ("panel_gender", "panel_age", "profile_id", "cond"):
        assert leaked not in columns


def test_the_worst_case_transcript_fits_the_context_the_cli_asks_for():
    """A cap below the longest arm does not slow the run down, it breaks it."""
    tokenizer = _tokenizer()
    if tokenizer is None:
        return
    longest = icpc_run.max_transcript_tokens(TOKENIZER_MODEL)
    assert longest < icpc_cli.MAX_MODEL_LEN, (longest, icpc_cli.MAX_MODEL_LEN)
    # And it is genuinely long: a cap sized for the Pfaender instrument would not
    # hold this one, which is the mistake this guards against.
    assert longest > 8192


def test_transcripts_tokenise_incrementally():
    """Submitting ids instead of text must be byte-identical, or the run is a lie."""
    tokenizer = _tokenizer()
    if tokenizer is None:
        return
    for session in icpc_run.worst_case_sessions():
        prompts = []
        while (step := session.next_prompt()) is not None:
            prompts.append(step[0])
            session.submit(step[1], icpc_run.widest_answer(step[1]))
        verify(tokenizer, prompts)


# --------------------------------------------------------------------------- #
# the range statement, the escape flag, and three screens that went missing
# --------------------------------------------------------------------------- #


@requires_qsf
def test_no_rendered_integer_position_hides_its_range():
    """The same regression guard as above, on the prose instead of on the slot.

    ``test_every_slider_states_the_range_a_legal_answer_falls_in`` reads
    ``slot.anchors``, which is where the sentence is *built*.  This one reads the
    rendered template, which is where a model *sees* it, and the two can disagree:
    a ``describe()`` override can append to ``anchors``, drop it, or state a range
    that is not the slot's own, and none of that shows up in an anchors check.  The
    defect being guarded is the one that cost this study its first run -- 80% to 94%
    of every 0-100 answer came back as an integer of ten or less against 8% to 31%
    for real participants -- and it is guarded twice because a missing sentence
    looks like a shorter sentence.

    Bounds come off the marker, so a position stating somebody else's range fails,
    and the count has to be exactly one, so a doubled statement fails as well.
    """
    checked = 0
    for arm in inst.ARMS:
        text = render_template(inst.header("p00001", arm), inst.assemble(arm).elements)
        lines = text.splitlines()
        for index, line in enumerate(lines):
            found = re.match(r"Response: <<[^:]+ :: int :: (-?\d+)\.\.(-?\d+)", line)
            if not found:
                continue
            prose = lines[index - 1]
            stated = f"from {found.group(1)} to {found.group(2)}"
            assert prose.count(stated) == 1, (arm.key, index + 1, prose)
            checked += 1
    # 245 slider positions, 12 `Age` boxes, 2 Qualtrics-validated threshold boxes.
    assert checked == 259, checked


@requires_qsf
def test_the_qsf_escape_note_follows_the_flag_and_not_the_label():
    """A slider can carry an ``NA`` label with the box switched off.

    Nine ``.qsf`` sliders have a ``Labels["NA"]`` entry and only five had
    ``Configuration.NotApplicable`` on, so handing ``Labels`` straight to
    ``anchors_from_labels`` invents a third scale point on ``Enviro_motiv``,
    ``Enviro_ID``, ``ID_hum`` and ``ID_GC`` and buries a real separate control
    inside the endpoint line on the other five.  The flag decides; the label only
    supplies the words.

    The words are then cross-checked against the hand-transcribed side, which
    reached the same screens independently: if the two ever disagree, one of them
    is describing a screen that did not exist.
    """
    state = inst.loaded()
    labelled = 0
    escapes = {}
    for qid, payload in state.payloads.items():
        if payload.get("QuestionType") != "Slider":
            continue
        labels = payload.get("Labels")
        if not (isinstance(labels, dict) and "NA" in labels):
            continue
        labelled += 1
        anchors, escape = convert.qsf_escape_note(payload)
        assert "Applicable" not in anchors and "Opinion" not in anchors, qid
        assert "Prefer not" not in anchors, qid
        tag = state.survey.questions[qid].export_tag
        escapes[tag] = escape
    assert labelled == 9, labelled
    # And the converter uses the flag, not just exposes it: an `NA` label with the
    # box off must not reach the slider's own prose line either.
    for qid, payload in state.payloads.items():
        if payload.get("QuestionType") != "Slider":
            continue
        labels = payload.get("Labels")
        if not (isinstance(labels, dict) and "NA" in labels):
            continue
        question = state.survey.questions[qid]
        produced, _ = convert.convert_question(question, payload, {})
        anchors = {slot.anchors for slot in produced if getattr(slot, "anchors", None)}
        assert anchors, question.export_tag
        wording = convert.qsf_escape_note(payload)[1]
        for line in anchors:
            assert ("the screen also offered" in line) == bool(wording), (
                question.export_tag,
                line,
            )
            if wording:
                assert f"'{wording}' box" in line, (question.export_tag, line)
            else:
                assert "Applicable" not in line, (question.export_tag, line)
    assert escapes["CC_policy"] == "Not Applicable"
    assert escapes["Politics2"] == "Prefer not to respond"
    for tag in ("Trust_sci1", "Trust_sci2", "Trust_gov"):
        assert escapes[tag] == "No Opinion", tag
    for tag in ("Enviro_motiv", "Enviro_ID", "ID_hum", "ID_GC"):
        assert escapes[tag] == "", tag


@requires_qsf
def test_no_template_renders_a_page_with_nothing_on_it():
    """A page break emitted twice is a screen the respondent never saw.

    ``convert_qsf_block`` opens with a ``PageBreak`` and then emits one per
    ``Page Break`` element, so a block whose own first element is a page break --
    arm 3's is -- produced an empty page 4, the only empty page in the twelve
    templates.  Checked across all twelve rather than on arm 3, because the cause is
    the block boundary and any arm can acquire one.
    """
    for arm in inst.ARMS:
        text = render_template(inst.header("p00001", arm), inst.assemble(arm).elements)
        lines = text.splitlines()
        rules = [
            i for i, line in enumerate(lines) if line.strip().startswith("- - - [")
        ]
        for start, stop in zip(rules, rules[1:] + [len(lines)]):
            body = [line for line in lines[start + 1 : stop] if line.strip()]
            assert body, (arm.key, lines[start])


@requires_qsf
def test_the_wept_practice_screen_says_it_could_not_be_got_wrong():
    """The practice grid was forced-correct, and that is why its answer is filled in.

    ``WEPTdemo1`` and ``WEPTdemo2`` are the only two questions in the ``.qsf`` with
    ``CustomValidation``: a nine-term ``And`` chain per item pinning every cell, so
    the page would not advance until the respondent had ticked exactly ``{67, 85}``
    and ``{23, 81}``.  Those answers are prefilled from
    :data:`~silicon_sampling.icpc.profiles.WEPT_DEMO_ANSWERS`, and without the
    sentence the transcript showed two correct answers appearing from nowhere.  The
    sentence lived on ``Screen.condition``, which ``convert_screens`` used only for
    *gated* screens, so it reached neither a template nor a session.
    """
    sentence = "could not be advanced until both rows were answered correctly"
    for arm in inst.ARMS:
        text = render_template(inst.header("p00001", arm), inst.assemble(arm).elements)
        assert sentence in text, arm.key
    # And it survives into a driven session, which block notes do not.
    walked = validate.dry_run(profiles.build(seed=5, per_arm=1)[0])
    assert sentence in walked.transcript


@requires_qsf
def test_the_panel_record_says_where_it_came_from_in_a_driven_session():
    """A caveat only a human reads is a caveat the model does not have.

    The panel record is our own addition -- this instrument asks its demographics
    last, so a respondent walking it in order knows nothing about itself until after
    every outcome -- and it said so in the block's ``note``.  ``Session.transcript()``
    strips block notes, so a driven session went from the file header straight to
    ``PARTICIPANT PANEL RECORD`` and the model saw seven answered questions with no
    account of why they were answered.
    """
    walked = validate.dry_run(profiles.build(seed=5, per_arm=1)[0])
    assert "PARTICIPANT PANEL RECORD" in walked.transcript
    assert inst.PANEL_PROVENANCE in walked.transcript
    # The block note is still there for a template reader, and still stripped.
    text = render_template(
        inst.header("p00001", inst.ARMS[0]), inst.assemble(inst.ARMS[0]).elements
    )
    assert "not a screen the participant filled in here" in text
    assert "not a screen the participant filled in here" not in walked.transcript
