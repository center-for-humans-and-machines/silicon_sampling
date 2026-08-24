"""``answers.jsonl`` -> ``samples.csv``, with every scored outcome computed.

One row per respondent, the answers as sampled, the four preregistered composites
and the eleven scored outcomes from :mod:`~silicon_sampling.goldwert.outcomes`,
and the four moderator columns :mod:`~silicon_sampling.goldwert.score` cuts
subgroups on.

The whole design of this module is one decision: **the frame is put into the shape
of the published file, and then the study's own construction is run on it
unchanged.**  ``outcomes.compute`` is transcribed from the authors' cleaning
notebook and verified against their published columns; the moment we build a
composite a second way, the comparison it exists for stops meaning anything.  So
the translation happens on the inputs, and it is exactly two things.

**Choice answers become their Qualtrics codes.**  A transcript records what was on
screen — the petition item answers ``"Yes"`` — while the published file records the
code Qualtrics wrote, ``4``, which the cleaning script then recodes to ``1``.  Hand
``"Yes"`` to ``derive_items`` and ``pd.to_numeric`` returns ``NaN``, the ``4 -> 1``
recode never fires, and ``newsletter`` comes out ``0`` for every respondent who
said yes — a silent, total loss on one of the two outcomes this study exists to
anchor.  So every choice slot's answer is mapped back through its own code table.
Two columns are excluded because the authors published *them* as text and not as
codes: see :data:`TEXT_CODED`.  That inconsistency is theirs; reproducing it is
what makes a sampled row and a real row interchangeable.

**The letter is coded, and the coding is not the study's.**  ``letter`` in the
published file is a 0/1 judgement, made by GPT-3.5 and checked by hand, of whether
the free-text letter expressed clear thoughts about climate change; the text itself
was de-identified out of the file, so the classifier cannot be re-run or even
inspected.  :func:`letter_code` applies a stated keyword-and-length rule instead.
Its base rate does not match and cannot be made to, and the reason is not that our
rule is crude.

The human column is 0.4155 over all 31,324 rows, and it is *zero-filled*: only
23,575 respondents reached the letter screen, essentially every ``1`` is among
those, and 42% of the zeros are therefore people who never saw the question.  A
sampled respondent cannot drop out, so no rule over their text reproduces 0.4155
without fabricating attrition.  Nor is the optionality reproducible: the item is
Qualtrics ``RequestResponse``, which nags once and lets you click past, and the
*screen* never says the answer is optional — only the zip-code box below it does.
The permission to skip was a property of the survey software, not of the words on
the page, so putting "you may leave this blank" into the transcript would be adding
text no respondent read, and a slot that ends at ``"Response: "`` has nothing to
click past.  And "wrote nothing usable" cannot be detected either: the keyword rule
returns 1 for any on-topic sentence, which is what a model asked to write a letter
about climate change produces, so ours runs about 0.85.

What *is* available is honesty about the consequence, which
:func:`~silicon_sampling.goldwert.score.letter_contribution` measures on the human
data.  Holding ``letter`` at a constant — at 0.85, at 0.41, at anything — is an
affine transform of dropping it, so all three give the same answer: the arm-level
effects on ``political_advocacy`` correlate 0.82 (Spearman 0.72) with the real ones,
i.e. a third of the between-arm variance in that composite is carried by ``letter``
alone and nothing recovers it.  Worse than the attenuation is the direction: the
arms that gain most on the other three items gain *least* on ``letter``
(``LetterFuture`` is the study's strongest political-advocacy arm and its lowest
letter arm), so holding it constant inflates the mean absolute ATE by 69%, from
0.019 to 0.032.  Our 0.85 adds a level bias of +0.073 on a 0-1 scale, 0.520 to
0.593.

So the column is still filled — leaving it missing makes ``political_advocacy``
``NaN`` for every respondent and deletes a preregistered composite outright — and
``outcomes.compute`` now also reports ``political_advocacy_no_letter`` over the
three codeable members, so a reader can see the composite with and without the item
that is doing the damage.  The summary dict reports ``letter_rate`` so the size of
the departure is visible rather than inferred.

**``newsletter`` is zero-filled the same way, and nothing can be done about it.**
This is the second of the two outcomes the study exists to anchor, and it carries
exactly the defect just described for ``letter`` — the column is 1 for a signup, 0
for a refusal, and 0 again for the 4,140 of 19,141 kept-arm respondents who never
reached the page.  ``newsletter`` is a standalone member of :data:`SCORED` at
weight 1.0 *and* one of four in ``public_awareness``, and
:func:`~silicon_sampling.goldwert.score.effects` takes the all-rows mean, so the
attrition rides straight into both.

The direction is not ambiguous and it is not small.  Reach is 69.2% in the control
arm against 65.1%-83.3% in the treatments, so nine of the ten interventions
*raised* the share of people who got as far as the signup form; the published
all-rows ATEs therefore mix "more people signed up" with "more people were still
there to be asked".  Conditioning on reach cuts the mean absolute arm effect from
0.0372 to 0.0191 — the zero-fill **inflates** it by 1.95x — and reverses the sign
on four of the ten arms.  A silicon sample has reach 1.0 by construction, so it
can reproduce the reach-conditional effect and nothing else; scored against the
published all-rows effect it will look attenuated by about half, and on
``MispCorrectionRisks`` (+0.0405 -> -0.0006), ``HopeAngerNarratives`` (+0.0397 ->
-0.0048), ``IndStructuralChange`` (+0.0160 -> -0.0230) and
``EcologicalDisruptions`` (+0.0151 -> -0.0252) it will look *wrong-signed* while
being right.  Correlation between the two sets of arm effects is r = 0.45,
Spearman 0.61 — worse than ``letter``'s 0.82.  Our level will also sit high: the reached-only rate is 0.3183
against 0.2495 all-rows, so about +0.07 on a 0-1 scale, the same direction and
almost the same size as the ``letter`` level bias.
:func:`~silicon_sampling.goldwert.score.newsletter_contribution` is the
``letter_contribution`` counterpart that puts all of this on a table, and the
summary dict below reports ``newsletter_rate`` next to ``letter_rate`` so the
departure is visible at every build rather than inferred later.  ``donation_bin``
is zero-filled identically (r = 0.62) but is in no composite; ``video`` is
NaN-not-zero for non-reachers and is clean.

**The donation's constant-sum rule is enforced on the recorded pair, and the
draws are kept.**  ``QID34`` is a Qualtrics ``CS`` question with
``Validation: {"EnforceRange": "ON", "Type": "ChoicesTotal", "ChoiceTotal": "10"}``:
the survey refused a submission whose two boxes did not total ten, and all 23,732
non-null human rows total exactly ten.  The transcript can only *state* the rule,
because the driver samples one slot at a time and cannot hold a constraint across
two, and the models mostly honour it anyway — 98.4% of ``qwen25_72b`` rows, 96.0%
of ``v4_flash``, 93.1% of ``qwen25_7b``, against 9% for a stand-in that draws
each box uniformly at random (11 of the 121 pairs total ten).  For the minority that do not, the pair is reconciled the way the
survey reconciled it, to ``donation_keep = 10 - donation``: ``donation`` is the
scored outcome and the column the study reports, ``donation_keep`` is
definitionally ``10 - donation`` in the published file, and leaving the pair
inconsistent would put rows in ``samples.csv`` that the instrument could not have
produced.  The drawn value is not thrown away — it is kept as
``donation_keep_drawn``, and every row carries ``donation_sums_to_ten`` so an
analysis can condition on coherence instead of taking it on trust.

What reconciliation does **not** do is rescue ``donation`` itself.  A model that
answered 7 and 7, or 0 and 0, has misread a two-box allocation, and there is no
reason to think it got the first box right and only the second wrong; the 124 to
172 rows per run with a total of zero record "donated nothing" on the
highest-weighted outcome in the study and reconciliation leaves them at zero.
That is what the per-row flag is for.

Two smaller fidelity notes, stated here because ``samples.csv`` is where someone
will meet them.  The opt-out escapes on ``pol_candidate``, ``flyless``,
``lessbeef``, ``Politics_Soc`` and ``Politics_Econ`` **are** in the transcript and
**are** accepted — see
:class:`~silicon_sampling.goldwert.convert.EscapableIntSlot` — and record
:data:`~silicon_sampling.goldwert.convert.NOT_APPLICABLE`, which
``pd.to_numeric(..., errors="coerce")`` turns into the same ``NaN`` the published
file has.  An earlier version of this note claimed the escapes did not exist in
the transcript; they did, on the screen, and what did not exist was a slot that
would accept one, so a model following the printed instruction produced an
unparseable draw and the forced default recorded a 50.  What remains unmatched is
the *rate*: a synthetic respondent decides whether the item applies to it from a
one-line profile, and the human rates are 41% on ``flyless``, 8% on ``lessbeef``
and 5% on ``pol_candidate``.  And ``bankscore`` required looking up the
respondent's own real bank on a live site, so a sampled answer to it is a guess —
which then decides, through ``derive_items``, whose ``bank`` value survives into
``financial_advocacy``.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

from ..survey.render import slot_manifest
from ..survey.slots import ChoiceSlot
from . import instrument as inst
from . import outcomes as oc
from . import score as sc

#: Columns carried straight off the profile that produced the respondent.
META_COLUMNS = (
    "profile_id",
    "condition",
    "cond",
    "gender",
    "party",
    "age",
    "age_band",
    "education",
    "income",
    "ses",
    "battery",
    "n_asked",
)

#: Choice items the published file holds as on-screen text rather than as a code.
#: ``Edu``, ``Income`` and ``MacArthur_SES`` are numeric there and are recoded;
#: these two are not, and recoding them would break the join the slot ids exist to
#: make.
TEXT_CODED = ("Gender", "Party")

#: The derived column that has no slot behind it, and the words a letter has to
#: contain to be coded as being about climate change.  Deliberately short: a long
#: list would look like a calibrated classifier, and this is not one.
LETTER_COLUMN = "letter"
LETTER_MIN_CHARS = 40
LETTER_TERMS = re.compile(
    r"climate|global warming|warming|greenhouse|carbon|emission|fossil|"
    r"environment|renewable|clean energy|pollut",
    re.I,
)

#: The total the donation's two boxes had to reach before Qualtrics would accept
#: the page, read off ``QID34``'s ``Validation.Settings.ChoiceTotal``.
DONATION_TOTAL = 10
#: Where the drawn second box is kept once the pair has been reconciled, and the
#: per-row flag that says whether reconciling it changed anything.
DONATION_DRAWN = "donation_keep_drawn"
DONATION_COHERENT = "donation_sums_to_ten"

#: The two human ``newsletter`` rates over the eleven kept arms, printed beside
#: ours at every build.  The all-rows figure is what the published file and
#: :func:`~silicon_sampling.goldwert.score.effects` use; the reached-only figure
#: is the one a sample with reach 1.0 is actually comparable to.  Recomputed by
#: :func:`~silicon_sampling.goldwert.score.newsletter_contribution`, and pinned
#: here so a build that has no access to the human file still prints them.
HUMAN_NEWSLETTER_RATE_ALL_ROWS = 0.2495
HUMAN_NEWSLETTER_RATE_REACHED = 0.3183


def read_answers(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def item_columns() -> list[str]:
    """Every slot id, in the order it first appears across the usable arms."""
    order: list[str] = []
    for name in inst.CONDITIONS:
        elements = inst.elements_for(name, battery=list(inst.DV_BLOCK_ORDER))
        for entry in slot_manifest(elements):
            if entry["id"] not in order:
                order.append(entry["id"])
    return order


def choice_codes() -> dict[str, dict[str, int]]:
    """Slot id -> {on-screen option: Qualtrics code}, for every coded choice item."""
    tables: dict[str, dict[str, int]] = {}
    for name in inst.CONDITIONS:
        elements = inst.elements_for(name, battery=list(inst.DV_BLOCK_ORDER))
        for event, payload in _slots(elements):
            if event != "slot" or not isinstance(payload, ChoiceSlot):
                continue
            if payload.id in tables or payload.id in TEXT_CODED or not payload.codes:
                continue
            tables[payload.id] = {
                option: int(code)
                for option, code in zip(payload.options, payload.codes)
                if str(code).lstrip("-").isdigit()
            }
    return tables


def _slots(elements):
    from ..survey.render import walk

    return walk(elements)


def letter_code(text: object) -> float:
    """Whether a letter reads as clear thoughts about climate change.

    A stand-in for a classifier that cannot be re-run; see the module docstring for
    what it does and does not reproduce.
    """
    if not isinstance(text, str):
        return 0.0
    stripped = text.strip()
    if len(stripped) < LETTER_MIN_CHARS:
        return 0.0
    return 1.0 if LETTER_TERMS.search(stripped) else 0.0


def enforce_donation_total(frame: pd.DataFrame) -> pd.DataFrame:
    """Make the donation's two boxes total ten, and record whether they already did.

    The survey would not accept the page otherwise, and all 23,732 non-null human
    rows total exactly ten, so a sampled row that does not is a row the instrument
    could not have produced.  The driver cannot prevent it — it samples one slot at
    a time and has no way to hold a constraint across two — so the reconciliation
    happens here, on the recorded pair, and it always resolves in favour of
    ``donation``: that is the scored outcome, the column the study reports, and the
    one the published file defines ``donation_keep`` as the complement of.

    Nothing is destroyed.  The drawn second box survives as
    :data:`DONATION_DRAWN` and :data:`DONATION_COHERENT` marks the rows that
    needed no help, both as columns rather than as a summary statistic, because
    the rows that needed help are the rows whose ``donation`` is least
    trustworthy and an analysis has to be able to find them.
    """
    data = frame.copy()
    if "donation" not in data.columns or "donation_keep" not in data.columns:
        return data
    given = pd.to_numeric(data["donation"], errors="coerce")
    kept = pd.to_numeric(data["donation_keep"], errors="coerce")
    data[DONATION_DRAWN] = kept
    data[DONATION_COHERENT] = (given + kept == DONATION_TOTAL).astype(int)
    data["donation_keep"] = DONATION_TOTAL - given
    return data


def to_published_coding(frame: pd.DataFrame) -> pd.DataFrame:
    """Put the sampled answers into the coding the published file uses."""
    data = frame.copy()
    for column, table in choice_codes().items():
        if column in data.columns:
            data[column] = data[column].map(table)
    if "letter_content" in data.columns:
        data[LETTER_COLUMN] = data["letter_content"].map(letter_code)
    return enforce_donation_total(data)


def add_moderators(frame: pd.DataFrame) -> pd.DataFrame:
    """``age_band``, cut on the same breaks the human loader uses.

    The other three moderators — party, gender, education — come off the profile
    already worded the way :mod:`~silicon_sampling.goldwert.score` words them, so
    there is nothing to translate; the age band is the one that has to be cut.
    """
    data = frame.copy()
    data["age_band"] = pd.cut(
        pd.to_numeric(data["age"], errors="coerce"),
        sc.AGE_BANDS[0],
        labels=sc.AGE_BANDS[1],
    ).astype("object")
    return data


def build_frame(records: list[dict]) -> pd.DataFrame:
    """The analysis frame: profile columns, item answers, moderators, outcomes."""
    items = item_columns()
    rows = []
    for record in records:
        row = {key: record.get(key) for key in META_COLUMNS}
        answers = record.get("answers", {})
        row.update({item: answers.get(item) for item in items})
        rows.append(row)
    frame = pd.DataFrame(rows, columns=list(META_COLUMNS) + items)
    frame = add_moderators(to_published_coding(frame))
    frame = oc.compute(frame)
    derived = [LETTER_COLUMN] + [
        name for name in oc.NORMALIZED if name in frame.columns
    ]
    outcomes = [name for name in oc.SCORED if name not in items]
    columns = (
        list(META_COLUMNS)
        + items
        + derived
        + outcomes
        + [
            "donation_bin",
            "bank",
            "pos_emo",
            "neg_emo",
            # Per-row and not merely summarised: these two say which rows the
            # donation's constant-sum rule had to be imposed on, and those are
            # exactly the rows whose `donation` is least trustworthy.
            DONATION_DRAWN,
            DONATION_COHERENT,
        ]
    )
    seen: list[str] = []
    for column in columns:
        if column in frame.columns and column not in seen:
            seen.append(column)
    return frame[seen].sort_values("profile_id").reset_index(drop=True)


def build_csvs(out_dir: Path) -> dict:
    """Flatten the answer log, add the outcomes, write ``samples.csv``."""
    out_dir = Path(out_dir)
    frame = build_frame(read_answers(out_dir / "answers.jsonl"))
    path = out_dir / "samples.csv"
    frame.to_csv(path, index=False)
    return {
        "rows": len(frame),
        "columns": len(frame.columns),
        "arms": int(frame["condition"].nunique()),
        "per_condition": frame["condition"].value_counts().sort_index().to_dict(),
        "outcome_coverage": {
            name: int(pd.to_numeric(frame[name], errors="coerce").notna().sum())
            for name in oc.SCORED
        },
        "outcome_means": {
            name: round(float(pd.to_numeric(frame[name], errors="coerce").mean()), 4)
            for name in oc.SCORED
        },
        # The two rates below are the two columns this package fills by a rule of
        # its own rather than by the study's, and both departures have a known
        # sign. `letter` is a keyword stand-in for a classifier that no longer
        # exists: the human base rate is 0.4155 over all rows and ours runs near
        # 0.85. `newsletter` needs no stand-in but cannot match either, because
        # 4,140 of the 19,141 human rows are zeros contributed by people who never
        # reached the signup form and a sampled respondent cannot drop out; the
        # comparable human number is the reached-only 0.3183, not the published
        # all-rows 0.2495. Both are printed at every build so that neither has to
        # be rediscovered from the module docstring.
        "letter_rate": round(
            float(pd.to_numeric(frame[LETTER_COLUMN], errors="coerce").mean()), 4
        ),
        "newsletter_rate": round(
            float(pd.to_numeric(frame["newsletter"], errors="coerce").mean()), 4
        ),
        "human_newsletter_rate_reached": HUMAN_NEWSLETTER_RATE_REACHED,
        "human_newsletter_rate_all_rows": HUMAN_NEWSLETTER_RATE_ALL_ROWS,
        # How often the model honoured the constant-sum rule *before*
        # `enforce_donation_total` imposed it, and what the two failure shapes
        # cost. A total of zero records "donated nothing" on the highest-weighted
        # outcome in the study; a total of twenty is the model answering the same
        # question twice. Reconciliation fixes `donation_keep` and cannot fix
        # either.
        "donation_sums_to_ten": round(float(frame[DONATION_COHERENT].mean()), 4),
        "donation_reconciled_rows": int((frame[DONATION_COHERENT] == 0).sum()),
        "donation_drawn_total_zero": int(
            (
                pd.to_numeric(frame["donation"], errors="coerce")
                + pd.to_numeric(frame[DONATION_DRAWN], errors="coerce")
                == 0
            ).sum()
        ),
        "donation_drawn_total_twenty": int(
            (
                pd.to_numeric(frame["donation"], errors="coerce")
                + pd.to_numeric(frame[DONATION_DRAWN], errors="coerce")
                == 2 * DONATION_TOTAL
            ).sum()
        ),
        "samples_csv": str(path),
    }
