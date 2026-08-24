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

Two smaller fidelity notes, stated here because ``samples.csv`` is where someone
will meet them.  The ``"Not Applicable"`` escapes on ``pol_candidate``, ``flyless``
and ``lessbeef`` do not exist in the transcript, so those columns have no
missingness here where the human ones are 28-55% missing.  And ``bankscore``
required looking up the respondent's own real bank on a live site, so a sampled
answer to it is a guess — which then decides, through ``derive_items``, whose
``bank`` value survives into ``financial_advocacy``.
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


def to_published_coding(frame: pd.DataFrame) -> pd.DataFrame:
    """Put the sampled answers into the coding the published file uses."""
    data = frame.copy()
    for column, table in choice_codes().items():
        if column in data.columns:
            data[column] = data[column].map(table)
    if "letter_content" in data.columns:
        data[LETTER_COLUMN] = data["letter_content"].map(letter_code)
    return data


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
        + ["donation_bin", "bank", "pos_emo", "neg_emo"]
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
        # Reported because it is the one column built by a rule of our own rather
        # than by the study's: the human base rate is 0.41 and ours will not be.
        "letter_rate": round(
            float(pd.to_numeric(frame[LETTER_COLUMN], errors="coerce").mean()), 4
        ),
        # The donation is a Qualtrics *constant-sum* item: the survey refused a
        # submission whose two boxes did not total ten, and the transcript can
        # only state that rule in prose because the driver samples one slot at a
        # time and has no way to enforce a constraint across two. `donation` is
        # the scored outcome and is unaffected, but the share below says how often
        # the model honoured the constraint, which is the thing to watch.
        "donation_sums_to_ten": round(
            float(
                (
                    pd.to_numeric(frame["donation"], errors="coerce")
                    + pd.to_numeric(frame["donation_keep"], errors="coerce")
                    == 10
                ).mean()
            ),
            4,
        ),
        "samples_csv": str(path),
    }
