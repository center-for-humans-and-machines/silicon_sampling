"""``answers.jsonl`` -> ``samples.csv``, with the four outcomes computed.

One row per respondent, every answer as it was sampled, the four outcomes built by
:mod:`~silicon_sampling.icpc.outcomes`, and the five moderator columns
:mod:`~silicon_sampling.icpc.score` scores subgroups on.

The moderators are the part that needed a decision.  A synthetic respondent's
demographics exist twice over — once as the on-screen wording the transcript
carries (``"13-16 (college/undergraduate university/certificate training)"``) and
once as the band a subgroup table is cut on (``"College / undergraduate"``) — and
the two are *not* interchangeable, because ``score.subgroup_table`` intersects the
human frame's levels with ours and an intersection of two different vocabularies
is empty.  A silently empty subgroup table looks exactly like a study with no
subgroup signal, so the bands are derived here from the same code tables
:mod:`~silicon_sampling.icpc.score` maps the published columns through, and the
on-screen wording is kept beside them under its own name rather than overwritten.

Nothing is dropped for being prefilled.  The demographics, the consent item and
the WEPT demonstration are inputs rather than responses, but this instrument asks
them on screen and their columns exist in the published export, so they belong in
the frame; the echo-only keys that never were questions (``panel_*``, the profile
id, the condition code) are the ones that do not, and they are excluded by
building the column list from the instrument's own slots rather than from whatever
keys the answer log happens to hold.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ..survey.render import slot_manifest
from . import instrument as inst
from . import outcomes as oc
from . import profiles as prof
from . import score as sc

#: Columns carried straight off the profile that produced the respondent.
META_COLUMNS = (
    "profile_id",
    "condition",
    "cond",
    "gender",
    "age",
    "age_band",
    "education_onscreen",
    "income",
    "ses_ladder",
    "politics_social",
    "politics_economic",
    "battery",
    "extras",
    "probe_index",
    "n_asked",
)

#: Bands cut here so a subgroup table has the same level vocabulary on both sides.
MODERATOR_COLUMNS = ("education", "income_band", "ideology_band")

#: On-screen wording -> published code, inverted from the tables the profiles draw
#: their demographics with.  Inverting rather than restating them is what keeps
#: this file from becoming a third place the code tables are written down.
EDUCATION_CODES = {label: code for code, label in prof.EDUCATION_OPTIONS.items()}
INCOME_CODES = {label: code for code, label in prof.INCOME_OPTIONS.items()}


def read_answers(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def item_columns() -> list[str]:
    """Every slot id, in the order it first appears across the twelve arms.

    The control arm's terms-probing item is one of nine wordings under nine
    different ids, and each respondent got one, so all nine are visited: a column
    that exists for one respondent in nine still has to exist in the frame.
    """
    order: list[str] = []
    for arm in inst.ARMS:
        probes = range(inst.PROBE_WORDINGS) if arm.code == 1 else (0,)
        for probe in probes:
            elements = inst.elements_for(arm, probe_index=probe)
            for entry in slot_manifest(elements):
                if entry["id"] not in order:
                    order.append(entry["id"])
    return order


def add_moderators(frame: pd.DataFrame) -> pd.DataFrame:
    """The five columns ``score.VISIBLE_MODERATORS`` names, on the human side's terms."""
    data = frame.copy()
    breaks, labels = sc.AGE_BREAKS
    data["age_band"] = pd.cut(
        pd.to_numeric(data["age"], errors="coerce"), breaks, labels=labels
    ).astype(str)
    data["education"] = (
        data["education_onscreen"].map(EDUCATION_CODES).map(sc.EDUCATION_LABELS)
    )
    data["income_band"] = data["income"].map(INCOME_CODES).map(sc.INCOME_BANDS)
    mean_ideology = (
        pd.to_numeric(data["politics_social"], errors="coerce")
        + pd.to_numeric(data["politics_economic"], errors="coerce")
    ) / 2
    breaks, labels = sc.IDEOLOGY_BREAKS
    data["ideology_band"] = pd.cut(mean_ideology, breaks, labels=labels).astype(str)
    return data


def build_frame(records: list[dict]) -> pd.DataFrame:
    """The analysis frame: profile columns, item answers, moderators, outcomes."""
    items = item_columns()
    rows = []
    for record in records:
        row = {
            key: record.get(key) for key in META_COLUMNS if key != "education_onscreen"
        }
        row["education_onscreen"] = record.get("education")
        answers = record.get("answers", {})
        row.update({item: answers.get(item) for item in items})
        rows.append(row)
    frame = pd.DataFrame(rows, columns=list(META_COLUMNS) + items)
    frame = add_moderators(frame)
    frame = oc.compute(frame)
    columns = list(META_COLUMNS) + list(MODERATOR_COLUMNS) + items + list(oc.OUTCOMES)
    return frame[columns].sort_values("profile_id").reset_index(drop=True)


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
            for name in oc.OUTCOMES
        },
        "outcome_means": {
            name: round(
                float(pd.to_numeric(frame[name], errors="coerce").mean()),
                4,
            )
            for name in oc.OUTCOMES
        },
        "samples_csv": str(path),
    }
