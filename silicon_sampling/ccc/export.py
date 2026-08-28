"""``answers.jsonl`` -> ``samples.csv`` for the Climate Change Challenge.

Item columns are written **as answered on screen**, and composites are built with
the published reverse-coding applied, so the composites are directly comparable to
the human columns while the items stay a faithful record of what the model said.
Mixing the two conventions in one column set is how a sign error hides.
"""

from __future__ import annotations

import json

import pandas as pd

from . import outcomes as oc
from .paths import samples_dir

PROFILE_COLUMNS = (
    "profile_id",
    "condition",
    "gender",
    "age",
    "age_band",
    "race",
    "education",
    "party",
    "n_asked",
)


def _numeric(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def build_frame(records) -> pd.DataFrame:
    rows = []
    for record in records:
        answers = record.get("answers") or {}
        row = {name: record.get(name) for name in PROFILE_COLUMNS}
        row.update(answers)
        numeric = {k: _numeric(v) for k, v in answers.items()}
        for name in oc.COMPOSITES:
            row[name] = oc.composite(
                {k: v for k, v in numeric.items() if v is not None}, name
            )
        rows.append(row)
    return pd.DataFrame(rows)


def read_records(path):
    """One JSON record per line, splitting on newlines and nothing else.

    ``str.splitlines()`` is the obvious way to write this and it is wrong here: it
    also breaks on U+2028, U+2029, VT, FF and NEL. Model free text contains them —
    this run held six U+2028 LINE SEPARATORs — so ``splitlines()`` cut six records
    in half and JSON parsing died with "Unterminated string" on a file that was
    perfectly well formed. Iterating the handle splits on newlines only.
    """
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.rstrip("\n")
            if line.strip():
                yield json.loads(line)


def build_csv(run: str) -> dict:
    """Read one run's answers and write its analysis frame."""
    out_dir = samples_dir(run)
    path = out_dir / "answers.jsonl"
    if not path.exists():
        raise SystemExit(f"no answers at {path}")
    records = [
        json.loads(line) for line in path.read_text().splitlines() if line.strip()
    ]
    frame = build_frame(records)
    target = out_dir / "samples.csv"
    frame.to_csv(target, index=False)
    complete = {
        name: int(frame[name].notna().sum())
        for name in oc.SCORED
        if name in frame.columns
    }
    return {
        "rows": len(frame),
        "columns": frame.shape[1],
        "csv": str(target),
        "complete_per_scored_outcome": complete,
    }
