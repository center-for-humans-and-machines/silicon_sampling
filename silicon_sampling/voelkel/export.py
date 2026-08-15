"""``answers.jsonl`` -> ``samples.csv``, with the nine outcomes computed."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from . import outcomes as oc

META = (
    "profile_id",
    "condition",
    "party_gen",
    "inparty",
    "gender",
    "race",
    "age",
    "age_band",
    "education",
    "ideology",
    "scenario",
    "n_asked",
)


def read_answers(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build_csv(run_dir: Path) -> dict:
    """Flatten the answer log, add the outcomes, write ``samples.csv``."""
    records = read_answers(run_dir / "answers.jsonl")
    rows = []
    for record in records:
        row = {key: record.get(key) for key in META}
        row.update(record["answers"])
        rows.append(row)
    frame = pd.DataFrame(rows).sort_values("profile_id").reset_index(drop=True)
    # Piped party strings are inputs, not responses; they would otherwise look
    # like sampled columns in the output.
    frame = frame.drop(
        columns=[
            c
            for c in (
                "Inparty_Person",
                "Inparty_Party",
                "Outparty_Person",
                "Outparty_Party",
                "Inparty_Elite",
                "Outparty_Friendship_ID",
            )
            if c in frame.columns
        ]
    )
    frame = oc.compute(frame, inparty="inparty")
    frame.to_csv(run_dir / "samples.csv", index=False)
    return {
        "rows": len(frame),
        "columns": len(frame.columns),
        "per_condition": frame["condition"].value_counts().to_dict(),
        "outcome_coverage": {
            name: int(frame[name].notna().sum()) for name in oc.OUTCOMES
        },
        "path": str(run_dir / "samples.csv"),
    }
