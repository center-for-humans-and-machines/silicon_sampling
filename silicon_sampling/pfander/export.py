"""``answers.jsonl`` -> ``samples.csv`` and ``tier1_submission.csv``.

``samples.csv`` is the full record: one row per respondent, every answer exactly
as sampled, plus the derived subscales and outcomes.  ``tier1_submission.csv`` is
the subset the benchmark's Tier-1 schema asks for, in its column order, with the
codebook's exact moderator level strings.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from ..survey.render import slot_manifest
from . import outcomes, templates
from .conditions import CONDITIONS

META_COLUMNS = (
    "profile_id",
    "condition",
    "code_name",
    "control_text",
    "consensus_order",
    "post_order",
    "n_asked",
)

DERIVED_COLUMNS = ("age", "age_band") + tuple(outcomes.SUBSCALES) + outcomes.OUTCOMES


def item_columns() -> list[str]:
    """Every slot id, in the order it first appears across the 17 conditions."""
    order: list[str] = []
    for condition in CONDITIONS:
        for entry in slot_manifest(templates.template_elements(condition)):
            if entry["id"] not in order:
                order.append(entry["id"])
    return order


def read_answers(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build_row(record: dict, items: list[str]) -> dict:
    answers = record["answers"]
    row = {key: record.get(key, "") for key in META_COLUMNS}
    row.update({item: answers.get(item, "") for item in items})
    computed = outcomes.compute(answers)
    row["age"] = outcomes.SURVEY_YEAR - int(answers["year_birth"])
    row["age_band"] = computed["age_band"]
    for name in outcomes.SUBSCALES:
        row[name] = computed[name]
    for name in outcomes.OUTCOMES:
        row[name] = computed[name]
    row["_tier1"] = {
        column: computed.get(column, record.get(column, ""))
        for column in outcomes.TIER1_COLUMNS
    }
    row["_tier1"]["profile_id"] = record["profile_id"]
    row["_tier1"]["condition"] = record["condition"]
    return row


def build_csvs(out_dir: Path) -> dict:
    """Write both CSVs; return a small summary."""
    records = read_answers(out_dir / "answers.jsonl")
    items = item_columns()
    columns = list(META_COLUMNS) + items + list(DERIVED_COLUMNS)

    rows = [build_row(record, items) for record in records]
    rows.sort(key=lambda row: row["profile_id"])

    with (out_dir / "samples.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    with (out_dir / "tier1_submission.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(outcomes.TIER1_COLUMNS))
        writer.writeheader()
        writer.writerows(row["_tier1"] for row in rows)

    conditions = {}
    for row in rows:
        conditions[row["condition"]] = conditions.get(row["condition"], 0) + 1
    return {
        "rows": len(rows),
        "columns": len(columns),
        "conditions": len(conditions),
        "per_condition": dict(sorted(conditions.items())),
        "samples_csv": str(out_dir / "samples.csv"),
        "tier1_csv": str(out_dir / "tier1_submission.csv"),
    }
