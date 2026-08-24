"""The benchmark's Tier-1 submission schema, as data.

The published schema lives in ``scripts/lib/submission_spec.R`` of the
submission-template repository, and the gate that enforces it is ~750 lines of
R.  This container has no R and cannot get one, so the schema has to exist a
second time, in Python, or nothing we produce can be validated before the
deposit lock.  This module is that second copy.

Two copies of a schema drift.  The defence is that everything derivable from a
*shipped data file* is derived from it and compared against the literal below:
``codebook.csv`` carries the exact moderator level strings and the response
options that pin each outcome's range, and ``survey/condition_codenames.csv``
carries the 17 condition titles.  :func:`verify_against_codebook` walks those
files and returns every disagreement; the test suite asserts the list is empty,
and the CLI prints it before it prints a verdict.  What cannot be derived — the
Tier-1 *column order*, the reference level of each moderator, the precision
floor — is written out once here and nowhere else.

The reference levels are not in ``submission_spec.R`` as a separate field: the
first level of each moderator *is* the reference, because the analysis code
dummy-codes factors in the order the spec lists them.  That ordering is load
bearing for every interaction term and every stereotyping coefficient, so it is
recorded explicitly rather than left implicit in a tuple's order.
"""

from __future__ import annotations

import csv
import os
import re
from pathlib import Path

from .. import paths as _paths

#: The 16 text interventions, in the benchmark's own order (``submission_spec.R``).
INTERVENTIONS: tuple[str, ...] = (
    "Corporate reliance",
    "Social justice",
    "Interview Prof. Maraun",
    "Funding",
    "Oil industry misinformation",
    "Measurement & modeling (1)",
    "Former skeptics",
    "High public trust",
    "Measurement & modeling (2)",
    "Peer-review",
    "Scientist community helpers",
    "Consensus",
    "Portrait Prof. Cherry",
    "Model accuracy",
    "Interview Prof. Sebille",
    "Extreme weather predictions",
)

CONTROL = "control"

#: The 17 condition labels a Tier-1 file may use.
CONDITIONS: tuple[str, ...] = (CONTROL,) + INTERVENTIONS

#: The 12 trust items, sub-components of the primary outcome.
TRUST_ITEMS: tuple[str, ...] = tuple(
    f"trust_{scale}_{index}"
    for scale in ("competence", "integrity", "benevolence", "openness")
    for index in (1, 2, 3)
)

TRUST_SUBSCALES: tuple[str, ...] = (
    "competence",
    "integrity",
    "benevolence",
    "openness",
)

#: The 13 scored outcomes, in the benchmark's order.
OUTCOMES: tuple[str, ...] = (
    "trust_multidimensional",
    "trust_post",
    "distrust_post",
    "funding_perceptions",
    "policy_role_mean",
    "inst_trust_mean",
    "belief_post",
    "concern_mean",
    "policy_general",
    "policy_specific_mean",
    "behavior_mean",
    "donation_ams",
    "newsletter_signup",
)

#: The 11 outcomes on the 0-100 slider scale (``sst$scale_0_100``).
SCALE_0_100: tuple[str, ...] = tuple(
    name for name in OUTCOMES if name not in ("donation_ams", "newsletter_signup")
)

DONATION_RANGE: tuple[float, float] = (0.0, 10.0)

#: Value range per outcome, on the outcome's native scale.
OUTCOME_RANGE: dict[str, tuple[float, float]] = {
    **{name: (0.0, 100.0) for name in SCALE_0_100},
    "donation_ams": DONATION_RANGE,
    "newsletter_signup": (0.0, 1.0),
}

#: The six moderators with their exact level strings.  **The first level of each
#: is the dummy-coding reference** — see :data:`REFERENCE_LEVELS`.
MODERATORS: dict[str, tuple[str, ...]] = {
    "gender": ("Male", "Female", "Other"),
    "age_band": ("18-29", "30-44", "45-59", "60+"),
    "race": (
        "White / Caucasian",
        "Black / African American",
        "Hispanic / Latino",
        "Asian / Asian American",
        "Other",
    ),
    "education": (
        "Less than high school",
        "High school diploma / GED",
        "Some college or Associate's degree",
        "Bachelor's degree",
        "Master's degree / Professional degree",
        "Doctorate degree / Ph.D.",
    ),
    "income": (
        "Less than $30,000",
        "$30,000 to $55,999",
        "$56,000 to $99,999",
        "$100,000 to $167,999",
        "$168,000 or more",
    ),
    "party": ("Republican", "Democrat", "Independent", "Other"),
}

#: Dummy-coding reference level per moderator: the spec's first level.
REFERENCE_LEVELS: dict[str, str] = {
    name: levels[0] for name, levels in MODERATORS.items()
}

#: Tier-1 required columns, in the order the example submission uses (33).
TIER1_COLUMNS: tuple[str, ...] = (
    ("profile_id", "condition")
    + tuple(MODERATORS)
    + ("trust_multidimensional",)
    + TRUST_ITEMS
    + (
        "trust_post",
        "distrust_post",
        "funding_perceptions",
        "policy_role_mean",
        "inst_trust_mean",
        "belief_post",
        "concern_mean",
        "policy_general",
        "policy_specific_mean",
        "behavior_mean",
        "donation_ams",
        "newsletter_signup",
    )
)

TIER2_MAIN_COLUMNS: tuple[str, ...] = ("condition", "outcome", "mean")
TIER2_MOD_COLUMNS: tuple[str, ...] = (
    "condition",
    "moderator",
    "moderator_level",
    "outcome",
    "mean",
)
TIER3_COLUMNS: tuple[str, ...] = ("condition", "outcome", "ate")

#: What ``coverage`` in metadata.json must declare, exactly.
N_INTERVENTIONS = len(INTERVENTIONS)
N_OUTCOMES = len(OUTCOMES)

#: Preregistered precision floor for a Tier-1 entry (a WARN below it, not a FAIL).
MIN_N_INTERVENTION = 500
MIN_N_CONTROL = 1000

#: ``check_lib.R`` warns when a moderator is NA in more than this share of rows.
MODERATOR_NA_WARN_SHARE = 0.1

#: Tolerance on ``trust_multidimensional`` vs its items.  The R message says
#: "> 0.5" but the code compares against 0.51; the code is what runs.
COMPOSITE_TOLERANCE = 0.51

ENTRY_PATTERN = re.compile(r"^(primary|secondary-\d+)$")

DISCLOSURE_CLASSES = ("A", "B", "C")

#: Where the shipped materials live: the live template checkout if the
#: environment names one, otherwise this project's snapshot of it.
TEMPLATE_REPO_ENV = "SILICON_SAMPLING_SUBMISSION_REPO"
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SNAPSHOT = _paths.resolve("pfander", "submission_template")


def filename_pattern(team_id: str, tier: int = 1) -> re.Pattern[str]:
    """The file-name grammar ``check_lib.R`` enforces for one entry's files.

    Tier 1 and Tier 3: ``^<team_id>_T<tier>_(primary|secondary-\\d+)_v\\d+\\.csv$``.
    (The R pastes ``team_id`` into the pattern unescaped; escaping it here can
    only make a pathological team id match *less* loosely.)
    """
    team = re.escape(team_id)
    if tier == 2:
        return re.compile(
            rf"^{team}_T2_(primary|secondary-\d+)_v\d+_cells_(main|moderator)\.csv$"
        )
    return re.compile(rf"^{team}_T{tier}_(primary|secondary-\d+)_v\d+\.csv$")


def prediction_filename(
    team_id: str, entry: str = "primary", version: int = 1, tier: int = 1
) -> str:
    """The name this entry's prediction file must carry."""
    if not ENTRY_PATTERN.match(entry):
        raise ValueError(f"entry must be primary or secondary-k, got {entry!r}")
    if version < 1:
        raise ValueError(f"version counter starts at 1, got {version}")
    return f"{team_id}_T{tier}_{entry}_v{version}.csv"


def default_template_root() -> Path | None:
    """The submission-template checkout, if one is reachable."""
    named = os.environ.get(TEMPLATE_REPO_ENV, "/opt/silicon-sample-submission")
    path = Path(named)
    return path if path.is_dir() else None


def default_codebook_path() -> Path | None:
    """``codebook.csv``, preferring the live checkout over our snapshot."""
    root = default_template_root()
    for candidate in (
        root / "codebook.csv" if root else None,
        _SNAPSHOT / "codebook.csv",
    ):
        if candidate is not None and candidate.is_file():
            return candidate
    return None


def default_codenames_path() -> Path | None:
    """``survey/condition_codenames.csv``, preferring the live checkout."""
    root = default_template_root()
    candidates = [
        root / "survey" / "condition_codenames.csv" if root else None,
        _SNAPSHOT / "survey" / "condition_codenames.csv",
    ]
    for candidate in candidates:
        if candidate is not None and candidate.is_file():
            return candidate
    return None


def read_codebook(path: Path) -> list[dict[str, str]]:
    """``codebook.csv`` as a list of rows."""
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def derive_moderator_levels(rows: list[dict[str, str]]) -> dict[str, tuple[str, ...]]:
    """The moderator levels the codebook spells out, in the codebook's order.

    Each level row's ``response_options`` reads
    ``Exact submission levels: A | B | C (raw Qualtrics codes ...)``; the
    parenthetical is dropped because it contains pipes of its own.
    """
    derived: dict[str, tuple[str, ...]] = {}
    for row in rows:
        options = row.get("response_options") or ""
        marker = "Exact submission levels:"
        if marker not in options:
            continue
        listed = options.split(marker, 1)[1]
        listed = listed.split(" (", 1)[0]
        levels = tuple(part.strip() for part in listed.split("|") if part.strip())
        if levels:
            derived[row["target_label"]] = levels
    return derived


def derive_condition_titles(path: Path) -> tuple[str, ...]:
    """The condition titles ``condition_codenames.csv`` maps its code names to."""
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        titles = [row["title"] for row in csv.DictReader(handle)]
    seen: list[str] = []
    for title in titles:
        if title not in seen:
            seen.append(title)
    return tuple(seen)


def _range_is_consistent(name: str, options: str) -> bool:
    """Does the codebook's response-option prose agree with our declared range?"""
    low, high = OUTCOME_RANGE[name]
    text = options.replace("–", "-").replace("—", "-")
    if (low, high) == (0.0, 1.0):
        return "0 / 1" in text or "1/0" in text
    if (low, high) == (0.0, 10.0):
        return "$0-$10" in text or "0-10" in text
    return "0-100" in text or ("0 =" in text and "100 =" in text)


def verify_against_codebook(
    codebook: Path | None = None, codenames: Path | None = None
) -> list[str]:
    """Compare this module's literals against the shipped materials.

    Returns one string per disagreement; an empty list means the hardcoded
    schema still matches ``codebook.csv`` and ``condition_codenames.csv``.  A
    missing shipped file is reported as a problem too — silently skipping it
    would turn the whole check into a no-op exactly when it matters.
    """
    problems: list[str] = []

    codebook = codebook or default_codebook_path()
    if codebook is None:
        problems.append("codebook.csv not found — cannot verify the schema")
    else:
        rows = read_codebook(codebook)
        derived = derive_moderator_levels(rows)
        for name, levels in MODERATORS.items():
            got = derived.get(name)
            if got is None:
                problems.append(
                    f"{name}: no 'Exact submission levels' row in the codebook"
                )
            elif got != levels:
                problems.append(
                    f"{name}: codebook says {list(got)}, spec says {list(levels)}"
                )
        extra = set(derived) - set(MODERATORS)
        if extra:
            problems.append(
                f"codebook spells out levels for unknown moderators: {sorted(extra)}"
            )

        labels = {row["target_label"] for row in rows}
        for column in TIER1_COLUMNS:
            if column in ("profile_id", "condition"):
                continue
            if column not in labels:
                problems.append(
                    f"{column}: Tier-1 column is not a codebook target_label"
                )

        options = {
            row["target_label"]: (row.get("response_options") or "") for row in rows
        }
        for name in OUTCOMES:
            text = options.get(name, "")
            if text and not _range_is_consistent(name, text):
                low, high = OUTCOME_RANGE[name]
                problems.append(
                    f"{name}: declared range [{low:g},{high:g}] does not match the "
                    f"codebook's response options ({text[:60]!r})"
                )

    codenames = codenames or default_codenames_path()
    if codenames is None:
        problems.append(
            "condition_codenames.csv not found — cannot verify condition labels"
        )
    else:
        titles = set(derive_condition_titles(codenames))
        missing = set(CONDITIONS) - titles
        unknown = titles - set(CONDITIONS)
        if missing:
            problems.append(
                f"conditions in the spec but not in condition_codenames.csv: {sorted(missing)}"
            )
        if unknown:
            problems.append(
                f"conditions in condition_codenames.csv but not in the spec: {sorted(unknown)}"
            )

    return problems


assert len(TIER1_COLUMNS) == 33, len(TIER1_COLUMNS)
assert len(set(TIER1_COLUMNS)) == 33
assert len(CONDITIONS) == 17
assert len(TRUST_ITEMS) == 12
assert (N_INTERVENTIONS, N_OUTCOMES) == (16, 13)
