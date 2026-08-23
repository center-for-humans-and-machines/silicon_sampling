"""``make check``, in Python: the submission format gate the container cannot run.

The benchmark ships its own validator — ``scripts/check.R`` calling
``scripts/lib/check_lib.R`` — and it is the only statement of what "accepted"
means.  R is not installable in this container (``r-base-core`` is not in the
apt index), so without a port we would be depositing a file whose format has
never been tested.  This module is a line-by-line port of ``check_repo`` and
``check_submission`` for the checks a Tier-1 entry hits.

Fidelity, not improvement, is the whole point.  Two consequences:

* **The FAIL/WARN split is copied exactly.**  Promoting a WARN to a FAIL would
  make us throw away a submission the organizers would have accepted; demoting
  a FAIL would make us deposit one they reject.  Every status here matches the
  ``ok()`` (FAIL) / ``warn()`` (WARN) helper the R used at that line, including
  the ones that look backwards: a duplicated ``profile_id``, an out-of-range
  value and a below-floor sample size are WARNs, while a single unknown
  moderator level is a FAIL.
* **Check names and detail strings are reproduced verbatim**, so a report from
  this port can be compared against a report from the real thing line by line
  once anyone has R.

Tiers 2 and 3 are deliberately *not* ported: a Tier-1 entry is already scored on
every analysis, so we will never file one, and an untested port of checks we do
not use would be a liability rather than insurance.  Handing this a Tier-2 or
Tier-3 bundle raises instead of quietly reporting a PASS.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np
import pandas as pd

from . import spec

PASS = "PASS"
WARN = "WARN"
FAIL = "FAIL"

_MARK = {PASS: "[ok]  ", WARN: "[warn]", FAIL: "[FAIL]"}

_ZENODO_UPLOAD_TYPES = (
    "publication",
    "poster",
    "presentation",
    "dataset",
    "image",
    "video",
    "software",
    "lesson",
    "physicalobject",
    "other",
)
_ZENODO_ACCESS_RIGHTS = ("open", "embargoed", "restricted", "closed")
_ZENODO_RELATIONS = (
    "isCitedBy",
    "cites",
    "isSupplementTo",
    "isSupplementedBy",
    "isContinuedBy",
    "continues",
    "isDescribedBy",
    "describes",
    "hasMetadata",
    "isMetadataFor",
    "isNewVersionOf",
    "isPreviousVersionOf",
    "isPartOf",
    "hasPart",
    "isReferencedBy",
    "references",
    "isDocumentedBy",
    "documents",
    "isCompiledBy",
    "compiles",
    "isVariantFormOf",
    "isOriginalFormOf",
    "isIdenticalTo",
    "isAlternateIdentifier",
    "isReviewedBy",
    "reviews",
    "isDerivedFrom",
    "isSourceOf",
    "requires",
    "isRequiredBy",
    "isObsoletedBy",
    "obsoletes",
)

_LABEL_RE = re.compile(r"^- \*\*.*\*\* [—-] .*:\s*$")


@dataclass(frozen=True)
class CheckRow:
    """One line of the report: what was checked, the verdict, and why."""

    check: str
    status: str
    detail: str = ""


@dataclass
class CheckResult:
    """Every check that ran, plus the overall verdict."""

    rows: list[CheckRow] = field(default_factory=list)

    @property
    def verdict(self) -> str:
        if any(row.status == FAIL for row in self.rows):
            return FAIL
        if any(row.status == WARN for row in self.rows):
            return "PASS WITH WARNINGS"
        return PASS

    @property
    def passed(self) -> bool:
        """True unless something FAILed — warnings do not sink a submission."""
        return not self.failures

    @property
    def failures(self) -> list[CheckRow]:
        return [row for row in self.rows if row.status == FAIL]

    @property
    def warnings(self) -> list[CheckRow]:
        return [row for row in self.rows if row.status == WARN]

    def counts(self) -> tuple[int, int, int]:
        statuses = [row.status for row in self.rows]
        return (statuses.count(PASS), statuses.count(WARN), statuses.count(FAIL))

    def report_lines(self) -> list[str]:
        """The report ``check_lib.R`` prints, character for character."""
        n_pass, n_warn, n_fail = self.counts()
        body = [
            _MARK[row.status]
            + f" {row.check}"
            + (f"  — {row.detail}" if row.detail else "")
            for row in self.rows
        ]
        return [
            "Silicon Sample Benchmark — submission self-check",
            "-" * 52,
            *body,
            "-" * 52,
            f"OVERALL: {self.verdict}   ({n_pass} pass, {n_warn} warn, {n_fail} fail)",
        ]


def _is_true(condition: Any) -> bool:
    """R's ``isTRUE``: only a real TRUE passes, so an NA never reads as a pass.

    numpy's boolean counts as a real TRUE — it is what a pandas comparison
    returns — but an integer, a string or ``None`` does not.
    """
    return condition is True or (isinstance(condition, np.bool_) and bool(condition))


class _Recorder:
    """The R's ``add`` / ``ok`` / ``warn`` closures, kept in the same shape."""

    def __init__(self) -> None:
        self.rows: list[CheckRow] = []

    def add(self, check: str, status: str, detail: str = "") -> None:
        self.rows.append(CheckRow(check, status, detail))

    def ok(self, condition: Any, check: str, bad: str, good: str = "") -> None:
        (
            self.add(check, PASS, good)
            if _is_true(condition)
            else self.add(check, FAIL, bad)
        )

    def warn(self, condition: Any, check: str, bad: str, good: str = "") -> None:
        (
            self.add(check, PASS, good)
            if _is_true(condition)
            else self.add(check, WARN, bad)
        )


def sha256_file(path: Path) -> str:
    """SHA-256 of a file, as ``digest(file = ..., algo = "sha256")`` computes it."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float):
        return math.isnan(value)
    return value is pd.NA


def _nzchar(value: Any) -> bool:
    """R's ``is.character(x) && nzchar(x)``."""
    return isinstance(value, str) and value != ""


def _as_int(value: Any) -> int | None:
    """R's ``suppressWarnings(as.integer(x))``: an int, or None for NA."""
    try:
        if isinstance(value, bool) or value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _or_else(value: Any, fallback: Any) -> Any:
    """R's ``%||%``: the value unless it is absent — 0 and "" are values."""
    return fallback if value is None else value


def _paste(*parts: Any) -> str:
    """R's ``paste`` over arguments, where NULL contributes nothing."""
    return " ".join(str(part) for part in parts if part is not None and part != "")


def _as_character(value: Any) -> str:
    """R's ``as.character`` on one readr-parsed value."""
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if isinstance(value, float):
        if math.isnan(value):
            return "NA"
        if value == int(value) and abs(value) < 1e15:
            return str(int(value))
        return repr(value)
    if _is_missing(value):
        return "NA"
    return str(value)


def _unique_character(series: pd.Series, drop_na: bool = False) -> list[str]:
    """``unique(as.character(col))``, optionally ``na.omit``-ed, order preserved."""
    seen: list[str] = []
    for value in series.tolist():
        if drop_na and _is_missing(value):
            continue
        text = _as_character(value)
        if text not in seen:
            seen.append(text)
    return seen


def _out_of_range(series: pd.Series, low: float, high: float) -> int:
    """R's ``rng``: how many non-NA values fall outside ``[low, high]``."""
    numeric = pd.to_numeric(series, errors="coerce")
    return int(((numeric < low) | (numeric > high)).sum())


def _table(series: pd.Series) -> dict[str, int]:
    """R's ``table``: counts per distinct character value, NAs dropped.

    Keys come out sorted, as R's factor levels are; Python sorts by code point
    where R sorts by locale collation, which can reorder the punctuated
    condition titles inside a report's detail text but never a status.
    """
    labels = [
        _as_character(value) for value in series.tolist() if not _is_missing(value)
    ]
    counts: dict[str, int] = {}
    for label in labels:
        counts[label] = counts.get(label, 0) + 1
    return {key: counts[key] for key in sorted(counts)}


# ---------------- Tier-1 structural checks ----------------


def check_tier1(frame: pd.DataFrame, name: str, rec: _Recorder) -> None:
    """``.check_t1``: the Tier-1 respondent-level file's structure and values."""
    columns = list(frame.columns)
    missing = [column for column in spec.TIER1_COLUMNS if column not in columns]
    rec.ok(
        len(missing) == 0,
        f"Tier-1 required columns: {name}",
        _paste("missing:", ", ".join(missing)),
    )

    if "condition" in columns:
        present_raw = _unique_character(frame["condition"])
        bad = [value for value in present_raw if value not in spec.CONDITIONS]
        rec.ok(
            len(bad) == 0,
            f"condition labels valid: {name}",
            _paste("unknown:", ", ".join(bad)),
        )
        present = [value for value in spec.CONDITIONS if value in present_raw]
        rec.warn(
            len(present) == len(spec.CONDITIONS),
            f"all {len(spec.CONDITIONS)} conditions present: {name}",
            f"{len(present)} of {len(spec.CONDITIONS)} present",
        )

    n_rows = len(frame)
    for moderator, levels in spec.MODERATORS.items():
        if moderator not in columns:
            continue
        observed = _unique_character(frame[moderator], drop_na=True)
        bad = [value for value in observed if value not in levels]
        rec.ok(
            len(bad) == 0,
            f"{moderator} levels valid: {name}",
            "unknown: "
            + ", ".join(bad)
            + " — must exactly match the spec strings (see codebook.csv)",
        )
        n_na = int(frame[moderator].isna().sum())
        if n_na == n_rows:
            rec.add(
                f"{moderator} has data: {name}",
                FAIL,
                "entirely NA — this moderator would be missing from all subgroup analyses",
            )
        elif n_na > spec.MODERATOR_NA_WARN_SHARE * n_rows:
            rec.add(
                f"{moderator} mostly present: {name}",
                WARN,
                f"{n_na} of {n_rows} rows NA — these drop out of "
                f"{moderator} subgroup analyses",
            )

    if "condition" in columns:
        counts = _table(frame["condition"])
        smallest = min(counts, key=lambda key: counts[key])
        largest = max(counts, key=lambda key: counts[key])
        detail = (
            f"{counts[smallest]} in every condition"
            if counts[smallest] == counts[largest]
            else f"min {counts[smallest]} ({smallest}), max {counts[largest]} ({largest})"
        )
        rec.add(f"per-condition N: {name}", PASS, detail)

        below = [
            label
            for label, count in counts.items()
            if label != spec.CONTROL and count < spec.MIN_N_INTERVENTION
        ]
        if spec.CONTROL in counts and counts[spec.CONTROL] < spec.MIN_N_CONTROL:
            below.append(spec.CONTROL)
        rec.warn(
            len(below) == 0,
            f"precision floor ({spec.MIN_N_INTERVENTION}/intervention, "
            f"{spec.MIN_N_CONTROL:,} control): {name}",
            f"{len(below)} condition(s) below the preregistered minimum: "
            + ", ".join(below[:5])
            + (", ..." if len(below) > 5 else ""),
        )

    if "profile_id" in columns:
        duplicated = int(frame["profile_id"].duplicated().sum())
        rec.warn(
            duplicated == 0,
            f"profile_id unique: {name}",
            f"{duplicated} duplicate(s)",
        )

    if all(
        column in columns for column in ("trust_multidimensional", *spec.TRUST_ITEMS)
    ):
        subscales = pd.DataFrame(
            {
                scale: frame[[f"trust_{scale}_{index}" for index in (1, 2, 3)]]
                .apply(pd.to_numeric, errors="coerce")
                .mean(axis=1, skipna=True)
                for scale in spec.TRUST_SUBSCALES
            }
        )
        expected = subscales.mean(axis=1, skipna=True)
        composite = pd.to_numeric(frame["trust_multidimensional"], errors="coerce")
        n_bad = int(((composite - expected).abs() > spec.COMPOSITE_TOLERANCE).sum())
        rec.warn(
            n_bad == 0,
            f"trust_multidimensional consistent with items: {name}",
            f"{n_bad} row(s) deviate from the codebook definition by > 0.5 — "
            "scoring uses the composite as submitted",
        )

    for outcome in (*spec.SCALE_0_100, *spec.TRUST_ITEMS):
        if outcome not in columns:
            continue
        n_bad = _out_of_range(frame[outcome], 0, 100)
        if n_bad > 0:
            rec.add(
                f"{outcome} in [0,100]: {name}", WARN, f"{n_bad} value(s) out of range"
            )

    if "donation_ams" in columns:
        low, high = spec.DONATION_RANGE
        rec.warn(
            _out_of_range(frame["donation_ams"], low, high) == 0,
            f"donation_ams in [0,10]: {name}",
            "value(s) out of range",
        )

    if "newsletter_signup" in columns:
        allowed = ("0", "1", "TRUE", "FALSE")
        values = _unique_character(frame["newsletter_signup"], drop_na=True)
        rec.warn(
            all(value in allowed for value in values),
            f"newsletter_signup binary: {name}",
            _paste(
                "unexpected:",
                ", ".join(value for value in values if value not in allowed),
            ),
        )


# ---------------- bundle: metadata.json + its prediction files ----------------


def _read_json(path: Path) -> Any | None:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _prediction_files(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, dict):
        return [value]
    if isinstance(value, list):
        return [entry for entry in value if isinstance(entry, dict)]
    return []


def check_bundle(metadata_path: Path, directory: Path) -> list[CheckRow]:
    """``.check_bundle``: metadata.json's schema, then each prediction file."""
    rec = _Recorder()
    metadata_path = Path(metadata_path)
    directory = Path(directory)

    if not metadata_path.exists():
        rec.add("metadata.json present", FAIL, str(metadata_path))
        return rec.rows
    meta = _read_json(metadata_path)
    if not isinstance(meta, dict):
        rec.add("metadata.json parses", FAIL, "invalid JSON")
        return rec.rows
    rec.add("metadata.json parses", PASS)

    rec.ok(_nzchar(meta.get("team_id")), "team_id present", "missing/empty")
    rec.ok(_nzchar(meta.get("team_name")), "team_name present", "missing/empty")
    rec.ok(_nzchar(meta.get("contact")), "contact present", "missing/empty")
    tier = _as_int(meta.get("tier"))
    rec.ok(tier in (1, 2, 3), "tier in {1,2,3}", _paste("got", meta.get("tier")))
    entry = meta.get("entry")
    rec.ok(
        isinstance(entry, str) and spec.ENTRY_PATTERN.match(entry) is not None,
        "entry is primary|secondary-k",
        _paste("got", entry),
    )
    models = meta.get("models") or []
    rec.ok(len(models) >= 1, "models listed", "none listed")
    rec.ok(_nzchar(meta.get("approach_family")), "approach_family present", "missing")
    disclosure = meta.get("disclosure_class")
    rec.ok(
        disclosure in spec.DISCLOSURE_CLASSES,
        "disclosure_class in {A,B,C}",
        _paste("got", disclosure),
    )
    if disclosure == "A":
        rec.warn(
            _is_missing(meta.get("escrow_doi")),
            "escrow_doi null for Class A",
            "set but class A",
        )
    if disclosure in ("B", "C"):
        rec.warn(
            not _is_missing(meta.get("escrow_doi")),
            "escrow_doi set for Class B/C",
            "missing escrow_doi",
        )
    rec.ok(
        meta.get("blinding_attestation") is True,
        "blinding_attestation == true",
        "must be true",
    )
    coverage = meta.get("coverage") or {}
    if not isinstance(coverage, dict):
        coverage = {}
    rec.ok(
        coverage.get("interventions") is not None
        and coverage.get("outcomes") is not None,
        "coverage declared",
        "coverage.interventions/outcomes missing",
    )
    rec.ok(
        _as_int(coverage.get("interventions")) == spec.N_INTERVENTIONS
        and _as_int(coverage.get("outcomes")) == spec.N_OUTCOMES,
        f"coverage is full ({spec.N_INTERVENTIONS} interventions, {spec.N_OUTCOMES} outcomes)",
        "declared {} intervention(s), {} outcome(s) — partial coverage is not "
        "accepted; every intervention and outcome must be predicted".format(
            _or_else(coverage.get("interventions"), "?"),
            _or_else(coverage.get("outcomes"), "?"),
        ),
    )

    files = _prediction_files(meta.get("prediction_files"))
    if not files:
        rec.add("prediction_files listed", FAIL, "none")
        return rec.rows
    rec.add("prediction_files listed", PASS, f"{len(files)} file(s)")

    team = meta.get("team_id") or ""
    if (
        team
        and team != "example"
        and all(
            Path(str(item.get("file", ""))).name.startswith("example_")
            for item in files
        )
    ):
        rec.add(
            f"submission staged for team '{team}'",
            WARN,
            "prediction_files still reference the example — generate your predictions, "
            "then run `make manifest`, before depositing",
        )
        return rec.rows

    n_expected = 2 if tier == 2 else 1
    rec.warn(
        len(files) == n_expected,
        f"file count for Tier {tier if tier is not None else 'NA'}",
        f"expected {n_expected} for one entry, got {len(files)} — a repo holds one "
        "entry; put extra entries in their own repo",
    )

    for item in files:
        relative = str(item.get("file", ""))
        recorded_sha = item.get("sha256")
        path = directory / relative
        pattern = spec.filename_pattern(str(team), tier if tier in (1, 2, 3) else 1)
        rec.ok(
            pattern.match(Path(relative).name) is not None,
            f"filename ok: {Path(relative).name}",
            "does not match <team_id>_T<tier>_<entry>_v<n>[...].csv",
        )

        if not path.exists():
            rec.add(f"file present: {relative}", FAIL, "not found")
            continue
        rec.add(f"file present: {relative}", PASS)
        actual = sha256_file(path)
        rec.ok(
            actual.lower() == str(recorded_sha or "").lower(),
            f"sha256 matches: {relative}",
            f"metadata={str(recorded_sha or '')[:10]}… actual={actual[:10]}…",
        )

        try:
            frame = pd.read_csv(path)
        except Exception:  # noqa: BLE001 - any parse failure is one FAIL row
            rec.add(f"readable CSV: {relative}", FAIL, "could not parse")
            continue

        if tier == 1:
            check_tier1(frame, relative, rec)
        elif tier in (2, 3):
            raise NotImplementedError(
                f"only Tier-1 structural checks are ported; metadata declares tier {tier}. "
                "Run the benchmark's own scripts/check.R for Tier 2/3."
            )
        # A tier outside {1,2,3} already FAILed above, so there is nothing a
        # structural check could add to the verdict.
    return rec.rows


# ---------------- repo: the whole submission directory ----------------


def _count_blank_registration_items(lines: Sequence[str]) -> int:
    """The R's blank-item count: a label line whose answer never arrives."""
    following = [line.strip() for line in lines[1:]] + [""]
    blank = 0
    for line, nxt in zip(lines, following):
        if _LABEL_RE.match(line) and (
            nxt == "" or nxt.startswith("- **") or nxt.startswith("#")
        ):
            blank += 1
    return blank


def _check_zenodo(root: Path, rec: _Recorder) -> None:
    """The ``.zenodo.json`` block: every row of it is a WARN in the R, too."""
    path = root / ".zenodo.json"
    if not path.exists():
        rec.warn(
            False,
            ".zenodo.json present",
            "missing — run `make zenodo_citation` to generate Zenodo deposit metadata",
        )
        return
    zenodo = _read_json(path)
    rec.ok(zenodo is not None, ".zenodo.json parses", "invalid JSON in .zenodo.json")
    if zenodo is None:
        return
    creators = zenodo.get("creators") or []
    names = [
        str(creator.get("name", ""))
        for creator in creators
        if isinstance(creator, dict)
    ]
    rec.warn(
        _nzchar(zenodo.get("title"))
        and _nzchar(zenodo.get("description"))
        and len(names) >= 1
        and all(name != "" for name in names),
        ".zenodo.json has title/description/creator",
        "fill title, description, and at least one creator name",
    )
    rec.warn(
        zenodo.get("upload_type") in _ZENODO_UPLOAD_TYPES,
        ".zenodo.json upload_type valid",
        f"upload_type '{zenodo.get('upload_type') or ''}' not a Zenodo upload type",
    )
    rec.warn(
        zenodo.get("access_right") is None
        or zenodo.get("access_right") in _ZENODO_ACCESS_RIGHTS,
        ".zenodo.json access_right valid",
        f"access_right '{zenodo.get('access_right') or ''}' invalid",
    )
    rec.warn(
        _nzchar(zenodo.get("license")),
        ".zenodo.json license set",
        "license must be a non-empty id string (e.g. CC-BY-4.0)",
    )
    keywords = zenodo.get("keywords")
    rec.warn(
        keywords is None
        or (
            isinstance(keywords, list)
            and all(isinstance(word, str) and word != "" for word in keywords)
        ),
        ".zenodo.json keywords valid",
        "keywords must be a list of strings",
    )
    rec.warn(
        not any("Lastname, Firstname" in name for name in names),
        ".zenodo.json creators filled",
        "creator still 'Lastname, Firstname' — fill `creators` in metadata.json, "
        "then re-run make zenodo_citation",
    )
    orcids = [
        str(creator.get("orcid"))
        for creator in creators
        if isinstance(creator, dict) and _nzchar(creator.get("orcid"))
    ]
    invalid = [orcid for orcid in orcids if not orcid_is_valid(orcid)]
    rec.warn(
        len(invalid) == 0,
        ".zenodo.json ORCIDs valid (format + checksum)",
        "invalid ORCID (format or MOD-11-2 checksum): "
        + ", ".join(invalid)
        + " — Zenodo would reject the deposit (HTTP 500); fix or remove it",
    )
    related = zenodo.get("related_identifiers")
    rec.warn(
        related is None
        or all(
            _nzchar(item.get("identifier"))
            and item.get("relation") in _ZENODO_RELATIONS
            for item in related
            if isinstance(item, dict)
        ),
        ".zenodo.json related_identifiers valid",
        "a related_identifier is missing an identifier or uses an unknown relation",
    )


def orcid_is_valid(value: str) -> bool:
    """ISO 7064 MOD-11-2, the ORCID checksum Zenodo enforces before depositing."""
    digits = value.replace("-", "")
    if not re.match(r"^[0-9]{15}[0-9X]$", digits):
        return False
    total = 0
    for character in digits[:15]:
        total = (total + int(character)) * 2
    result = (12 - total % 11) % 11
    return ("X" if result == 10 else str(result)) == digits[15]


def check_repo(root: Path) -> CheckResult:
    """``check_repo``: the submission directory, then the bundle inside it."""
    root = Path(root)
    rec = _Recorder()

    def has(relative: str) -> bool:
        return (root / relative).exists()

    rec.ok(has("metadata.json"), "metadata.json present", "missing at repo root")
    rec.ok(has("registration.md"), "registration.md present", "missing at repo root")
    rec.ok(has("codebook.csv"), "codebook.csv present", "missing at repo root")
    rec.ok((root / "survey").is_dir(), "survey/ present", "missing")
    rec.warn(
        has("survey/survey.qsf"),
        "survey/survey.qsf present",
        "not present yet (provided on invitation)",
    )

    predictions = sorted(_csv_names(root / "predictions"))
    rec.ok(
        len(predictions) >= 1,
        "predictions/ has a CSV",
        "no prediction file in predictions/",
    )

    if has("registration.md"):
        lines = (root / "registration.md").read_text(encoding="utf-8").splitlines()
        blank = _count_blank_registration_items(lines)
        rec.warn(
            blank == 0,
            "registration.md filled in",
            f"{blank} checklist item(s) still blank",
        )

    is_example = True
    if has("metadata.json"):
        meta = _read_json(root / "metadata.json")
        if isinstance(meta, dict):
            is_example = meta.get("team_id") == "example"
            rec.warn(
                not is_example,
                "team_id set (not the example)",
                "still 'example' — edit metadata.json before submitting",
            )
            repository = meta.get("code_repository")
            rec.warn(
                _nzchar(repository) and "your-team/your-repo" not in str(repository),
                "code_repository set",
                "link your generation code in metadata.json (code_repository / code_doi)",
            )

    _check_zenodo(root, rec)

    leftover = [name for name in predictions if name.startswith("example_")]
    if not is_example and leftover:
        rec.warn(
            False,
            "example files removed from predictions/",
            "delete before depositing: " + ", ".join(leftover),
        )
    if not is_example and has("raw_data_deposit/example_raw_export.csv"):
        rec.warn(
            False,
            "example raw export removed from raw_data_deposit/",
            "delete raw_data_deposit/example_raw_export.csv before depositing",
        )

    rows = rec.rows + check_bundle(root / "metadata.json", root)
    return CheckResult(rows)


def check_submission(
    metadata: str = "metadata.json", directory: Path = Path(".")
) -> CheckResult:
    """``check_submission``: one deposited bundle, without the repo-level checks."""
    directory = Path(directory)
    return CheckResult(check_bundle(directory / metadata, directory))


def _csv_names(directory: Path) -> Iterable[str]:
    if not directory.is_dir():
        return []
    return [path.name for path in directory.iterdir() if path.suffix == ".csv"]


def write_report(result: CheckResult, path: Path) -> Path:
    """Write the printed report, as ``make check`` leaves it beside metadata.json.

    The R always writes ``metadata_check_report.txt``; here it is explicit,
    because this project's data tree must never be overwritten by accident.
    """
    path = Path(path)
    path.write_text("\n".join(result.report_lines()) + "\n", encoding="utf-8")
    return path


def print_report(result: CheckResult, printer: Callable[[str], None] = print) -> None:
    for line in result.report_lines():
        printer(line)
