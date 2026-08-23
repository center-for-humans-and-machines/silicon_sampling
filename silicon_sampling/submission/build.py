"""Assemble a deposit-ready submission directory from a finished run.

The template's own path from raw output to a valid submission is
``make clean`` + ``make manifest``, both R.  This is the Python equivalent, and
it is deliberately dumber than the R in one respect: it does not repair the
frame it is handed.  Columns are written in the schema's order, anything extra
is appended, and nothing is recoded or recomputed — because the thing that
decides whether a file is acceptable is :mod:`.check`, and a builder that
quietly fixes its input would hide exactly the defects the checker exists to
surface.  Build, then check, then fix the *generator*.

The one transformation worth having is the one that is not a transformation at
all: when the source CSV already carries the 33 Tier-1 columns in the right
order, the file is copied byte for byte instead of round-tripped through pandas.
Re-serialising floats is a needless way to change values (and the SHA-256) of a
file that was already correct.

Nothing here overwrites: this project's data tree is git-ignored, so a
clobbered file is gone for good.  A rebuild has to say ``overwrite=True``, and
even then ``metadata.json`` is *patched* — the fields a human filled in by hand
survive a re-run of the manifest.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field, replace
from pathlib import Path

import pandas as pd

from . import spec
from .check import sha256_file

#: Marker for every field a human still has to fill in before the deposit.  It
#: is ugly on purpose: it has to be impossible to mistake for a real value in a
#: diff, in a JSON dump, or on a Zenodo landing page.
TODO = "TODO-BEFORE-DEPOSIT"


@dataclass(frozen=True)
class SubmissionMeta:
    """The ``metadata.json`` fields a build needs, with obvious placeholders.

    Defaults describe *this* project's entry as far as it is already decided —
    Tier 1, one primary entry, per-respondent simulation, disclosure class A —
    and mark everything identifying a person as unfilled.
    """

    team_id: str = "mpib"
    entry: str = "primary"
    tier: int = 1
    version: int = 1
    team_name: str = f"{TODO} — team name"
    contact: str = f"{TODO}@mpib-berlin.mpg.de"
    creators: tuple[dict[str, str], ...] = field(
        default_factory=lambda: (
            {
                "name": f"{TODO}, Lastname Firstname",
                "affiliation": "Max Planck Institute for Human Development, Berlin",
                "orcid": "",
            },
        )
    )
    abstract: str = f"{TODO} — one-paragraph description of the approach"
    license: str = "CC-BY-4.0"
    approach_family: str = "per-respondent simulation, single model"
    models: tuple[str, ...] = (f"{TODO} — exact model id, provider and version",)
    code_repository: str = (
        "https://github.com/center-for-humans-and-machines/silicon_sampling"
    )
    code_doi: str | None = None
    disclosure_class: str = "A"
    escrow_doi: str | None = None
    zenodo_doi: str | None = None
    blinding_attestation: bool = True

    def __post_init__(self) -> None:
        if spec.ENTRY_PATTERN.match(self.entry) is None:
            raise ValueError(
                f"entry must be primary or secondary-k, got {self.entry!r}"
            )
        if self.tier != 1:
            raise ValueError("only Tier-1 submissions are supported here")

    @property
    def filename(self) -> str:
        """The prediction file's name under the benchmark's grammar."""
        return spec.prediction_filename(
            self.team_id, self.entry, self.version, self.tier
        )

    def to_json(self, prediction_files: list[dict[str, str]]) -> dict:
        """The metadata document, in the template's own key order."""
        return {
            "team_id": self.team_id,
            "team_name": self.team_name,
            "contact": self.contact,
            "creators": [dict(creator) for creator in self.creators],
            "abstract": self.abstract,
            "license": self.license,
            "tier": self.tier,
            "entry": self.entry,
            "approach_family": self.approach_family,
            "models": list(self.models),
            "code_repository": self.code_repository,
            "code_doi": self.code_doi,
            "disclosure_class": self.disclosure_class,
            "escrow_doi": self.escrow_doi,
            "zenodo_doi": self.zenodo_doi,
            "prediction_files": prediction_files,
            "coverage": {
                "interventions": spec.N_INTERVENTIONS,
                "outcomes": spec.N_OUTCOMES,
            },
            "blinding_attestation": self.blinding_attestation,
        }


@dataclass
class BuildResult:
    """What a build put on disk."""

    root: Path
    predictions: Path
    metadata: Path
    sha256: str
    rows: int
    raw_export: Path | None = None
    staged: tuple[Path, ...] = ()
    copied_verbatim: bool = False

    def summary(self) -> str:
        lines = [
            f"submission root  {self.root}",
            f"predictions      {self.predictions.relative_to(self.root)}  "
            f"({self.rows} rows, {'verbatim copy' if self.copied_verbatim else 'rewritten'})",
            f"sha256           {self.sha256}",
            f"metadata         {self.metadata.relative_to(self.root)}",
        ]
        if self.raw_export is not None:
            lines.append(f"raw deposit      {self.raw_export.relative_to(self.root)}")
        if self.staged:
            lines.append(
                "staged from template  "
                + ", ".join(str(path.relative_to(self.root)) for path in self.staged)
            )
        return "\n".join(lines)


def order_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Tier-1 columns first, in the schema's order; anything else after them."""
    known = [column for column in spec.TIER1_COLUMNS if column in frame.columns]
    extra = [column for column in frame.columns if column not in spec.TIER1_COLUMNS]
    return frame[known + extra]


def _guard(path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} exists; pass overwrite=True to replace it")


def write_predictions(
    predictions: pd.DataFrame | Path | str,
    out_dir: Path,
    meta: SubmissionMeta,
    *,
    overwrite: bool = False,
) -> tuple[Path, int, bool]:
    """Write ``predictions/<team_id>_T1_<entry>_v<n>.csv``.

    Returns the path, the row count, and whether the source was copied verbatim.
    """
    target = Path(out_dir) / "predictions" / meta.filename
    target.parent.mkdir(parents=True, exist_ok=True)
    _guard(target, overwrite)

    if isinstance(predictions, (str, Path)):
        source = Path(predictions)
        frame = pd.read_csv(source)
        if tuple(frame.columns) == spec.TIER1_COLUMNS:
            shutil.copyfile(source, target)
            return target, len(frame), True
    else:
        frame = predictions

    ordered = order_columns(frame)
    ordered.to_csv(target, index=False)
    return target, len(ordered), False


def write_metadata(
    out_dir: Path,
    meta: SubmissionMeta,
    prediction_files: list[dict[str, str]],
) -> Path:
    """Write ``metadata.json``, or patch the one already there.

    Patching is what ``make manifest`` does: the fingerprint block and the
    fields this builder owns are refreshed, and every other key a human edited
    is left exactly as it was.
    """
    path = Path(out_dir) / "metadata.json"
    document = meta.to_json(prediction_files)
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(existing, dict):
            merged = dict(existing)
            merged.update(
                {
                    "team_id": document["team_id"],
                    "tier": document["tier"],
                    "entry": document["entry"],
                    "prediction_files": document["prediction_files"],
                    "coverage": document["coverage"],
                }
            )
            document = merged
    # ensure_ascii=False: jsonlite writes UTF-8, and a team name or affiliation
    # with a non-ASCII character has to stay readable in the deposited file.
    path.write_text(
        json.dumps(document, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return path


def stage_raw_export(
    raw_export: Path | str,
    out_dir: Path,
    meta: SubmissionMeta,
    *,
    overwrite: bool = False,
) -> Path:
    """Copy the run's raw export into ``raw_data_deposit/``.

    The deposit keeps the simulation's raw output as part of the transparency
    record, so this is a copy of the file the predictions were derived *from* —
    respondent-level, in the survey's own item names.
    """
    source = Path(raw_export)
    target = (
        Path(out_dir)
        / "raw_data_deposit"
        / f"{Path(meta.filename).stem}_raw_export.csv"
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    _guard(target, overwrite)
    shutil.copyfile(source, target)
    return target


def stage_template(template_root: Path | str, out_dir: Path) -> tuple[Path, ...]:
    """Copy the shipped files a submission repo must contain.

    ``registration.md``, ``codebook.csv`` and ``survey/`` are the template's,
    not ours, and ``check_repo`` FAILs without them.  Existing files are never
    touched, so a registration form that has been filled in survives a rebuild.
    """
    source_root = Path(template_root)
    out_dir = Path(out_dir)
    staged: list[Path] = []
    for name in ("registration.md", "codebook.csv"):
        source = source_root / name
        target = out_dir / name
        if source.is_file() and not target.exists():
            shutil.copyfile(source, target)
            staged.append(target)
    survey = source_root / "survey"
    if survey.is_dir():
        for source in sorted(survey.iterdir()):
            if not source.is_file():
                continue
            target = out_dir / "survey" / source.name
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                shutil.copyfile(source, target)
                staged.append(target)
    return tuple(staged)


def build_submission(
    predictions: pd.DataFrame | Path | str,
    out_dir: Path | str,
    meta: SubmissionMeta | None = None,
    *,
    raw_export: Path | str | None = None,
    template_root: Path | str | None = None,
    models: tuple[str, ...] | None = None,
    overwrite: bool = False,
) -> BuildResult:
    """Build a complete Tier-1 submission directory and return what it wrote."""
    meta = meta or SubmissionMeta()
    if models:
        meta = replace(meta, models=tuple(models))
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "raw_data_deposit").mkdir(exist_ok=True)

    target, rows, verbatim = write_predictions(
        predictions, out_dir, meta, overwrite=overwrite
    )
    digest = sha256_file(target)
    metadata = write_metadata(
        out_dir,
        meta,
        [{"file": f"predictions/{target.name}", "sha256": digest}],
    )
    staged = stage_template(template_root, out_dir) if template_root else ()
    deposited = (
        stage_raw_export(raw_export, out_dir, meta, overwrite=overwrite)
        if raw_export
        else None
    )
    return BuildResult(
        root=out_dir,
        predictions=target,
        metadata=metadata,
        sha256=digest,
        rows=rows,
        raw_export=deposited,
        staged=staged,
        copied_verbatim=verbatim,
    )


def models_from_run_meta(path: Path | str) -> tuple[str, ...]:
    """The model id a run recorded, for ``metadata.json``'s ``models`` list."""
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    model = document.get("model") or document.get("engine", {}).get("model")
    return (str(model),) if model else ()
