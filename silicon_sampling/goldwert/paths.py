"""Where the Goldwert study's inputs and outputs live.

Two things are worth knowing about this layout.  The response data is *not*
under ``data/Goldwert`` — it arrived through the calibration-dataset drop and
stays where the other calibration studies keep theirs, so
:data:`RESPONSES_CSV` points sideways.  And the questionnaire is not one file:
the authors published eighteen per-arm Qualtrics exports, two of which happen to
be exports of the whole master survey with everything but their own arm swept
into a "Trash / Unused Questions" block.  That trash block is the only surviving
copy of the outcome battery, so :data:`MASTER_QSF` is a load-bearing path rather
than a convenience.
"""

from __future__ import annotations

from pathlib import Path

from ..models import DEFAULT_RUN

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "Goldwert"
DOCS = ROOT / "docs"

MATERIALS = DATA / "Materials"
ARM_QSF_DIR = MATERIALS / "intervention_qsfs"
ARM_DOCX_DIR = MATERIALS / "intervention_docx"
CODE_DIR = MATERIALS / "code"
OSF_MANIFEST = MATERIALS / "osf_manifest.json"

#: Whichever per-arm export also carries the full survey in its trash block.
#: Both master exports contain the same battery; this one is the reference.
MASTER_QSF = ARM_QSF_DIR / "Misperception_Correction_Risks.qsf"
#: The other master export, kept for cross-checking the battery against.
MASTER_QSF_ALT = ARM_QSF_DIR / "Linking_Individual_and_Structural_Change.qsf"

CALIBRATION = ROOT / "data" / "calibration"
RESPONSES_CSV = CALIBRATION / "datasets" / "goldwert_etal2026.csv"
CODEBOOK_PDF = CALIBRATION / "codebooks" / "goldwert_etal2026_codebook.pdf"
PAPER_PDF = CALIBRATION / "papers" / "goldwert_et_al.pdf"

TEMPLATES = DATA / "text_templates"
MODALITY_AUDIT = TEMPLATES / "modality_audit.csv"
MANIFEST = TEMPLATES / "manifest.json"
FORMAT_DOC = TEMPLATES / "00_FORMAT.md"

#: Root of every model's silicon-sampling output.
RUNS = DATA / "silicon_sampling"


def samples_dir(run: str = DEFAULT_RUN) -> Path:
    """Where one model's run is filed."""
    return RUNS / run


SAMPLES = samples_dir()
SAMPLES_CSV = SAMPLES / "samples.csv"
PROFILES_CSV = SAMPLES / "profiles.csv"

REPORT = DOCS / "reports" / "goldwert_calibration"
PLOTS = REPORT / "plots"


def arm_qsf(stem: str) -> Path:
    """The Qualtrics export for one arm, by its OSF filename stem."""
    return ARM_QSF_DIR / f"{stem}.qsf"
