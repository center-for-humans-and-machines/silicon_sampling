"""Where the Climate Change Challenge inputs and outputs live.

The study arrived as a replication package under ``data/calibration/SI/Voelkel``
rather than as its own top-level study directory, so the paths point there.  Note
the deliberate asymmetry with :mod:`silicon_sampling.voelkel.paths`, which points
at a *different* study; pointing this module at that one was flagged as a repo
hazard during verification and would silently score the wrong instrument.
"""

from __future__ import annotations

from pathlib import Path

from .. import paths as _paths
from ..models import DEFAULT_RUN

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs"

#: The replication package.
DATA = _paths.resolve("calibration", "SI", "Voelkel")

QSF = DATA / "Materials" / "CCC - Questionnaire - Qualtrics.qsf"
QUESTIONNAIRE_PDF = DATA / "Materials" / "CCC - Questionnaire.pdf"
QUALTRICS_PDF = DATA / "Materials" / "CCC - Questionnaire - Qualtrics.pdf"

#: The analysis file the published scripts read.  ``CCC - Data Attriter -
#: Recoded.csv`` is the attrition variant, and the copy at
#: ``data/calibration/datasets/voelkel_etal2026.csv`` is byte-equivalent to *that*
#: one rather than to this — a distinction worth keeping straight, since the two
#: differ by 49 derived columns.
RECODED_CSV = DATA / "Data" / "CCC - Data - Recoded.csv"
ATTRITER_CSV = DATA / "Data" / "CCC - Data Attriter - Recoded.csv"
DEIDENTIFIED_CSV = DATA / "Data" / "CCC - Data - Deidentified.csv"
CODE = DATA / "Code"

TEMPLATES = _paths.output("CCC", "text_templates")
MODALITY_AUDIT = TEMPLATES / "modality_audit.csv"

#: Root of every model's silicon-sampling output.
RUNS = _paths.output("CCC", "silicon_sampling")


def samples_dir(run: str = DEFAULT_RUN) -> Path:
    """Where one model's run is filed."""
    return RUNS / run


SAMPLES = samples_dir()
SAMPLES_CSV = SAMPLES / "samples.csv"
PROFILES_CSV = SAMPLES / "profiles.csv"

REPORT = DOCS / "reports" / "ccc_validation"
PLOTS = REPORT / "plots"
