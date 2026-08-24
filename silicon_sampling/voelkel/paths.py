"""Where the Voelkel study's inputs and outputs live."""

from __future__ import annotations

from pathlib import Path

from ..models import DEFAULT_RUN
from .. import paths as _paths

ROOT = Path(__file__).resolve().parents[2]
DATA = _paths.resolve("Voelkel")
DOCS = ROOT / "docs"

QSF = DATA / "Materials" / "SDC - Questionnaire - Qualtrics.qsf"
QUESTIONNAIRE_PDF = DATA / "Materials" / "SDC - Questionnaire.pdf"
RECODED_CSV = DATA / "Data" / "SDC - Data - Recoded.csv"
ANONYMIZED_CSV = DATA / "Data" / "SDC - Data - Anonymized.csv"
INTERVENTION_NAMES = DATA / "Data" / "SDC - Data - Intervention Names.csv"
OUTCOME_NAMES = DATA / "Data" / "SDC - Data - Outcome Names.csv"

TEMPLATES = DATA / "text_templates"
MODALITY_AUDIT = TEMPLATES / "modality_audit.csv"

#: Root of every model's silicon-sampling output.
RUNS = _paths.output("Voelkel", "silicon_sampling")


def samples_dir(run: str = DEFAULT_RUN) -> Path:
    """Where one model's run is filed."""
    return RUNS / run


#: The default run.  Anything that has to work for both models takes a directory.
SAMPLES = samples_dir()
SAMPLES_CSV = SAMPLES / "samples.csv"
PROFILES_CSV = SAMPLES / "profiles.csv"

REPORT = DOCS / "reports" / "voelkel_validation"
PLOTS = REPORT / "plots"
