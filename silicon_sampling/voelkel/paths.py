"""Where the Voelkel study's inputs and outputs live."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "Voelkel"
DOCS = ROOT / "docs"

QSF = DATA / "Materials" / "SDC - Questionnaire - Qualtrics.qsf"
QUESTIONNAIRE_PDF = DATA / "Materials" / "SDC - Questionnaire.pdf"
RECODED_CSV = DATA / "Data" / "SDC - Data - Recoded.csv"
ANONYMIZED_CSV = DATA / "Data" / "SDC - Data - Anonymized.csv"
INTERVENTION_NAMES = DATA / "Data" / "SDC - Data - Intervention Names.csv"
OUTCOME_NAMES = DATA / "Data" / "SDC - Data - Outcome Names.csv"

TEMPLATES = DATA / "text_templates"
MODALITY_AUDIT = TEMPLATES / "modality_audit.csv"

SAMPLES = DATA / "silicon_sampling" / "qwen25_7b"
SAMPLES_CSV = SAMPLES / "samples.csv"
PROFILES_CSV = SAMPLES / "profiles.csv"

REPORT = DOCS / "reports" / "voelkel_validation"
PLOTS = REPORT / "plots"
