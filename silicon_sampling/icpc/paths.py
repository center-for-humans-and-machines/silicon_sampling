"""Where the ICPC study's inputs and outputs live.

The instrument is vendored under ``data/ICPC/Materials`` from the study's own OSF
project (``osf.io/ytf89``) rather than read from the network at import time: the
Qualtrics graphic CDN has already lost six of the fifty-seven stimulus images —
five of them survive only inside ``master_survey.pdf`` and one is gone for good —
so anything not held locally is not reproducible.

The response data sits with the other calibration studies under
``data/calibration/datasets`` — it arrived through that route and is shared with
the item-matching work, so it is referenced here rather than copied.
"""

from __future__ import annotations

from pathlib import Path

from ..models import DEFAULT_RUN
from .. import paths as _paths

ROOT = Path(__file__).resolve().parents[2]
DATA = _paths.resolve("ICPC")
DOCS = ROOT / "docs"

MATERIALS = DATA / "Materials"

#: The three US teams each fielded the same survey.  The block ids, the flow and
#: the stimulus wording agree across all three files down to three spellings —
#: usa_1 has "gray", "labor" and "droughts" where usa_2 and usa_3 have "grey",
#: "labour" and the master's typo "draughts" — plus non-breaking-space noise.  ``usa_3`` is the primary because it is the
#: largest team (n = 5,055 of 8,253) and because the hand transcription in
#: ``vlasceanu.content_shared`` matches its spellings.  ``QSF_ALL`` is kept so
#: that equivalence claim can be re-checked rather than trusted, which
#: ``tests/test_icpc.py`` does.
QSF = MATERIALS / "usa_3.qsf"
QSF_ALL = (MATERIALS / "usa_1.qsf", MATERIALS / "usa_2.qsf", MATERIALS / "usa_3.qsf")

MASTER_SURVEY_PDF = MATERIALS / "master_survey.pdf"
ADAPTATION_MANUAL_PDF = MATERIALS / "intervention_adaptation_manual.pdf"
CLEANING_R = MATERIALS / "datapaper_cleaning.R"
CODEBOOK = MATERIALS / "codebook.xlsx"

#: Every ``<img>`` the US survey referenced, downloaded from the Qualtrics CDN
#: (or recovered out of ``master_survey.pdf`` where the CDN had dropped it).
STIMULI = MATERIALS / "stimuli"

CALIBRATION = _paths.resolve("calibration")
DOELL_CSV = CALIBRATION / "datasets" / "doell_etal2024.csv"
VLASCEANU_XLSX = CALIBRATION / "datasets" / "vlasceanu_etal2024.xlsx"
DOELL_PAPER = CALIBRATION / "papers" / "doell.pdf"

TEMPLATES = DATA / "text_templates"
MODALITY_AUDIT = TEMPLATES / "modality_audit.csv"

#: Root of every model's silicon-sampling output.
RUNS = _paths.output("ICPC", "silicon_sampling")


def samples_dir(run: str = DEFAULT_RUN) -> Path:
    """Where one model's run is filed."""
    return RUNS / run


SAMPLES = samples_dir()
SAMPLES_CSV = SAMPLES / "samples.csv"
PROFILES_CSV = SAMPLES / "profiles.csv"

REPORT = DOCS / "reports" / "icpc_validation"
PLOTS = REPORT / "plots"
