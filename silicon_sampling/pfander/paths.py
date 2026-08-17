"""Where this study's inputs and outputs live.

The benchmark's submission-template repository is the authoritative source for
the instrument, the codebook and the condition labels.  A snapshot of the three
files we depend on is kept under ``data/pfander/submission_template`` so a run is
reproducible without that checkout; ``SILICON_SAMPLING_SUBMISSION_REPO`` points at
the live checkout when one is available.
"""

from __future__ import annotations

import os
from pathlib import Path

from ..models import DEFAULT_RUN
from ..sampling.engine import cache_root

#: Repository root (``.../silicon_sampling``), found by walking up from this file.
ROOT = Path(__file__).resolve().parents[2]

DATA = ROOT / "data" / "pfander"
DOCS = ROOT / "docs"

#: Snapshot of the shipped benchmark materials.
SUBMISSION_SNAPSHOT = DATA / "submission_template"

QUESTIONNAIRE_TXT = SUBMISSION_SNAPSHOT / "survey" / "questionnaire.txt"
CODENAMES_CSV = SUBMISSION_SNAPSHOT / "survey" / "condition_codenames.csv"
CODEBOOK_CSV = SUBMISSION_SNAPSHOT / "codebook.csv"

#: Rendered fill-in-the-blank templates, one per condition.
TEMPLATES = DATA / "text_templates"

#: Root of every model's silicon-sampling output.
RUNS = DATA / "silicon_sampling"


def samples_dir(run: str = DEFAULT_RUN) -> Path:
    """Where one model's run is filed."""
    return RUNS / run


#: Silicon-sampling output for the default run.  The module-level names below are
#: kept for the Qwen run that the existing reports refer to; anything that has to
#: work for both models takes a directory instead.
SAMPLES = samples_dir()
RAW = SAMPLES / "raw"
SAMPLES_CSV = SAMPLES / "samples.csv"
TIER1_CSV = SAMPLES / "tier1_submission.csv"
DRAWS_JSONL = SAMPLES / "draws.jsonl"
RUN_META = SAMPLES / "run_meta.json"
PROFILES_CSV = SAMPLES / "profiles.csv"


#: Writable cache root for compiled graphs and plotting fonts.  ``~/.cache`` is the
#: natural home and is mounted from the host here; the git-ignored data tree is the
#: fallback for an image where it is not writable, and ``SILICON_SAMPLING_CACHE``
#: overrides both, which is how the cluster points at a bound-in scratch path.
CACHE = cache_root(DATA.parent / ".cache")

#: Analysis output.
REPORT = DOCS / "reports" / "pfander_silicon_sample"
PLOTS = REPORT / "plots"


def submission_repo() -> Path | None:
    """The live submission-template checkout, if the environment names one."""
    raw = os.environ.get(
        "SILICON_SAMPLING_SUBMISSION_REPO", "/opt/silicon-sample-submission"
    )
    path = Path(raw)
    return path if path.is_dir() else None
