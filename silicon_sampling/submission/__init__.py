"""Packaging and validating a Silicon Sample Benchmark submission, without R.

The benchmark's format gate is an R script this container cannot run, so the
schema (:mod:`.spec`), the gate (:mod:`.check`) and the packaging step
(:mod:`.build`) exist here in Python.  Without them a deposit would be a guess.
"""

from __future__ import annotations

from .build import SubmissionMeta, build_submission
from .check import (
    CheckResult,
    CheckRow,
    check_bundle,
    check_repo,
    check_submission,
    print_report,
    write_report,
)
from .spec import (
    CONDITIONS,
    INTERVENTIONS,
    MODERATORS,
    OUTCOMES,
    OUTCOME_RANGE,
    REFERENCE_LEVELS,
    TIER1_COLUMNS,
    TRUST_ITEMS,
    prediction_filename,
    verify_against_codebook,
)

__all__ = [
    "CONDITIONS",
    "CheckResult",
    "CheckRow",
    "INTERVENTIONS",
    "MODERATORS",
    "OUTCOMES",
    "OUTCOME_RANGE",
    "REFERENCE_LEVELS",
    "SubmissionMeta",
    "TIER1_COLUMNS",
    "TRUST_ITEMS",
    "build_submission",
    "check_bundle",
    "check_repo",
    "check_submission",
    "prediction_filename",
    "print_report",
    "verify_against_codebook",
    "write_report",
]
