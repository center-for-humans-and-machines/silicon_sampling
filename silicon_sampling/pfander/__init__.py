"""Silicon sample of the Pfänder megastudy (the Silicon Sample Benchmark).

The instrument, the 17 conditions, the respondent profiles, the sampling run and
the analysis for a Tier-1 submission: one synthetic respondent per row, 18,000
rows, 17 conditions, 13 outcomes.

Pipeline, in order::

    python -m silicon_sampling.pfander.cli render-templates
    python -m silicon_sampling.pfander.cli validate
    python -m silicon_sampling.pfander.cli build-profiles
    scripts/run_full_sample.sh          # long; resumable
    python -m silicon_sampling.pfander.cli build-csv

Content is read from the benchmark's own shipped materials — the instrument, the
codebook and the condition code names — snapshotted under
``data/pfander/submission_template``.
"""

from __future__ import annotations

__all__ = [
    "build",
    "conditions",
    "export",
    "instrument",
    "outcomes",
    "paths",
    "profiles",
    "report",
    "run",
    "sources",
    "templates",
    "validate",
]
