"""A representative-US joint distribution over a study's demographic moderators.

The Pfänder preregistration quota-matches its panel on age, gender and race and
leaves education, income and party to fall where they may.  A base model asked to
invent the other three does badly enough to break an analysis, so this package
supplies them instead: ``codebook`` names one study's answer options and how CCAM
codes onto them, ``ccam`` recodes a nationally representative survey through that
naming, and ``joint`` turns the result into
``P(education, income, party | gender, age band, race)`` plus a sampler that draws
from it without ever disturbing the published quotas.  ``crossval`` is the
held-out comparison that picked the model.

Two studies use it — ``pfander.profiles`` and ``voelkel.profiles`` — and adding a
third is a new ``Codebook`` plus a fit, not a fork.

Rebuild the shipped tables and check them with::

    python -m silicon_sampling.demographics.cli crosswalk
    python -m silicon_sampling.demographics.cli fit
    python -m silicon_sampling.demographics.cli fit --study voelkel
    python -m silicon_sampling.demographics.cli check
    python -m silicon_sampling.demographics.cli arms
    python -m silicon_sampling.demographics.cli compare
    python -m silicon_sampling.demographics.crossval
"""

from __future__ import annotations

from .codebook import PFANDER, VOELKEL, Codebook, study
from .joint import (
    Fit,
    Sampler,
    draw,
    fit,
    marginals,
    read_table,
    sample,
    shipped,
    space,
    table_path,
    write_table,
)

__all__ = [
    "Codebook",
    "Fit",
    "PFANDER",
    "Sampler",
    "VOELKEL",
    "ccam",
    "codebook",
    "crossval",
    "draw",
    "fit",
    "joint",
    "marginals",
    "read_table",
    "sample",
    "shipped",
    "space",
    "study",
    "table_path",
    "write_table",
]
