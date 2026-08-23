"""Data-only corrections to a silicon sample, and the machinery to apply them.

Four layers.  :mod:`~silicon_sampling.calibration.components` decomposes a
respondent-level sample into level, condition effect, demographic offset and
residual, and puts it back together with the condition means landing exactly
where they were aimed — that is how any correction reaches a Tier-1 submission,
since the benchmark refits everything from the rows we hand it.
:mod:`~silicon_sampling.calibration.effects` holds the transforms acting on the
fitted effects, :mod:`~silicon_sampling.calibration.offsets` those acting on how
strongly demographics move an answer, and
:mod:`~silicon_sampling.calibration.select` chooses between them under held-out
scoring.
"""

from __future__ import annotations

from . import effects, offsets, select
from .components import (
    Decomposition,
    cell_offsets,
    condition_effects,
    control_level,
    decompose,
    hybrid,
    recompose,
    recompose_frame,
)

__all__ = [
    "Decomposition",
    "effects",
    "offsets",
    "select",
    "cell_offsets",
    "condition_effects",
    "control_level",
    "decompose",
    "hybrid",
    "recompose",
    "recompose_frame",
]
