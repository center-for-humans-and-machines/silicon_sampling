"""Data-only corrections to a silicon sample, and the machinery to apply them.

Two layers.  :mod:`~silicon_sampling.calibration.components` decomposes a
respondent-level sample into level, condition effect, demographic offset and
residual, and puts it back together with the condition means landing exactly
where they were aimed — that is how any correction reaches a Tier-1 submission,
since the benchmark refits everything from the rows we hand it.
:mod:`~silicon_sampling.calibration.effects` holds the transforms that act on the
fitted effects themselves.
"""

from __future__ import annotations

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
    "cell_offsets",
    "condition_effects",
    "control_level",
    "decompose",
    "hybrid",
    "recompose",
    "recompose_frame",
]
