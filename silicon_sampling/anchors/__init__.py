"""Ground-truth control-arm levels for Pfänder's outcomes, borrowed from real surveys.

Pfänder publishes no human data by design, so nothing in a submission can be
checked against its own truth — and our synthetic respondents' *levels* are the
weakest part of what we produce.  Level error drives three of the benchmark's
scored analyses that no effect calibration can reach: the control-condition
response distributions (W1, OVL, KS), the demographic baselines over control cell
means, and the parity gap.  This package borrows those levels from nationally
representative US surveys instead of from another model.

Two sources, one honest answer.  :mod:`~silicon_sampling.anchors.tisp` carries the
same twelve-item Besley trust battery Pfänder uses as its primary outcome, plus
the norm items that became ``policy_role_mean``;
:mod:`~silicon_sampling.anchors.ccam` carries climate belief, worry and policy
support and, as it turns out, nothing that matches Pfänder closely enough to use.
:mod:`~silicon_sampling.anchors.crosswalk` holds every candidate mapping as graded
data so the grades are arguable without reading code,
:mod:`~silicon_sampling.anchors.scales` holds the categorical-to-slider conversion
and the size of its own uncertainty, and :mod:`~silicon_sampling.anchors.levels`
assembles the three anchors that survive.

:mod:`~silicon_sampling.anchors.validate` is the part that decides whether any of
it gets used.  Anchoring is rehearsed on Voelkel, where real responses exist, and
the number that matters is the break-even: the anchor error at which the gain
disappears.  It is about 4.7 points on the strictest metric for the better
sampler and about 13 for the weaker one, against an anchor uncertainty of 3 to 4
points.
"""

from __future__ import annotations

from . import ccam, crosswalk, levels, scales, tisp, validate
from .crosswalk import CROSSWALK, GRADES, UNMATCHED, Entry, at_least
from .levels import Anchor, build, facet_levels
from .scales import Weighted

__all__ = [
    "Anchor",
    "CROSSWALK",
    "Entry",
    "GRADES",
    "UNMATCHED",
    "Weighted",
    "at_least",
    "build",
    "ccam",
    "crosswalk",
    "facet_levels",
    "levels",
    "scales",
    "tisp",
    "validate",
]
