"""This study's facts, in the shape the scoring code asks for.

:mod:`silicon_sampling.benchmark.scored` deliberately knows nothing about
Pfänder: it takes a :class:`~silicon_sampling.benchmark.scored.ScoredDesign` and
scores whatever grid that describes.  Which arm is the control, which of the 13
outcomes is binary, and the codebook order of the six moderators are facts about
*this* megastudy, and the moderator order in particular is the one that silently
corrupts everything downstream when it is guessed — the interaction and
stereotyping coefficients are gaps to the first level, so an alphabetical
fallback reports the gap to "Bachelor's degree" where the codebook asks for the
gap to "Less than high school".

Hence one shared object rather than a construction at each call site.  Every
consumer — the report, the model comparison, the calibration search — scores
against the same 208-pair grid, and there is one place to change if the
benchmark's own definition moves.
"""

from __future__ import annotations

from ..benchmark.scored import ScoredDesign
from .conditions import CONDITIONS
from .outcomes import MODERATORS, SCALE_RANGE

#: The binary outcome: a Yes/No signup, scored through logistic regression and
#: average marginal effects rather than the linear model the other twelve use.
BINARY = ("newsletter_signup",)

#: The scored design of the Pfänder megastudy: 16 interventions against a shared
#: control x 13 outcomes = 208 ATE pairs, plus the six moderators' interactions.
DESIGN = ScoredDesign(
    outcomes=dict(SCALE_RANGE),
    control="control",
    moderators=MODERATORS,
    conditions=CONDITIONS,
    binary=BINARY,
)
