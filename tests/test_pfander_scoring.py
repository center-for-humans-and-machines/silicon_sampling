"""The shared Pfänder scoring design describes the grid the benchmark scores.

Small checks, but the ones that would have caught a wrong constant: the pair
count the grid assertion is compared against, and the reference level every
interaction and stereotyping coefficient is a contrast to.
"""

from __future__ import annotations

import sys

from silicon_sampling.pfander.outcomes import MODERATORS, OUTCOMES
from silicon_sampling.pfander.scoring import DESIGN

#: The codebook's first level per moderator — the omitted reference of every
#: dummy-coded contrast, and not the alphabetically first one for five of six.
CODEBOOK_REFERENCE = {
    "gender": "Male",
    "age_band": "18-29",
    "race": "White / Caucasian",
    "education": "Less than high school",
    "income": "Less than $30,000",
    "party": "Republican",
}


def test_the_pfander_grid_is_208_pairs():
    # 16 interventions against the shared control, times 13 outcomes.
    assert DESIGN.expected_pairs == 208
    assert len(DESIGN.conditions) == 17
    assert DESIGN.control == "control"
    assert DESIGN.control in DESIGN.conditions


def test_the_design_carries_the_studys_own_outcome_and_moderator_facts():
    assert tuple(DESIGN.outcomes) == OUTCOMES
    assert DESIGN.primary == "trust_multidimensional"
    assert DESIGN.binary == ("newsletter_signup",)
    # The two behavioral outcomes are off the shared 0-100 scale, so the
    # per-intervention cut runs on the other eleven.
    assert DESIGN.behavioral_outcomes == ("donation_ams", "newsletter_signup")
    assert len(DESIGN.continuous_outcomes) == 11
    assert DESIGN.scale("donation_ams") == 10.0

    assert tuple(DESIGN.moderators) == tuple(MODERATORS)
    for moderator, levels in DESIGN.moderators.items():
        assert levels[0] == CODEBOOK_REFERENCE[moderator], moderator


def main() -> int:
    tests = [
        v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)
    ]
    failures = 0
    for test in tests:
        try:
            test()
        except Exception as error:  # noqa: BLE001
            failures += 1
            print(f"FAIL  {test.__name__}: {error}")
        else:
            print(f"ok    {test.__name__}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
