"""Checks for the Pfänder silicon-sampling pipeline.

Written to run under plain ``python tests/test_pfander.py`` as well as pytest,
because this container has no pytest installed.
"""

from __future__ import annotations

import sys

import numpy as np

from silicon_sampling.analysis.ols import (
    benjamini_hochberg,
    design_matrix,
    holm,
    ols,
    wald,
)
from silicon_sampling.pfander import outcomes, profiles, templates, validate
from silicon_sampling.pfander.build import make_session, template_code_name
from silicon_sampling.pfander.conditions import CONDITIONS, case_for_state
from silicon_sampling.survey.render import MARKER_RE, slot_manifest
from silicon_sampling.survey.slots import ChoiceSlot, FreeTextSlot, IntSlot


def test_choice_parsing():
    slot = ChoiceSlot(id="x", options=("Yes", "No", "Not applicable"))
    assert slot.parse("Yes") == "Yes"
    assert slot.parse("  No.") == "No"
    # Longest match wins: "No" must not shadow "Not applicable".
    assert slot.parse("Not applicable") == "Not applicable"
    assert slot.parse("Not applicable.") == "Not applicable"
    # Trailing prose means the model kept talking rather than answering.
    assert slot.parse("Not applicable, I think") is None
    # Echoing the option list is a hedge, not an answer.
    assert slot.parse("Yes | No") is None
    assert slot.parse("No | Not applicable? I") is None
    assert slot.parse("2 - Working class") is None
    assert slot.parse("") is None
    # Generation stops at the newline; anything after it belongs to the next line.
    assert slot.parse("No\nQ5. Something else") == "No"


def test_int_parsing():
    slot = IntSlot(id="y", lo=0, hi=100)
    assert slot.parse("78") == 78
    assert slot.parse("0") == 0
    assert slot.parse("100") == 100
    assert slot.parse(" 42%") == 42
    assert slot.parse("101") is None
    # A slider is a continuous control the survey snaps to an integer, so a
    # decimal is a real position and is rounded rather than refused.
    assert slot.parse("78.5") == 79
    assert slot.parse("92.36") == 92
    assert slot.parse("100.4") == 100
    assert slot.parse("1,200") is None, "out of range once the separator is read"
    assert slot.parse("7 (somewhat)") is None
    assert slot.parse("many") is None
    money = IntSlot(id="d", lo=0, hi=10, allow_dollar=True)
    assert money.parse("$7") == 7
    assert money.parse("$11") is None


def test_money_options_tolerate_reformatting():
    """The model retypes money; the same answer must not be rejected for it.

    Rejecting these is not merely lossy — the failure rate depends on which
    option was meant, so rejection sampling turns it into a skewed distribution.
    """
    slot = ChoiceSlot(
        id="income",
        options=(
            "Less than $30,000",
            "$30,000 to $55,999",
            "$56,000 to $99,999",
            "$100,000 to $167,999",
            "$168,000 or more",
        ),
    )
    assert slot.parse("$56,000 to $99,999") == "$56,000 to $99,999"
    assert slot.parse("56,000 to $99,999") == "$56,000 to $99,999"
    assert slot.parse("100,000 to $167,999") == "$100,000 to $167,999"
    assert slot.parse("30000 to 55999") == "$30,000 to $55,999"
    assert slot.parse("168000 or more") == "$168,000 or more"
    assert slot.parse("Less than 30000") == "Less than $30,000"
    # Still not a licence to accept trailing prose.
    assert slot.parse("$56,000 to $99,999 and rising") is None


def test_pattern_parsing():
    slot = FreeTextSlot(id="zip_code", pattern=r"\d{5}")
    assert slot.parse("90210") == "90210"
    assert slot.parse("9021") is None
    assert slot.parse("90210-1234") is None


def test_ols_matches_hand_computed_hc1():
    rng = np.random.default_rng(0)
    n = 400
    group = np.array([0] * 200 + [1] * 200)
    y = np.where(group == 1, rng.normal(10, 5, n), rng.normal(4, 1, n))
    X, names = design_matrix({"g": group.tolist()})
    fit = ols(X, y, names)

    assert np.isclose(fit.beta[1], y[group == 1].mean() - y[group == 0].mean())
    v0, v1 = y[group == 0].var(ddof=0), y[group == 1].var(ddof=0)
    expected = np.sqrt((v0 / 200 + v1 / 200) * n / (n - 2))
    assert np.isclose(fit.se[1], expected, rtol=1e-10)
    assert wald(fit, ["g[1]"])["df"] == 1


def test_multiplicity_adjustments():
    raw = [0.001, 0.02, 0.03, 0.5]
    assert [round(value, 4) for value in holm(raw)] == [0.004, 0.06, 0.06, 0.5]
    assert [round(value, 4) for value in benjamini_hochberg(raw)] == [
        0.004,
        0.04,
        0.04,
        0.5,
    ]
    # Both are monotone in the raw p-values and never shrink them.
    for adjusted in (holm(raw), benjamini_hochberg(raw)):
        assert all(a >= b for a, b in zip(adjusted, raw))


def test_profiles_reproduce_the_preregistered_quotas():
    built = profiles.build()
    assert len(built) == profiles.TOTAL_N

    for band, total, male, female in profiles.AGE_QUOTA:
        got = sum(1 for p in built if p.age_band == band)
        assert abs(got - total) <= 3, f"{band}: {got} vs {total}"
        assert (
            abs(
                sum(1 for p in built if p.age_band == band and p.gender == "Male")
                - male
            )
            <= 3
        )

    for race, total, male, female in profiles.RACE_QUOTA:
        got = sum(1 for p in built if p.race == race)
        assert abs(got - total) <= 3, f"{race}: {got} vs {total}"

    counts = {}
    for profile in built:
        counts[profile.condition] = counts.get(profile.condition, 0) + 1
    assert counts["control"] == profiles.CONTROL_N
    assert all(
        counts[title] == profiles.PER_INTERVENTION_N
        for title in counts
        if title != "control"
    )

    # Same seed, same sample.
    again = profiles.build()
    assert [p.profile_id for p in built] == [p.profile_id for p in again]
    assert [p.age for p in built] == [p.age for p in again]


def test_age_band_boundaries():
    assert outcomes.age_band(2026 - 18) == "18-29"
    assert outcomes.age_band(2026 - 29) == "18-29"
    assert outcomes.age_band(2026 - 30) == "30-44"
    assert outcomes.age_band(2026 - 59) == "45-59"
    assert outcomes.age_band(2026 - 60) == "60+"


def test_state_to_case_mapping_is_total_and_disjoint():
    from silicon_sampling.pfander.instrument import STATES

    assert len(STATES) == 51
    cases = [case_for_state(state) for state in STATES]
    assert set(cases) == {
        1,
        2,
        3,
    }, "every state must map to a real case, never the fallback"
    assert case_for_state("Prefer not to say") == 4
    assert case_for_state(None) == 4


def test_instrument_validates():
    assert validate.check() == []


def test_every_condition_yields_every_outcome():
    for condition in CONDITIONS:
        session = make_session(
            "p00001", condition, code_name=template_code_name(condition)
        )
        while (step := session.next_prompt()) is not None:
            text, slot = step
            assert not MARKER_RE.search(
                text
            ), f"{condition}/{slot.id}: marker leaked into the prompt"
            assert text.endswith("Response: ")
            session.submit(slot, validate._dummy(slot))
        computed = outcomes.compute(session.answers)
        assert all(name in computed for name in outcomes.OUTCOMES)


def test_composites_follow_the_codebook():
    answers = {item: 0 for item in outcomes.REQUIRED_ITEMS}
    answers.update({item: 10 for item in outcomes.SUBSCALES["trust_competence"]})
    answers.update({item: 20 for item in outcomes.SUBSCALES["trust_integrity"]})
    answers.update({item: 30 for item in outcomes.SUBSCALES["trust_benevolence"]})
    answers.update({item: 40 for item in outcomes.SUBSCALES["trust_openness"]})
    answers["funding_5"] = 70
    answers["newsletter"] = "Yes"
    answers.update(
        {
            "gender": "Male",
            "year_birth": 1990,
            "race": "White / Caucasian",
            "education": "Bachelor's degree",
            "income": "$168,000 or more",
            "party": "Democrat",
        }
    )

    computed = outcomes.compute(answers)
    # Mean of the four subscale means, per the codebook — not the mean of 12 items.
    assert computed["trust_multidimensional"] == 25.0
    assert (
        computed["funding_perceptions"] == 30.0
    ), "higher must mean 'supports more funding'"
    assert computed["newsletter_signup"] == 1
    assert computed["age_band"] == "30-44"


def test_templates_declare_every_slot():
    manifest = templates.render_all()
    assert len(manifest["conditions"]) == 17
    for condition, info in manifest["conditions"].items():
        ids = [slot["id"] for slot in info["slots"]]
        assert len(ids) == len(set(ids)), f"{condition}: duplicate slot ids"
        assert info["code_names"], f"{condition}: no survey code name"
        for slot in info["slots"]:
            assert slot["legal"], f"{condition}/{slot['id']}: no legal-value spec"
        text = (templates.TEMPLATES / info["file"]).read_text(encoding="utf-8")
        for slot in info["slots"]:
            assert (
                f"<<{slot['id']} ::" in text
            ), f"{condition}: {slot['id']} missing from the template file"


def test_prefilled_slots_are_exactly_the_intended_ones():
    slots = slot_manifest(templates.template_elements("control"))
    prefilled = {slot["id"] for slot in slots if slot["source"] == "prefilled"}
    assert prefilled == {
        "filter",
        "filter_ai",
        "gender",
        "year_birth",
        "race",
        "attention1",
        "attention2",
    }


def main() -> int:
    tests = [
        value
        for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    ]
    failures = 0
    for test in tests:
        try:
            test()
        except Exception as error:  # noqa: BLE001 - a test runner reports everything
            failures += 1
            print(f"FAIL  {test.__name__}: {error}")
        else:
            print(f"ok    {test.__name__}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
