"""The demographic module: crosswalk, fit, and the quota margins it must not move.

Three kinds of claim are checked here, and they are worth separating because they
fail for different reasons.

**The crosswalk** is where this package can actually be wrong — it is a claim
about two questionnaires, not about arithmetic — so every mapping decision is
checked against the study's own answer options, twice over: once for Pfänder and
once for Voelkel.  Those tests need no data file at all, which is the point of
having moved the level strings into ``codebook``: a second study's vocabulary is
checkable without touching CCAM.

**The fit** is checked as identities rather than by eyeballing margins.  IPF is
multiplicative, so "stage two moves only main effects" is an exact factorisation
of the ratio between the two tables, not an approximate statement about a
marginal, and it is asserted that way.

**The quotas** are the constraint the whole module is subordinate to, and they are
integer counts, so they are asserted with ``==``.  The one place that cannot be an
equality is the comparison against the *published* tables: those two tables
disagree with each other by one man, so ±1 is the best any joint can do, and
``test_the_published_quota_tables_are_matched_to_the_rounding_residue`` pins the
residue to the apportionment that was already in ``pfander.profiles`` rather than
letting this module contribute to it.

The tradeoff the file makes: the fixtures refit from the .sav rather than reading
the shipped CSV, which costs a few seconds per session but means a stale shipped
table is caught by ``test_the_shipped_table_matches_a_fresh_fit`` instead of
silently becoming the thing every other test agrees with.
"""

from __future__ import annotations

import collections
import importlib.util
import random

import numpy as np
import pytest

from silicon_sampling.demographics import ccam, codebook, joint
from silicon_sampling.demographics.codebook import PFANDER, STUDIES, VOELKEL, Codebook
from silicon_sampling.pfander import instrument
from silicon_sampling.pfander.outcomes import MODERATORS
from silicon_sampling.pfander.profiles import AGE_QUOTA, RACE_QUOTA, cell_counts

BOOKS = [PFANDER, VOELKEL]
BOOK_IDS = [book.name for book in BOOKS]

_HAS_PYREADSTAT = importlib.util.find_spec("pyreadstat") is not None

#: The crosswalk tests are pure Python and always run.  Everything that reads
#: microdata is gated on the .sav *and* on the reader, because the reader is not
#: in the runtime image — that is why the fitted tables ship as CSVs.
needs_ccam = pytest.mark.skipif(
    not (_HAS_PYREADSTAT and ccam.CCAM_SAV.exists()),
    reason=f"{ccam.CCAM_SAV} or pyreadstat is not in this checkout",
)

#: A sampler can come from either source, so these tests need only one of them.
needs_sampler = pytest.mark.skipif(
    not (
        joint.table_path(PFANDER).exists()
        or (_HAS_PYREADSTAT and ccam.CCAM_SAV.exists())
    ),
    reason="neither a shipped table nor CCAM to refit one from",
)


def fit_for(book: Codebook) -> joint.Fit:
    """A study's fit with that study's own calibration targets.

    Pfänder takes ``fit``'s defaults — CCAM's 2024 levels and the published
    recruitment quota.  Voelkel has neither: its levels are its own published
    marginals and its demographic composition is the product of three of them, so
    the caller has to supply both.  Reaching into ``voelkel.profiles`` for them
    here is deliberate: if those targets were duplicated in the test, the test
    would keep passing after the study package stopped agreeing with it.
    """
    if book is VOELKEL:
        from silicon_sampling.voelkel import profiles as voelkel_profiles

        return joint.fit(
            book,
            targets=voelkel_profiles.level_targets(),
            demographics=voelkel_profiles.given_margin(),
        )
    return joint.fit(book)


def structure_only(book: Codebook) -> np.ndarray:
    """Stage one alone: the max-entropy table over CCAM's two-way margins."""
    shape = joint.space(book)
    observed = joint.contingency(ccam.donor_table(book), book)
    rank = len(shape.axes)
    table, _ = joint.ipf(
        np.ones(shape.dims),
        [
            (pair, observed.sum(axis=tuple(k for k in range(rank) if k not in pair)))
            for pair in shape.pairs
        ],
    )
    return table


# --------------------------------------------------------------------------- #
# the crosswalk
# --------------------------------------------------------------------------- #


def test_crosswalk_targets_are_exactly_the_benchmark_levels():
    assert PFANDER.education == MODERATORS["education"]
    assert PFANDER.income == MODERATORS["income"]
    assert PFANDER.party == MODERATORS["party"]
    assert PFANDER.age_bands == MODERATORS["age_band"]
    assert set(PFANDER.race_to_ccam) == set(MODERATORS["race"])
    assert set(PFANDER.education_from_ccam.values()) <= set(PFANDER.education)
    assert set(PFANDER.party_from_ccam.values()) <= set(PFANDER.party)
    assert set(codebook.NO_PARTY_SHARES) <= set(PFANDER.party)
    # The instrument's third gender option is the one level CCAM cannot hold, so
    # it has to be declared collapsible rather than crosswalked.
    covered = set(PFANDER.gender) | {
        level for axis, level in PFANDER.collapsible if axis == "gender"
    }
    assert covered == set(MODERATORS["gender"])


@pytest.mark.parametrize("book", BOOKS, ids=BOOK_IDS)
def test_every_ccam_code_maps_to_a_full_unit_of_probability(book):
    for axis, code in _all_codes(book):
        shares = book.shares(axis, code)
        assert set(shares) <= set(book.levels(axis)), f"{book.name}/{axis} {code}"
        assert sum(shares.values()) == pytest.approx(1.0), f"{book.name}/{axis} {code}"


def _all_codes(book: Codebook) -> list[tuple[str, int]]:
    """Every ``(axis, CCAM code)`` the codebook claims to handle."""
    out: list[tuple[str, int]] = []
    for axis in book.drawn:
        if axis == "income":
            out += [("income", code) for code in sorted(codebook.INCOME_BRACKETS)]
            continue
        direct = (
            book.education_from_ccam if axis == "education" else book.party_from_ccam
        )
        split = book.education_split if axis == "education" else book.party_split
        out += [(axis, code) for code in sorted(set(direct) | set(split))]
    return out


@pytest.mark.parametrize("book", BOOKS, ids=BOOK_IDS)
def test_education_covers_every_ccam_code_once(book):
    """Every CCAM ``educ`` code lands somewhere, and no code lands twice.

    "Once" is the whole content of the test: a code in both ``education_from_ccam``
    and ``education_split`` would have its split silently win, which is a mapping
    decision made by dictionary lookup order rather than by anyone.
    """
    direct = set(book.education_from_ccam)
    split = set(book.education_split)
    assert direct | split == set(range(1, 15))
    assert not direct & split


@pytest.mark.parametrize("book", BOOKS, ids=BOOK_IDS)
def test_only_the_declared_ccam_categories_span_two_levels(book):
    """``split_codes`` is the crosswalk's whole judgement-call budget.

    Four for Pfänder — one education merge, two income brackets, and the no-party
    group — and none at all for Voelkel, whose coarser levels happen to line up
    with CCAM's exactly.  A fifth appearing here is a mapping decision nobody
    wrote a paragraph about.
    """
    expected = {
        "pfander": (
            ("education", 14),
            ("income", 12),
            ("income", 18),
            ("party", codebook.PARTY_NO_PARTY),
        ),
        "voelkel": (),
    }[book.name]
    assert book.split_codes == expected


def test_only_two_income_brackets_straddle_a_benchmark_cut():
    split = [
        axis_code[1] for axis_code in PFANDER.split_codes if axis_code[0] == "income"
    ]
    assert split == [12, 18]
    assert PFANDER.income_shares(12)["$30,000 to $55,999"] == pytest.approx(
        0.6216, abs=1e-4
    )
    assert PFANDER.income_shares(18)["$100,000 to $167,999"] == pytest.approx(
        0.7352, abs=1e-4
    )


def test_income_brackets_are_ordered_and_cover_the_cuts():
    brackets = codebook.INCOME_BRACKETS
    codes = sorted(brackets)
    for left, right in zip(codes, codes[1:]):
        assert brackets[left][1] <= brackets[right][0]
    for cut in PFANDER.income_cuts:
        assert any(low <= cut <= high for low, high in brackets.values())


def test_the_open_ended_brackets_never_carry_a_cut_point():
    """The two end brackets' outer bounds are invented, so a cut inside one would
    put an invented dollar figure into the arithmetic.  ``Codebook.check`` refuses
    that per study rather than trusting the reader to notice."""
    with pytest.raises(ValueError, match="open-ended income bracket"):
        Codebook(
            name="bad-cut",
            education=PFANDER.education,
            education_from_ccam=PFANDER.education_from_ccam,
            education_split=PFANDER.education_split,
            party=PFANDER.party,
            party_from_ccam=PFANDER.party_from_ccam,
            party_split=PFANDER.party_split,
            income=("under $2,000", "$2,000 or more"),
            income_cuts=(2_000,),
        )


def test_a_crosswalk_onto_an_undeclared_level_is_refused():
    with pytest.raises(ValueError, match="unknown level"):
        Codebook(
            name="bad-target",
            education=("HS or less",),
            education_from_ccam={code: "HS or less" for code in range(1, 14)}
            | {14: "Doctorate"},
            party=("Republican",),
            party_from_ccam={1: "Republican"},
        )


def test_a_split_that_does_not_sum_to_one_is_refused():
    with pytest.raises(ValueError, match="shares sum to"):
        Codebook(
            name="bad-split",
            education=("HS or less", "Degree"),
            education_from_ccam={code: "HS or less" for code in range(1, 14)},
            education_split={14: {"HS or less": 0.4, "Degree": 0.4}},
            party=("Republican",),
            party_from_ccam={1: "Republican"},
        )


def test_age_bands_are_the_benchmark_cuts():
    assert [codebook.age_band(age) for age in (18, 29, 30, 44, 45, 59, 60, 95)] == [
        "18-29",
        "18-29",
        "30-44",
        "30-44",
        "45-59",
        "45-59",
        "60+",
        "60+",
    ]


def test_the_professional_share_is_pinned():
    """The module's least defensible number, and the only one not read off a file.

    It alone determines the ``Doctorate degree / Ph.D.`` level, so it is pinned
    here: a change to it should have to be an edit to a test, not a silent shift
    in a moderator level.
    """
    assert codebook.PROFESSIONAL_SHARE == 0.52
    assert PFANDER.education_split[14] == {
        "Master's degree / Professional degree": 0.52,
        "Doctorate degree / Ph.D.": pytest.approx(0.48),
    }


def test_every_study_is_reachable_by_name():
    assert STUDIES == {"pfander": PFANDER, "voelkel": VOELKEL}
    assert codebook.study("voelkel") is VOELKEL
    with pytest.raises(ValueError, match="known studies"):
        codebook.study("pfaender")


# --------------------------------------------------------------------------- #
# the second study: the same machinery, a different vocabulary
# --------------------------------------------------------------------------- #


def test_voelkel_draws_two_axes_and_pfander_three():
    """A study that never asks an item gets a joint without that axis, not a
    silent national-average fill for it."""
    assert PFANDER.drawn == ("education", "income", "party")
    assert VOELKEL.drawn == ("education", "party")
    assert VOELKEL.income == () and VOELKEL.income_cuts == ()
    assert VOELKEL.income_shares(12) == {}
    assert joint.space(PFANDER).dims == (2, 4, 4, 6, 5, 4)
    assert joint.space(VOELKEL).dims == (2, 4, 4, 4, 3)


def test_voelkel_folds_other_and_no_party_onto_independent():
    """Voelkel screened out non-leaning independents and has no "Other" level, so
    every respondent who named neither major party has exactly one place to go."""
    assert "Other" not in VOELKEL.party
    for code in (3, 4, codebook.PARTY_NO_PARTY):
        assert VOELKEL.party_shares(code) == {"Independent": 1.0}
    assert VOELKEL.party_codes == (1, 2, 3, 4, 5)
    # Pfänder's forced four-option stem cannot do that, hence its 94/6 split.
    assert set(PFANDER.party_shares(codebook.PARTY_NO_PARTY)) == {
        "Independent",
        "Other",
    }


def test_the_professional_share_never_enters_a_voelkel_number():
    """Voelkel's four education levels split no CCAM code, so the one judgement
    call in the Pfänder crosswalk cannot reach this study."""
    assert VOELKEL.education_split == {}
    assert VOELKEL.education_from_ccam[14] == "Postgraduate"
    assert VOELKEL.education_from_ccam[13] == "Postgraduate"


@pytest.mark.parametrize("book", BOOKS, ids=BOOK_IDS)
def test_each_study_ships_its_table_under_its_own_name(book):
    expected = {"pfander": "us_joint.csv", "voelkel": "voelkel_joint.csv"}[book.name]
    assert book.table_name == expected
    assert joint.table_path(book).name == expected


# --------------------------------------------------------------------------- #
# the donor table
# --------------------------------------------------------------------------- #


@needs_ccam
def test_no_party_split_matches_the_ratio_it_claims_to_come_from():
    """94/6 is meant to be CCAM's own Independent:Other ratio, not a guess."""
    frame = ccam.load()
    rows = frame[frame["wave"].isin(list(ccam.STRUCTURE_WAVES))]
    independent = float(rows.loc[rows["party"] == 3, "weight_aggregate"].sum())
    other = float(rows.loc[rows["party"] == 4, "weight_aggregate"].sum())
    observed = independent / (independent + other)
    assert codebook.NO_PARTY_SHARES["Independent"] == pytest.approx(observed, abs=0.01)


@needs_ccam
@pytest.mark.parametrize("book", BOOKS, ids=BOOK_IDS)
def test_donor_table_is_a_clean_reweighting_of_the_kept_respondents(book):
    donors = ccam.donor_table(book)
    frame = ccam.load()
    rows = frame[frame["wave"].isin(list(ccam.STRUCTURE_WAVES))]
    kept = rows[
        rows["income"].between(1, max(codebook.INCOME_BRACKETS))
        & rows["party"].isin(list(book.party_codes))
    ]
    # Splitting a bracket divides a weight; it must never create or destroy one.
    assert donors["weight"].sum() == pytest.approx(
        float(kept["weight_aggregate"].sum())
    )
    assert (donors["weight"] > 0).all()
    assert len(donors) >= len(kept)
    for axis in book.axes:
        assert set(donors[axis]) <= set(book.levels(axis))


@needs_ccam
def test_a_study_whose_crosswalk_splits_nothing_gets_one_row_per_respondent():
    """The donor table only grows where the crosswalk had a judgement to make.

    Voelkel splits no CCAM code, so its table is exactly the respondents it keeps
    — which is also the cheapest available check that ``dropped`` counts the same
    rows the donor table drops.  Pfänder's three splits turn 6,147 respondents
    into more rows than that.
    """
    counts = ccam.dropped(VOELKEL)
    kept = counts["rows"] - counts["party_unmapped"] - counts["income_out_of_range"]
    assert len(ccam.donor_table(VOELKEL)) == kept
    assert len(ccam.donor_table(PFANDER)) > kept


@needs_ccam
@pytest.mark.parametrize("book", BOOKS, ids=BOOK_IDS)
def test_every_conditioning_cell_has_donors(book):
    sizes = ccam.cell_sizes(ccam.donor_table(book))
    assert len(sizes) == len(book.gender) * len(book.age_bands) * len(
        codebook.CCAM_RACE
    )
    assert sizes.min() >= 40


@needs_ccam
def test_the_quoted_wave_drifts_are_what_the_file_says():
    """``PARTY_DRIFT`` and ``INCOME_TOP_DRIFT`` are the reason levels come from
    2024 and associations from 2022-2024, so they are measured, not asserted."""
    pooled = ccam.donor_table(PFANDER, ccam.STRUCTURE_WAVES)
    recent = ccam.donor_table(PFANDER, ccam.LEVEL_WAVES)
    last = ccam.donor_table(PFANDER, (max(ccam.STRUCTURE_WAVES),))
    party = [
        100 * ccam.marginal(donors, "party", PFANDER.party)["Republican"]
        for donors in (pooled, recent)
    ]
    top = [
        100 * ccam.marginal(donors, "income", PFANDER.income)["$168,000 or more"]
        for donors in (pooled, recent, last)
    ]
    assert [round(value, 1) for value in party] == list(ccam.PARTY_DRIFT)
    assert [round(value, 1) for value in top] == list(ccam.INCOME_TOP_DRIFT)


@needs_ccam
def test_the_two_ccam_weights_differ_by_a_per_wave_constant():
    ratios = ccam.weight_ratios()
    assert len(ratios) == len(ccam.STRUCTURE_WAVES)
    low, high = ccam.WEIGHT_RATIO_RANGE
    assert float(ratios.min()) == pytest.approx(low, abs=5e-5)
    assert float(ratios.max()) == pytest.approx(high, abs=5e-5)


@needs_ccam
def test_the_weight_choice_moves_a_conditional_by_a_rounding_error():
    """``weight_aggregate`` vs ``weight_wave``: the docstring's 0.0029, measured.

    The two weights reweight the waves against each other by 7.9%, so the claim
    that the choice is immaterial has to be a measurement of what it moves — the
    association structure — rather than an argument from the spread.
    """
    shape = joint.space(PFANDER)
    aggregate = joint.fit(PFANDER)
    per_wave = joint.fit(
        PFANDER,
        donors=ccam.donor_table(PFANDER, ccam.STRUCTURE_WAVES, None, "weight_wave"),
    )
    distance = 0.5 * np.abs(aggregate.conditional() - per_wave.conditional()).sum(
        axis=shape.drawn
    )
    assert float(distance.max()) == pytest.approx(ccam.WEIGHT_CHOICE_MAX_TV, abs=5e-5)
    # Stage two pins the drawn marginals, so the weight cannot move them at all.
    for axis in PFANDER.drawn:
        np.testing.assert_allclose(
            aggregate.marginal(axis).to_numpy(),
            per_wave.marginal(axis).to_numpy(),
            atol=1e-10,
        )


def test_the_unfillable_slots_are_real_pfander_items():
    """``UNFILLABLE_SLOTS`` and ``FILLABLE_SLOTS`` are claims about the instrument.

    A list of slot ids CCAM cannot fill is only useful if the ids exist; a typo
    would turn the claim into a comment about nothing.
    """
    slots = {
        payload.id
        for event, payload in _walk(
            instrument.PRE_CONDITION + instrument.POST_RANDOMISED
        )
        if event == "slot"
    }
    for slot_id in ccam.UNFILLABLE_SLOTS:
        assert slot_id in slots, slot_id
    for slot_id in ccam.FILLABLE_SLOTS:
        assert slot_id in slots, slot_id
    # Nothing CCAM could fill is claimed to be unfillable.
    assert not set(ccam.UNFILLABLE_SLOTS) & set(ccam.FILLABLE_SLOTS)
    # And none of them is a moderator: those are what the joint already draws.
    assert not (set(ccam.UNFILLABLE_SLOTS) | set(ccam.FILLABLE_SLOTS)) & set(MODERATORS)


def _walk(elements):
    from silicon_sampling.survey.render import walk

    return list(walk(list(elements)))


@needs_ccam
def test_the_fillable_slots_name_real_ccam_columns():
    columns = set(ccam.load().columns)
    for slot_id, column in ccam.FILLABLE_SLOTS.items():
        assert column in columns, f"{slot_id} -> {column}"


# --------------------------------------------------------------------------- #
# the fit
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def model():
    if not (_HAS_PYREADSTAT and ccam.CCAM_SAV.exists()):
        pytest.skip("CCAM is not in this checkout")
    return fit_for(PFANDER)


@pytest.fixture(scope="module")
def voelkel_model():
    if not (_HAS_PYREADSTAT and ccam.CCAM_SAV.exists()):
        pytest.skip("CCAM is not in this checkout")
    return fit_for(VOELKEL)


def test_both_ipf_passes_converge(model):
    assert model.structure_error < 1e-10
    assert model.level_error < 1e-10


def test_both_ipf_passes_converge_for_the_second_study(voelkel_model):
    assert voelkel_model.structure_error < 1e-10
    assert voelkel_model.level_error < 1e-10


@pytest.mark.parametrize("name", BOOK_IDS)
def test_the_fitted_table_has_no_structural_zeros(name, model, voelkel_model):
    fitted = {"pfander": model, "voelkel": voelkel_model}[name]
    assert fitted.smallest_cell > 0
    # A zero would be a level a whole quota cell could never draw; the smallest
    # cell here is far below one expected respondent in 18,000, which is fine —
    # it just must not be exactly zero.
    assert fitted.smallest_cell * 18000 < 1.0
    assert fitted.table.sum() == pytest.approx(1.0)


def test_levels_are_calibrated_to_the_level_waves(model):
    targets = ccam.level_targets()
    for axis in PFANDER.drawn:
        np.testing.assert_allclose(
            model.marginal(axis).to_numpy(), targets[axis].to_numpy(), atol=1e-10
        )


def test_the_second_study_is_calibrated_to_its_own_published_marginals(voelkel_model):
    """Voelkel's levels are its paper's, not CCAM's — CCAM only lends covariance.

    This is the test that would catch the failure the refactor makes possible: a
    second study silently inheriting ``fit``'s Pfänder defaults and coming out as
    a nationally-representative sample of the wrong country's education system.
    """
    from silicon_sampling.voelkel import profiles as voelkel_profiles

    national = ccam.level_targets(VOELKEL)
    for axis, published in (
        ("education", voelkel_profiles.EDUCATION),
        ("party", voelkel_profiles.PARTY_GEN),
    ):
        fitted = voelkel_model.marginal(axis)
        assert list(fitted.index) == list(published)
        np.testing.assert_allclose(
            fitted.to_numpy(), list(published.values()), atol=1e-10
        )
        # The test only means something if the two candidate targets differ, and
        # they differ a lot: Voelkel's panel is 42.6% Republican against CCAM's
        # 27.8%, and better educated by 19 points at the bottom of the scale.
        gap = np.abs(np.array(list(published.values())) - national[axis].to_numpy())
        assert gap.max() > 0.05, axis


def test_the_demographic_margin_is_the_quota(model):
    shape = joint.space(PFANDER)
    fitted = model.table.sum(axis=shape.drawn)
    np.testing.assert_allclose(fitted, joint.quota_margin(), atol=1e-10)


def test_the_second_studys_demographic_margin_is_the_one_it_was_given(voelkel_model):
    from silicon_sampling.voelkel import profiles as voelkel_profiles

    shape = joint.space(VOELKEL)
    fitted = voelkel_model.table.sum(axis=shape.drawn)
    np.testing.assert_allclose(fitted, voelkel_profiles.given_margin(), atol=1e-10)


@needs_ccam
@pytest.mark.parametrize("book", BOOKS, ids=BOOK_IDS)
def test_the_structure_pass_reproduces_every_two_way_ccam_margin(book):
    shape = joint.space(book)
    rank = len(shape.axes)
    observed = joint.contingency(ccam.donor_table(book), book)
    structure = structure_only(book)
    for pair in shape.pairs:
        collapse = tuple(k for k in range(rank) if k not in pair)
        np.testing.assert_allclose(
            structure.sum(axis=collapse), observed.sum(axis=collapse), atol=1e-9
        )


@needs_ccam
@pytest.mark.parametrize("book", BOOKS, ids=BOOK_IDS)
def test_the_level_pass_only_moves_main_effects(book):
    """Stage two is multiplicative in f(g,a,r).u(e).v(i).w(p), so it moves no interaction.

    Checked as an exact identity rather than by eyeballing a margin: if the ratio
    between the calibrated and the structural table factorises that way, then
    every odds ratio CCAM estimated survives the calibration.  The exponent is the
    number of drawn axes, which is what makes the same identity hold for a study
    that asks two of them rather than three.
    """
    shape = joint.space(book)
    ratio = fit_for(book).table / structure_only(book)
    origin = (0,) * len(shape.axes)
    given = len(shape.given)
    rng = np.random.default_rng(0)
    for _ in range(300):
        cell = tuple(int(rng.integers(dim)) for dim in shape.dims)
        left = ratio[cell] * ratio[origin] ** len(shape.drawn)
        right = ratio[cell[:given] + origin[given:]]
        for axis in shape.drawn:
            key = list(origin)
            key[axis] = cell[axis]
            right = right * ratio[tuple(key)]
        assert left == pytest.approx(right, rel=1e-8)


def test_the_conditional_keeps_ccams_party_by_race_gradient(model):
    """Black respondents must stay far more Democratic than White ones."""
    share = _by_race(model, "party")
    democrat = PFANDER.party.index("Democrat")
    republican = PFANDER.party.index("Republican")
    assert share["Black, Non-Hispanic"][democrat] > 0.5
    assert share["Black, Non-Hispanic"][republican] < 0.1
    assert share["White, Non-Hispanic"][republican] > 0.3
    assert share["Hispanic"][democrat] > share["White, Non-Hispanic"][democrat]


def test_the_second_study_keeps_the_same_gradient(voelkel_model):
    """The whole reason Voelkel borrows CCAM is covariance, so the covariance has
    to survive being calibrated to a different set of levels."""
    share = _by_race(voelkel_model, "party")
    democrat = VOELKEL.party.index("Democrat")
    republican = VOELKEL.party.index("Republican")
    assert (
        share["Black, Non-Hispanic"][democrat]
        > share["Black, Non-Hispanic"][republican]
    )
    assert (
        share["White, Non-Hispanic"][republican]
        > share["Black, Non-Hispanic"][republican]
    )
    assert share["Hispanic"][democrat] > share["White, Non-Hispanic"][democrat]


def _by_race(fitted: joint.Fit, axis: str) -> dict[str, np.ndarray]:
    """``P(axis | race)`` for each CCAM race category."""
    shape = fitted.space
    race = shape.axes.index("race")
    position = shape.axes.index(axis)
    out = {}
    for index, level in enumerate(codebook.CCAM_RACE):
        slab = np.take(fitted.table, index, axis=race)
        keep = position - 1 if position > race else position
        collapse = tuple(k for k in range(slab.ndim) if k != keep)
        totals = slab.sum(axis=collapse)
        out[level] = totals / totals.sum()
    return out


def test_the_conditional_keeps_the_education_income_gradient(model):
    shape = joint.space(PFANDER)
    education = shape.axes.index("education")
    income = shape.axes.index("income")
    top = PFANDER.income.index("$168,000 or more")
    bottom = PFANDER.income.index("Less than $30,000")
    by_education = []
    for index in range(len(PFANDER.education)):
        slab = np.take(model.table, index, axis=education)
        keep = income - 1 if income > education else income
        collapse = tuple(k for k in range(slab.ndim) if k != keep)
        totals = slab.sum(axis=collapse)
        by_education.append(totals / totals.sum())
    assert by_education[0][bottom] > 0.3
    assert by_education[0][top] < 0.1
    assert by_education[4][top] > 0.35
    assert by_education[4][bottom] < 0.05


# --------------------------------------------------------------------------- #
# the shipped table and the sampler
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("name", BOOK_IDS)
def test_the_shipped_table_round_trips(tmp_path, name, model, voelkel_model):
    fitted = {"pfander": model, "voelkel": voelkel_model}[name]
    path = joint.write_table(tmp_path / f"{name}.csv", fitted)
    np.testing.assert_allclose(
        joint.read_table(path, fitted.book), fitted.table, atol=1e-12
    )


@pytest.mark.parametrize("name", BOOK_IDS)
def test_the_shipped_table_matches_a_fresh_fit(name, model, voelkel_model):
    fitted = {"pfander": model, "voelkel": voelkel_model}[name]
    path = joint.table_path(fitted.book)
    if not path.exists():
        pytest.skip(f"no shipped {name} table in this checkout")
    np.testing.assert_allclose(
        joint.read_table(path, fitted.book), fitted.table, atol=1e-9
    )


def test_one_studys_table_is_not_readable_as_anothers(tmp_path, voelkel_model):
    """Two studies, two tables, and no silent fallback between them: a Voelkel
    table read with the Pfänder codebook is an error, not a reshape."""
    path = joint.write_table(tmp_path / "voelkel.csv", voelkel_model)
    with pytest.raises(ValueError, match="are not the pfander axes"):
        joint.read_table(path, PFANDER)


@needs_sampler
def test_draw_is_deterministic_and_returns_codebook_levels():
    first = [
        joint.draw("Male", "30-44", "White / Caucasian", random.Random(7))
        for _ in range(3)
    ]
    second = [
        joint.draw("Male", "30-44", "White / Caucasian", random.Random(7))
        for _ in range(3)
    ]
    assert first == second
    for answer in first:
        assert answer["education"] in MODERATORS["education"]
        assert answer["income"] in MODERATORS["income"]
        assert answer["party"] in MODERATORS["party"]


@needs_ccam
def test_the_second_studys_sampler_draws_the_second_studys_levels():
    """The one failure the strict codebook exists to prevent, from the other side:
    a Voelkel draw must come back in Voelkel's vocabulary, not Pfänder's."""
    from silicon_sampling.voelkel import profiles as voelkel_profiles

    sampler = voelkel_profiles.demographic_sampler()
    assert sampler.book is VOELKEL
    for seed in range(20):
        answer = sampler.draw("Female", "45-59", "Black", random.Random(seed))
        assert set(answer) == {"education", "party"}
        assert answer["education"] in VOELKEL.education
        assert answer["party"] in VOELKEL.party


@needs_sampler
def test_draw_accepts_both_race_spellings():
    onscreen = joint.cell_conditional("Female", "60+", "Black / African-American")
    submission = joint.cell_conditional("Female", "60+", "Black / African American")
    np.testing.assert_allclose(onscreen, submission)


@needs_sampler
def test_asian_and_other_share_ccams_residual_category():
    np.testing.assert_allclose(
        joint.cell_conditional("Male", "18-29", "Asian / Asian-American"),
        joint.cell_conditional("Male", "18-29", "Other"),
    )


@needs_sampler
def test_an_unknown_level_collapses_that_axis_instead_of_failing():
    """gender="Other" never comes out of the quotas, but the instrument allows it."""
    collapsed = joint.cell_conditional("Other", "45-59", "White / Caucasian")
    male = joint.cell_conditional("Male", "45-59", "White / Caucasian")
    female = joint.cell_conditional("Female", "45-59", "White / Caucasian")
    assert collapsed.sum() == pytest.approx(1.0)
    assert (np.minimum(male, female) - 1e-12 <= collapsed).all()
    assert (collapsed <= np.maximum(male, female) + 1e-12).all()
    answer = joint.draw("Other", "45-59", "White / Caucasian", random.Random(1))
    assert answer["party"] in MODERATORS["party"]


@needs_sampler
def test_a_level_the_codebook_does_not_know_raises_instead_of_collapsing():
    """Being strict is the whole point: a national-average draw looks exactly like
    a working sampler, so an unrecognised level has to be loud."""
    with pytest.raises(ValueError, match="not a codebook level"):
        joint.cell_conditional("Male", "45-59", "Klingon")
    with pytest.raises(ValueError, match="not a codebook level"):
        joint.cell_conditional("Male", "16-17", "White / Caucasian")
    # Pfänder's third gender option is collapsible; Voelkel's own race spellings
    # are not Pfänder's, and must not be quietly accepted by Pfänder's sampler.
    with pytest.raises(ValueError, match="not a codebook level"):
        joint.cell_conditional("Male", "45-59", "White")


# --------------------------------------------------------------------------- #
# the 18,000-respondent draw
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def drawn():
    if not (
        joint.table_path(PFANDER).exists()
        or (_HAS_PYREADSTAT and ccam.CCAM_SAV.exists())
    ):
        pytest.skip("no table to draw from")
    return joint.sample(cell_counts(), seed=20260814)


def test_the_sample_is_the_right_size(drawn):
    assert len(drawn) == 18000
    assert list(drawn.columns) == [
        "gender",
        "age_band",
        "race",
        "education",
        "income",
        "party",
    ]


def test_the_quota_cells_are_reproduced_exactly(drawn):
    """The three quota axes are integers, so this is equality, not closeness."""
    cells = cell_counts()
    realised = collections.Counter(
        zip(drawn["gender"], drawn["age_band"], drawn["race"])
    )
    assert dict(realised) == {key: value for key, value in cells.items() if value}


def test_the_two_way_quota_margins_are_reproduced_exactly(drawn):
    cells = cell_counts()
    for axes in (("gender", "age_band"), ("gender", "race")):
        realised = collections.Counter(zip(drawn[axes[0]], drawn[axes[1]]))
        expected: collections.Counter = collections.Counter()
        for (gender, band, race), count in cells.items():
            key = {"gender": gender, "age_band": band, "race": race}
            expected[(key[axes[0]], key[axes[1]])] += count
        assert dict(realised) == dict(expected)


def test_the_published_quota_tables_are_matched_to_the_rounding_residue(drawn):
    """The two published tables disagree by one man, so ±1 is the best possible.

    AGE_QUOTA totals 8,827 men and RACE_QUOTA totals 8,828, so no joint
    distribution reproduces both exactly.  What this asserts is that the residue
    is exactly the largest-remainder rounding already inside
    ``pfander.profiles.cell_counts`` and that this module adds nothing to it.
    """
    assert sum(row[2] for row in AGE_QUOTA) + 1 == sum(row[2] for row in RACE_QUOTA)
    age = collections.Counter(zip(drawn["gender"], drawn["age_band"]))
    race = collections.Counter(zip(drawn["gender"], drawn["race"]))
    for band, _, male, female in AGE_QUOTA:
        assert abs(age[("Male", band)] - male) <= 1
        assert abs(age[("Female", band)] - female) <= 1
    for label, _, male, female in RACE_QUOTA:
        assert abs(race[("Male", label)] - male) <= 1
        assert abs(race[("Female", label)] - female) <= 1
    # And the residue is the apportionment's own, not something the draw added.
    apportioned_age: collections.Counter = collections.Counter()
    apportioned_race: collections.Counter = collections.Counter()
    for (gender, band, label), count in cell_counts().items():
        apportioned_age[(gender, band)] += count
        apportioned_race[(gender, label)] += count
    assert age == apportioned_age
    assert race == apportioned_race


def test_the_drawn_marginals_land_on_the_fitted_ones(drawn, model):
    for axis in PFANDER.drawn:
        realised = drawn[axis].value_counts(normalize=True)
        for level in PFANDER.levels(axis):
            assert realised.get(level, 0.0) == pytest.approx(
                float(model.marginal(axis)[level]), abs=0.01
            )


def test_every_moderator_level_clears_the_benchmarks_control_arm_floor(drawn):
    """The benchmark skips a subgroup with fewer than 30 in the 2,000-person control.

    This is the failure the module exists to prevent: the model-generated income
    distribution put 139 of 18,000 respondents in the reference bracket, which was
    18 of the 2,000 control respondents.
    """
    control_n = 2000
    for axis in joint.space(PFANDER).axes:
        for level, share in drawn[axis].value_counts(normalize=True).items():
            expected = share * control_n
            assert expected >= 30, f"{axis}={level}: {expected:.0f} in the control arm"


# --------------------------------------------------------------------------- #
# the two hard constraints, as ``pfander.profiles`` actually builds them
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def prefilled_profiles():
    from silicon_sampling.pfander import profiles

    if not (
        joint.table_path(PFANDER).exists()
        or (_HAS_PYREADSTAT and ccam.CCAM_SAV.exists())
    ):
        pytest.skip("no table to draw from")
    return profiles.build(prefill=True), profiles.build(prefill=False)


def test_prefill_does_not_move_one_respondent_off_their_quota_cell(prefilled_profiles):
    """The hard constraint, at the only place it can actually be broken.

    ``joint`` never touches the quota axes, so this is an equality between two
    18,000-row builds rather than a comparison of two distributions: profile by
    profile, the prefilled build and the old one put the same respondent in the
    same cell, and the published two-way margins therefore come out bit-identical.
    """
    prefilled, plain = prefilled_profiles
    assert len(prefilled) == len(plain) == 18000
    for left, right in zip(prefilled, plain):
        assert left.profile_id == right.profile_id
        assert (left.gender, left.age_band, left.race, left.age, left.year_birth) == (
            right.gender,
            right.age_band,
            right.race,
            right.age,
            right.year_birth,
        )
        assert left.condition == right.condition and left.seed == right.seed
    for axes in (("gender", "age_band"), ("gender", "race")):
        counted = [
            collections.Counter(
                (getattr(p, axes[0]), getattr(p, axes[1])) for p in sample
            )
            for sample in (prefilled, plain)
        ]
        assert counted[0] == counted[1]


def test_the_published_quota_margins_come_out_as_integer_counts(prefilled_profiles):
    """Equality against ``cell_counts``, and the known ±1 against the two tables.

    The published tables cannot both be hit exactly — they disagree by one man —
    so the assertion that carries weight is the first one: the built sample *is*
    the apportionment, cell for cell, with prefill on.
    """
    prefilled, _ = prefilled_profiles
    realised = collections.Counter((p.gender, p.age_band, p.race) for p in prefilled)
    assert dict(realised) == {k: v for k, v in cell_counts().items() if v}
    age = collections.Counter((p.gender, p.age_band) for p in prefilled)
    race = collections.Counter((p.gender, p.race) for p in prefilled)
    for band, _, male, female in AGE_QUOTA:
        assert abs(age[("Male", band)] - male) <= 1
        assert abs(age[("Female", band)] - female) <= 1
    for label, _, male, female in RACE_QUOTA:
        assert abs(race[("Male", label)] - male) <= 1
        assert abs(race[("Female", label)] - female) <= 1


def test_the_reference_income_bracket_stops_being_a_rounding_error(
    prefilled_profiles, model
):
    """``Less than $30,000`` is the income moderator's dummy-coding reference.

    Qwen2.5-7B generated 139 of them in 18,000 — 18 in the 2,000-person control
    arm, against a benchmark floor of 30 — so every income interaction in that run
    was estimated against an empty cell.  CCAM's national share is about 13.5%,
    and the assertion is that the built sample lands there rather than near zero,
    in every arm.
    """
    prefilled, _ = prefilled_profiles
    reference = "Less than $30,000"
    fitted = float(model.marginal("income")[reference])
    assert fitted > 0.12
    count = sum(1 for p in prefilled if p.income == reference)
    assert count / len(prefilled) == pytest.approx(fitted, abs=0.01)
    assert count > 2000
    per_arm = collections.Counter(
        p.condition for p in prefilled if p.income == reference
    )
    assert len(per_arm) == 17
    assert min(per_arm.values()) >= 30
    control = [p for p in prefilled if p.condition == "control"]
    assert sum(1 for p in control if p.income == reference) >= 100


def test_every_prefilled_answer_is_a_legal_answer_to_its_slot(prefilled_profiles):
    """A drawn level that the instrument would reject is worse than no prefill:
    it goes into the transcript unparsed and out again as a scored moderator."""
    from silicon_sampling.survey.slots import ChoiceSlot

    prefilled, _ = prefilled_profiles
    slots = {
        payload.id: payload
        for event, payload in _walk(
            instrument.PRE_CONDITION + instrument.POST_RANDOMISED
        )
        if event == "slot" and isinstance(payload, ChoiceSlot)
    }
    for axis in PFANDER.drawn:
        legal = set(slots[axis].options)
        assert set(PFANDER.levels(axis)) == legal
        drawn_levels = {getattr(p, axis) for p in prefilled}
        assert drawn_levels <= legal
        assert all(slots[axis].parse(level) == level for level in drawn_levels)


def test_no_run_on_disk_has_been_rewritten_by_the_prefilled_build(tmp_path):
    """Every finished run's ``profiles.csv`` is still what the old path writes.

    Both study packages point ``build-profiles`` at a run directory that already
    holds a completed run, so the file is the only record of what those
    respondents were given.  A file here that no longer matches a build this
    module can reproduce means a run's provenance was overwritten, which is not
    recoverable — ``data/`` is not in version control.

    Runs now come in both schemas, so the check dispatches on the header rather
    than demanding one: the ``_demo`` runs are deliberately sampled on prefilled
    profiles, because a model left to invent its own income and party produces
    0.8% of respondents under $30,000 against a real 13.5%.  Either schema is
    legitimate; what is not legitimate is a file matching neither build.
    """
    from silicon_sampling.pfander import profiles as pfander_profiles
    from silicon_sampling.voelkel import profiles as voelkel_profiles

    for module, kwargs, folder in (
        (pfander_profiles, {"prefill": False}, "pfander"),
        (voelkel_profiles, {"demographics": False}, "Voelkel"),
    ):
        directory = ccam.ROOT / "data" / folder / "silicon_sampling"
        found = sorted(directory.glob("*/profiles.csv"))
        if not found:
            continue
        plain = tmp_path / f"{folder}-plain.csv"
        module.write_csv(module.build(**kwargs), plain)
        filled = tmp_path / f"{folder}-filled.csv"
        module.write_csv(module.build(**{k: True for k in kwargs}), filled)
        expected = {
            ",".join(module.BASE_FIELDS): plain.read_bytes(),
            ",".join(module.FIELDS): filled.read_bytes(),
        }
        identical = 0
        for path in found:
            header = path.read_text(encoding="utf-8").splitlines()[0]
            assert header in expected, path
            identical += path.read_bytes() == expected[header]
        # A replicate built at another seed is legitimately different; a run at
        # the default seed is not allowed to be.
        assert identical >= 1, f"{folder}: no run reproduces the default-seed build"
