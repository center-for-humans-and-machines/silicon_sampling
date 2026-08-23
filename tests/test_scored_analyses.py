"""Checks for the scored analyses beyond the pooled row.

Two kinds of check here.  Most assert an *identity* — a saturated interaction
coefficient is a difference in differences, HC2 in a saturated model is the exact
cell-variance formula, a logistic marginal effect on condition dummies is the
difference of two cell proportions — because an identity pins a number down
without a second implementation to compare against.  The rest assert that a
mislabeled input *raises*, since the failure mode this module exists to prevent
is a silently shrunken test set.
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd
import pytest

from silicon_sampling.analysis.ols import aliased_columns, design_matrix
from silicon_sampling.analysis.ols import interaction, level_order, ols
from silicon_sampling.benchmark import scored as SC
from silicon_sampling.benchmark.metrics import pooled_metrics
from silicon_sampling.benchmark.reference import ate_pairs, treatment_effects
from silicon_sampling.pfander.outcomes import MODERATORS, OUTCOMES, SCALE_RANGE

#: The codebook's reference level for each moderator — the first listed level.
#: Five of the six are *not* the alphabetically first one, which is the whole
#: reason ``design_matrix`` grew a ``levels=`` argument.
CODEBOOK_REFERENCE = {
    "gender": "Male",
    "age_band": "18-29",
    "race": "White / Caucasian",
    "education": "Less than high school",
    "income": "Less than $30,000",
    "party": "Republican",
}


def _levels_in_matrix(names: list[str], variable: str) -> list[str]:
    prefix = f"{variable}["
    return [name[len(prefix) : -1] for name in names if name.startswith(prefix)]


def test_design_matrix_omits_the_codebook_reference_for_all_six_moderators():
    for moderator, levels in MODERATORS.items():
        values = list(levels) * 3
        _, names = design_matrix({moderator: values}, levels={moderator: levels})
        present = _levels_in_matrix(names, moderator)
        omitted = [level for level in levels if level not in present]
        assert omitted == [CODEBOOK_REFERENCE[moderator]], moderator
        # The dummies follow the codebook order, not the alphabet.
        assert present == [
            level for level in levels if level != CODEBOOK_REFERENCE[moderator]
        ]


def test_alphabetical_default_picks_the_wrong_reference_for_five_of_six():
    wrong = []
    for moderator, levels in MODERATORS.items():
        _, names = design_matrix({moderator: list(levels)})
        present = _levels_in_matrix(names, moderator)
        omitted = [level for level in levels if level not in present][0]
        if omitted != CODEBOOK_REFERENCE[moderator]:
            wrong.append(moderator)
    assert sorted(wrong) == ["education", "gender", "income", "party", "race"]


def test_a_categoricals_own_order_supplies_the_reference():
    values = pd.Categorical(
        ["Female", "Male", "Other"], categories=["Male", "Female", "Other"]
    )
    assert level_order(values) == ["Male", "Female", "Other"]
    _, names = design_matrix({"gender": values})
    assert _levels_in_matrix(names, "gender") == ["Female", "Other"]


def test_an_explicit_reference_beats_both():
    _, names = design_matrix(
        {"party": list(MODERATORS["party"])},
        reference={"party": "Independent"},
        levels={"party": MODERATORS["party"]},
    )
    assert "Independent" not in _levels_in_matrix(names, "party")


def test_a_reference_level_that_is_not_in_the_data_raises():
    # The failure this prevents: no level is omitted, so every level gets a dummy,
    # the design loses rank and `pinv` answers with unidentified coefficients that
    # look like estimates.  Nothing downstream can tell.
    with pytest.raises(ValueError, match="does not occur in the data"):
        design_matrix({"g": ["a", "b", "c"] * 4}, reference={"g": "zzz"})
    X, names = design_matrix({"g": ["a", "b", "c"] * 4}, reference={"g": "a"})
    assert names == ["(Intercept)", "g[b]", "g[c]"]
    assert np.linalg.matrix_rank(X) == X.shape[1]


def test_a_repeated_level_in_the_intended_order_does_not_duplicate_a_dummy():
    # Callers naturally write [control, *conditions] where conditions already
    # holds the control arm.
    order = level_order(["a", "b", "control"], ["control", "control", "a", "b"])
    assert order == ["control", "a", "b"]


def test_classical_errors_are_the_textbook_ones():
    rng = np.random.default_rng(0)
    x = rng.normal(size=200)
    y = 2 + 3 * x + rng.normal(0, 2, 200)
    X = np.column_stack([np.ones(200), x])
    fit = ols(X, y, ["(Intercept)", "x"], robust="classical")
    xtx_inv = np.linalg.inv(X.T @ X)
    sigma2 = float(fit.resid @ fit.resid) / (200 - 2)
    assert np.allclose(fit.se, np.sqrt(np.diag(xtx_inv * sigma2)))
    # And they are not the robust ones.
    robust = ols(X, y, ["(Intercept)", "x"], robust="HC2")
    assert not np.allclose(fit.se, robust.se)


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #


def _two_arm_frame(seed: int = 3) -> pd.DataFrame:
    """One control and one treatment arm, three gender levels, planned shifts."""
    rng = np.random.default_rng(seed)
    shifts = {"Male": 5.0, "Female": -2.0, "Other": 9.0}
    parts = []
    for condition in ("control", "boost"):
        for level, mean in (("Male", 50.0), ("Female", 60.0), ("Other", 40.0)):
            n = 200 if level != "Other" else 60
            shift = shifts[level] if condition == "boost" else 0.0
            parts.append(
                pd.DataFrame(
                    {
                        "condition": condition,
                        "gender": level,
                        "y": rng.normal(mean + shift, 12.0, n),
                    }
                )
            )
    return pd.concat(parts, ignore_index=True)


def _two_arm_design() -> SC.ScoredDesign:
    return SC.ScoredDesign(
        outcomes={"y": 100.0},
        control="control",
        moderators={"gender": MODERATORS["gender"]},
        conditions=("control", "boost"),
    )


def _grid_frame(
    n_per_arm: int = 60, seed: int = 5, conditions: int = 17
) -> pd.DataFrame:
    """A Pfänder-shaped frame: 17 arms, 13 outcomes, six moderators."""
    rng = np.random.default_rng(seed)
    arms = ["control"] + [f"Intervention {index}" for index in range(1, conditions)]
    rows = []
    for arm in arms:
        for _ in range(n_per_arm):
            row = {"condition": arm}
            for moderator, levels in MODERATORS.items():
                row[moderator] = levels[rng.integers(len(levels))]
            for outcome in OUTCOMES:
                scale = SCALE_RANGE[outcome]
                if outcome == "newsletter_signup":
                    row[outcome] = float(rng.random() < 0.3)
                else:
                    row[outcome] = float(
                        np.clip(rng.normal(0.5 * scale, 0.2 * scale), 0, scale)
                    )
            rows.append(row)
    return pd.DataFrame(rows)


def _grid_design(**overrides) -> SC.ScoredDesign:
    arms = ["control"] + [f"Intervention {index}" for index in range(1, 17)]
    return SC.ScoredDesign(
        outcomes=dict(SCALE_RANGE),
        control="control",
        moderators=MODERATORS,
        conditions=arms,
        binary=("newsletter_signup",),
        **overrides,
    )


# --------------------------------------------------------------------------- #
# Section 2 — interaction estimates
# --------------------------------------------------------------------------- #


def test_interaction_coefficient_is_the_difference_in_differences():
    frame = _two_arm_frame()
    table = SC.run_moderator_model(frame, "y", "gender", _two_arm_design())
    cells = frame.groupby(["condition", "gender"], observed=True)["y"].agg(
        ["mean", "var", "count"]
    )

    def effect(level: str) -> float:
        return (
            cells.loc[("boost", level), "mean"] - cells.loc[("control", level), "mean"]
        )

    def hc2(level: str) -> float:
        keys = [
            ("boost", level),
            ("control", level),
            ("boost", "Male"),
            ("control", "Male"),
        ]
        return float(
            np.sqrt(
                sum(cells.loc[key, "var"] / cells.loc[key, "count"] for key in keys)
            )
        )

    indexed = table.set_index("moderator_level")
    for level in ("Female", "Other"):
        assert np.isclose(
            indexed.loc[level, "estimate"], effect(level) - effect("Male")
        )
        # HC2 in a saturated model is exactly the cell-variance formula.
        assert np.isclose(indexed.loc[level, "se"], hc2(level))
    assert set(indexed["reference_level"]) == {"Male"}


def test_the_reference_level_decides_which_estimand_is_scored():
    frame = _two_arm_frame()
    codebook = SC.run_moderator_model(frame, "y", "gender", _two_arm_design())
    alphabetical = SC.run_moderator_model(
        frame,
        "y",
        "gender",
        SC.ScoredDesign(
            outcomes={"y": 100.0}, control="control", conditions=("control", "boost")
        ),
    )
    assert set(codebook["moderator_level"]) == {"Female", "Other"}
    assert set(alphabetical["moderator_level"]) == {"Male", "Other"}
    assert set(codebook["reference_level"]) == {"Male"}
    assert set(alphabetical["reference_level"]) == {"Female"}


@pytest.mark.parametrize("empty_arm", ["boost", "control"])
def test_an_empty_cell_leaves_its_interaction_term_missing_not_wrong(empty_arm):
    # Both shapes of an empty cell.  With the *treatment* cell empty the
    # interaction column is all zeros and is easy to spot.  With the *control* cell
    # empty nothing is zero: `gender[Other]` and `condition[boost]:gender[Other]`
    # become the same column, and the minimum-norm solution splits the effect
    # between them and reports each half with a small standard error — an
    # unidentified quantity presented as a confident estimate, and the one that is
    # plausible at Pfänder scale for `gender[Other]` in a 1,000-person control arm.
    # R answers NA for the aliased coefficient and `broom::tidy` drops the row.
    frame = _two_arm_frame()
    frame = frame[~((frame["condition"] == empty_arm) & (frame["gender"] == "Other"))]
    table = SC.run_moderator_model(frame, "y", "gender", _two_arm_design())
    missing = table[table["moderator_level"] == "Other"]
    assert len(missing) == 1
    for column in ("estimate", "se", "conf_low", "conf_high", "p"):
        assert missing[column].isna().all(), column

    # The rest of the fit is untouched by the aliased column: the surviving design
    # is saturated on the cells that do exist, so the Female interaction is still
    # exactly the difference in differences of the four Male/Female cell means.
    cells = frame.groupby(["condition", "gender"], observed=True)["y"].mean()
    did = (cells[("boost", "Female")] - cells[("control", "Female")]) - (
        cells[("boost", "Male")] - cells[("control", "Male")]
    )
    assert np.isclose(table.set_index("moderator_level").loc["Female", "estimate"], did)
    intact = SC.run_moderator_model(
        frame[frame["gender"] != "Other"], "y", "gender", _two_arm_design()
    )
    assert np.isclose(
        table.set_index("moderator_level").loc["Female", "estimate"],
        intact.set_index("moderator_level").loc["Female", "estimate"],
    )


def test_aliasing_drops_the_interaction_and_keeps_the_main_effect():
    # Which of an aliased pair goes missing is part of the answer, and R decides it
    # by position: the main effect is fitted, the interaction is NA.  Keeping the
    # earlier column also keeps the fit saturated, so `gender[Other]` has to come
    # out as the plain gap between the two arms that do have Other respondents.
    frame = _two_arm_frame()
    frame = frame[~((frame["condition"] == "control") & (frame["gender"] == "Other"))]
    data = frame.reset_index(drop=True)
    cond_X, cond_names = design_matrix(
        {"condition": data["condition"]},
        reference={"condition": "control"},
        levels={"condition": ["control", "boost"]},
    )
    mod_X, mod_names = design_matrix(
        {"gender": data["gender"]}, levels={"gender": MODERATORS["gender"]}
    )
    inter_X, inter_names = interaction(cond_X, cond_names, mod_X, mod_names)
    X = np.hstack([cond_X, mod_X[:, 1:], inter_X])
    names = cond_names + mod_names[1:] + inter_names
    assert np.linalg.matrix_rank(X) == X.shape[1] - 1
    assert [names[index] for index in aliased_columns(X)] == [
        "condition[boost]:gender[Other]"
    ]
    kept = [index for index in range(X.shape[1]) if index not in aliased_columns(X)]
    fit = ols(X[:, kept], data["y"].to_numpy(float), [names[i] for i in kept])
    cells = data.groupby(["condition", "gender"], observed=True)["y"].mean()
    assert np.isclose(
        fit.term("gender[Other]")["estimate"],
        cells[("boost", "Other")] - cells[("boost", "Male")],
    )


def test_missing_values_are_not_a_scored_factor_level():
    # Item nonresponse in a demographic column is normal in human data.  Coerced
    # with `astype(str)` it becomes a group called "nan" that survives `dropna()`,
    # earns a dummy, and is reported as a scored subgroup — and a missing condition
    # becomes a study arm nobody was assigned to.  R keeps NA as NA and drops those
    # rows at fit time.
    frame = _two_arm_frame()
    frame.loc[frame.index[:40], "gender"] = np.nan
    frame.loc[frame.index[-5:], "condition"] = np.nan
    design = _two_arm_design()

    aligned = SC.align_submission_levels(frame, design)
    assert "nan" not in set(aligned["gender"].cat.categories)
    assert "nan" not in set(aligned["condition"].cat.categories)
    assert aligned["gender"].isna().sum() == 40

    table = SC.run_moderator_model(frame, "y", "gender", design)
    assert set(table["moderator_level"]) == {"Female", "Other"}
    effects = SC.ate_side(frame, design)
    assert set(effects["condition"]) == {"boost"}
    # And the fit is the one R reports: the rows with a missing moderator are gone,
    # not silently pooled into a level of their own.
    complete = SC.run_moderator_model(
        frame.dropna(subset=["gender"]), "y", "gender", design
    )
    assert np.allclose(table["estimate"], complete["estimate"])

    # The demographic tables have no "nan" cell either.
    human, llm, party_design = _demographic_frames()
    llm = llm.copy()
    llm.loc[llm.index[:50], "party"] = np.nan
    baselines = SC.compare_demographic_baselines(
        SC.align_submission_levels(human, party_design),
        SC.align_submission_levels(llm, party_design),
        "y",
        party_design,
    )
    assert baselines.iloc[0]["n_cells"] == 4

    # Same for a moderator whose levels the design does not list, where the level
    # order comes from the data itself.
    unlisted = SC.ScoredDesign(
        outcomes={"y": 100.0}, control="control", moderators={"party": ()}
    )
    parity = SC.demographic_parity_gap(human, llm, "y", unlisted, min_n=5)
    assert "nan" not in str(parity.iloc[0]["groups_skipped"])
    shape = SC.subgroup_distributions(human, llm, unlisted, min_n=5)
    assert "nan" not in set(shape["level"])


def test_subgroup_pairs_are_in_pp_of_scale_range():
    frame = _two_arm_frame()
    donation = frame.assign(donation_ams=frame["y"] / 10.0)
    design = SC.ScoredDesign(
        outcomes={"donation_ams": 10.0},
        control="control",
        moderators={"gender": MODERATORS["gender"]},
        conditions=("control", "boost"),
    )
    human_side = SC.subgroup_side(donation, design)
    pairs = SC.build_subgroup_pairs(human_side, donation, design)
    # A dollar effect on a 0-10 scale is ten pp per dollar.
    assert np.allclose(pairs["estimate_h"], human_side["estimate"] * 10)
    assert np.allclose(pairs["estimate_h"], pairs["estimate_l"])
    metrics = SC.subgroup_metrics(pairs)
    assert "rmse" not in metrics
    assert set(SC.SUBGROUP_METRICS) <= set(metrics)


def test_subgroup_pairs_raise_when_a_level_fails_to_join():
    frame = _two_arm_frame()
    design = _two_arm_design()
    human_side = SC.subgroup_side(frame, design)
    broken = frame.replace({"gender": {"Other": "other"}})
    with pytest.raises(ValueError, match="moderator level"):
        SC.build_subgroup_pairs(human_side, broken, design)


# --------------------------------------------------------------------------- #
# Section 1 — grid, breakdowns, binary path
# --------------------------------------------------------------------------- #


def test_the_full_grid_is_208_pairs_and_a_mislabeled_condition_raises():
    frame = _grid_frame()
    design = _grid_design()
    assert design.expected_pairs == 208
    side = SC.ate_side(frame, design)
    pairs = SC.build_ate_pairs(side, frame, design)
    assert len(pairs) == 208
    assert design.continuous_outcomes == tuple(OUTCOMES[:11])
    assert design.behavioral_outcomes == ("donation_ams", "newsletter_signup")

    mislabeled = frame.replace({"condition": {"Intervention 3": "intervention 3"}})
    with pytest.raises(ValueError, match="expected 208 rows"):
        SC.build_ate_pairs(side, mislabeled, design)


def test_a_mislabeled_control_arm_halts_instead_of_scoring_garbage():
    # The count check cannot see this one.  Renaming the control arm promotes it to
    # a seventeenth treatment dummy, the inner join against the human reference
    # drops that extra row, and exactly 208 pairs come back — scored against a
    # baseline that is now one of the interventions.  The reference level has to be
    # checked where it is used, so the fit is what halts.
    frame = _grid_frame(n_per_arm=25, seed=11)
    design = _grid_design()
    side = SC.ate_side(frame, design)
    mislabeled = frame.replace({"condition": {"control": "Control"}})
    with pytest.raises(ValueError, match="does not occur in the data") as raised:
        SC.build_ate_pairs(side, mislabeled, design)
    assert "expected 208 rows" not in str(raised.value)
    # A mislabeled moderator level is the same failure one model deeper.
    with pytest.raises(ValueError, match="does not occur in the data"):
        SC.subgroup_side(mislabeled, design, outcomes=("trust_post",))


def test_the_grid_check_is_never_a_no_op():
    # A design that cannot state its own pair count used to skip the count check
    # entirely, so a submission missing two arms scored on the arms it kept.  The
    # human reference is the grid whenever the design does not say.
    frame = _grid_frame(n_per_arm=25, seed=12)
    silent = SC.ScoredDesign(
        outcomes={"trust_post": 100.0, "belief_post": 100.0}, control="control"
    )
    assert silent.expected_pairs is None
    side = SC.ate_side(frame, silent)
    assert len(SC.build_ate_pairs(side, frame, silent)) == len(side)
    short = frame[~frame["condition"].isin(["Intervention 1", "Intervention 2"])]
    with pytest.raises(ValueError, match="expected 32 rows, got 28"):
        SC.build_ate_pairs(side, short, silent)


def test_missing_estimates_raise_even_when_the_count_is_right():
    frame = _two_arm_frame()
    design = _two_arm_design()
    side = SC.ate_side(frame, design)
    pairs = ate_pairs(side, side)
    pairs.loc[0, "estimate_l"] = np.nan
    with pytest.raises(ValueError, match="missing values"):
        SC.assert_full_grid(pairs, len(pairs))


def test_binary_marginal_effects_match_the_saturated_linear_model():
    rng = np.random.default_rng(7)
    parts = [
        pd.DataFrame(
            {
                "condition": condition,
                "newsletter_signup": (rng.random(800) < share).astype(float),
            }
        )
        for condition, share in (("control", 0.2), ("boost", 0.35), ("nudge", 0.22))
    ]
    frame = pd.concat(parts, ignore_index=True)
    design = SC.ScoredDesign(
        outcomes={"newsletter_signup": 1.0},
        control="control",
        binary=("newsletter_signup",),
        conditions=("control", "boost", "nudge"),
    )
    logit = SC.binary_marginal_effects(frame, "newsletter_signup", design).set_index(
        "condition"
    )
    lpm = treatment_effects(
        frame, {"newsletter_signup": 1.0}, "control", robust="HC2"
    ).set_index("condition")
    shares = frame.groupby("condition")["newsletter_signup"].mean()

    assert set(logit["model"]) == {"logit"}
    # The benchmark's binary model adjusts its p-values too, so these rows must not
    # arrive at the leaderboard with an empty p_bh column.
    assert logit["p_bh"].notna().all()
    assert (logit["p_bh"] >= logit["p"]).all()
    for condition in ("boost", "nudge"):
        # The marginal effect is the difference of two fitted cell proportions.
        assert np.isclose(
            logit.loc[condition, "estimate"],
            (shares[condition] - shares["control"]) * 100,
        )
        # In this saturated, covariate-free specification the delta-method HC2
        # error coincides with the linear model's, so even se_l is unaffected.
        assert np.isclose(logit.loc[condition, "se"], lpm.loc[condition, "se"])


def test_a_cell_with_no_signups_falls_back_to_the_linear_model_visibly():
    rng = np.random.default_rng(9)
    frame = pd.DataFrame(
        {
            "condition": np.repeat(["control", "boost"], 300),
            "newsletter_signup": np.concatenate(
                [(rng.random(300) < 0.3).astype(float), np.zeros(300)]
            ),
        }
    )
    design = SC.ScoredDesign(
        outcomes={"newsletter_signup": 1.0},
        control="control",
        binary=("newsletter_signup",),
        conditions=("control", "boost"),
    )
    table = SC.binary_marginal_effects(frame, "newsletter_signup", design)
    assert set(table["model"]) == {"lpm"}
    assert np.isfinite(table["estimate"]).all()


def test_metrics_by_group_scores_each_level_on_its_own_rows():
    frame = _grid_frame(n_per_arm=40, seed=8)
    design = _grid_design()
    side = SC.ate_side(frame, design)
    pairs = SC.build_ate_pairs(side, frame, design)
    per_outcome = SC.metrics_by_group(pairs, "outcome")
    assert len(per_outcome) == 13
    assert (per_outcome["n_pairs"] == 16).all()
    one = pooled_metrics(pairs[pairs["outcome"] == "trust_post"], include_rmse=True)
    row = per_outcome[per_outcome["outcome"] == "trust_post"].iloc[0]
    assert np.isclose(row["rmse"], one["rmse"])

    cuts = SC.breakdowns(pairs, design)
    # The per-intervention cut drops the two behavioral outcomes.
    assert (cuts["by_intervention"]["n_pairs"] == 11).all()
    assert set(cuts["by_class"]["outcome_class"]) == {"Self-report", "Behavioral"}


# --------------------------------------------------------------------------- #
# Section 3 — distributions and demographics
# --------------------------------------------------------------------------- #


def _demographic_frames() -> tuple[pd.DataFrame, pd.DataFrame, SC.ScoredDesign]:
    rng = np.random.default_rng(13)
    levels = ("Republican", "Democrat", "Independent", "Other")
    sizes = {"Republican": 400, "Democrat": 400, "Independent": 200, "Other": 10}
    human_means = {
        "Republican": 40.0,
        "Democrat": 60.0,
        "Independent": 50.0,
        "Other": 55.0,
    }
    offsets = {"Republican": 6.0, "Democrat": 1.0, "Independent": -3.0, "Other": 20.0}
    human, llm = [], []
    for level in levels:
        n = sizes[level]
        human.append(
            pd.DataFrame(
                {
                    "condition": "control",
                    "party": level,
                    "y": human_means[level] + rng.normal(0, 15, n),
                }
            )
        )
        llm.append(
            pd.DataFrame(
                {
                    "condition": "control",
                    "party": level,
                    "y": human_means[level] + offsets[level] + rng.normal(0, 5, n),
                }
            )
        )
    design = SC.ScoredDesign(
        outcomes={"y": 100.0},
        control="control",
        moderators={"party": levels},
        conditions=("control",),
    )
    return (
        pd.concat(human, ignore_index=True),
        pd.concat(llm, ignore_index=True),
        design,
    )


def test_baselines_are_the_rmse_of_raw_control_cell_means():
    human, llm, design = _demographic_frames()
    table = SC.compare_demographic_baselines(human, llm, "y", design)
    assert len(table) == 1
    row = table.iloc[0]
    left = human.groupby("party")["y"].mean()
    right = llm.groupby("party")["y"].mean()
    assert np.isclose(row["rmse"], float(np.sqrt(np.mean((left - right) ** 2))))
    assert np.isclose(row["r"], float(left.corr(right)))
    assert row["n_cells"] == 4
    # Raw points, not pp: a 6-point miss reads as 6, whatever the scale range.
    assert 2 < row["rmse"] < 20


def test_parity_gap_is_the_spread_of_group_errors_and_names_skipped_groups():
    human, llm, design = _demographic_frames()
    table = SC.demographic_parity_gap(human, llm, "y", design)
    row = table.iloc[0]
    errors = (
        (llm.groupby("party")["y"].mean() - human.groupby("party")["y"].mean())
        .abs()
        .drop("Other")  # 10 respondents: under min_n on both sides
    )
    assert np.isclose(row["dpd"], errors.max() - errors.min())
    assert np.isclose(row["worst_abs_err"], errors.max())
    assert row["worst_group"] == errors.idxmax()
    assert row["best_group"] == errors.idxmin()
    assert row["n_skipped"] == 1 and row["groups_skipped"] == "Other"
    assert "human 10" in row["groups_skipped_detail"]
    assert row["n_groups"] == 3


def test_parity_gap_is_not_the_within_sample_spread_of_group_means():
    # The distinction that matters: a sample can place every group at the right
    # level and still be scored badly here, and vice versa.
    human, llm, design = _demographic_frames()
    gap = SC.demographic_parity_gap(human, llm, "y", design).iloc[0]["dpd"]
    within = float(
        llm.groupby("party")["y"].mean().max() - llm.groupby("party")["y"].mean().min()
    )
    assert not np.isclose(gap, within)


def test_stereotyping_returns_both_r_squared_and_coefficients_with_intervals():
    human, llm, design = _demographic_frames()
    out = SC.compare_demographic_predictability(human, llm, "y", design)
    r2 = out["r_squared"].iloc[0]
    coefficients = out["coefficients"]
    assert r2["reference_level"] == "Republican"
    # The synthetic side has bigger gaps and less noise, so it must look more
    # predictable: that is the diagnostic firing.
    assert r2["r_squared_l"] > r2["r_squared_h"]
    assert np.isclose(r2["r_squared_gap"], r2["r_squared_l"] - r2["r_squared_h"])
    assert set(coefficients["level"]) == {"Democrat", "Independent", "Other"}
    assert (coefficients["lo_h"] < coefficients["est_h"]).all()
    assert (coefficients["est_h"] < coefficients["hi_h"]).all()
    # One condition only, so the coefficient is the plain group-mean difference.
    means = human.groupby("party")["y"].mean()
    democrat = coefficients.set_index("level").loc["Democrat", "est_h"]
    assert np.isclose(democrat, means["Democrat"] - means["Republican"])


def test_subgroup_distributions_skip_small_groups_but_report_them():
    human, llm, design = _demographic_frames()
    table = SC.subgroup_distributions(human, llm, design)
    assert len(table) == 4
    skipped = table[table["skipped"]]
    assert list(skipped["level"]) == ["Other"]
    assert skipped["n_human"].iloc[0] == 10
    assert skipped["variance_ratio"].isna().all()
    scored = table[~table["skipped"]]
    # The synthetic side is deliberately under-dispersed (sd 5 against 15).
    assert (scored["variance_ratio"] < 0.5).all()
    assert scored["ovl"].between(0, 1).all()


def test_summarise_field_excludes_the_reference_row():
    field = pd.DataFrame(
        {
            "submission": ["a", "b", "Human replication"],
            "metric": ["pearson_r"] * 3,
            "value": [0.2, 0.4, 0.9],
        }
    )
    out = SC.summarise_field(field)
    assert list(out.columns) == ["metric", "mean", "median", "sd", "min", "max"]
    assert len(out) == 1
    assert np.isclose(out.iloc[0]["mean"], 0.3)
    assert np.isclose(out.iloc[0]["max"], 0.4)


# --------------------------------------------------------------------------- #
# the assembler
# --------------------------------------------------------------------------- #


def test_align_submission_levels_puts_the_control_arm_first():
    frame = _grid_frame(n_per_arm=5)
    aligned = SC.align_submission_levels(frame, _grid_design())
    assert list(aligned["condition"].cat.categories)[0] == "control"
    for moderator, levels in MODERATORS.items():
        assert (
            list(aligned[moderator].cat.categories)[0] == CODEBOOK_REFERENCE[moderator]
        )


def test_leaderboard_row_is_flat_and_covers_every_scored_analysis():
    frame = _grid_frame(n_per_arm=60, seed=17)
    # min_group_n=5 keeps the six-level moderators scorable at this test size; the
    # real design carries 1,000 control respondents and uses the preregistered 30.
    design = _grid_design(
        subgroup_outcomes=("trust_multidimensional", "donation_ams"), min_group_n=5
    )
    row = SC.leaderboard_row(frame, frame, design, label="self")

    assert all(not isinstance(value, (list, dict, tuple)) for value in row.values())
    assert row["submission"] == "self" and row["n_pairs"] == 208
    # Scored against itself: every effect is recovered exactly.
    assert np.isclose(row["pearson_r"], 1.0) and np.isclose(row["beta"], 1.0)
    assert np.isclose(row["rmse"], 0.0)
    assert np.isclose(row["subgroup/pooled/pearson_r"], 1.0)

    prefixes = [
        "outcome/trust_post/pearson_r",
        "intervention/Intervention 1/rmse",
        "class/Behavioral/pearson_r",
        "subgroup/party/pearson_r",
        "shape/trust_post/variance_ratio",
        "shape/party/variance_ratio_median",
        "baseline/trust_multidimensional/party/rmse",
        "parity/trust_multidimensional/party/dpd",
        "stereo/trust_multidimensional/party/r2_h",
        "stereo/trust_multidimensional/party/Democrat/est_h",
        "median/outcome/pearson_r",
        "median/baseline/rmse",
        "median/parity/dpd",
        "median/stereo/coef_rmse",
    ]
    missing = [key for key in prefixes if key not in row]
    assert not missing, missing
    # Self-scoring means perfect demographic recovery too.
    assert np.isclose(row["baseline/trust_multidimensional/party/rmse"], 0.0)
    assert np.isclose(row["parity/trust_multidimensional/party/dpd"], 0.0)
    assert np.isclose(row["shape/trust_post/variance_ratio"], 1.0)
    assert np.isclose(row["stereo/trust_multidimensional/party/r2_gap"], 0.0)


def test_asking_for_subgroups_with_a_reference_that_has_none_raises():
    # Section 2 is fifty-odd keys of the flat row.  Dropping it because the passed
    # reference was fitted without interactions used to be silent, and the shorter
    # row reads exactly like a submission that was never scored on subgroups.
    frame = _grid_frame(n_per_arm=20, seed=21)
    design = _grid_design(subgroup_outcomes=("trust_post",))
    without = SC.reference_sides(frame, design, subgroup=False)
    with pytest.raises(ValueError, match="no interaction"):
        SC.score_submission(frame, frame, design, sides=without, subgroups=True)
    # Deliberately scoring without Section 2 is still allowed, and says so.
    row = SC.leaderboard_row(frame, frame, design, sides=without, subgroups=False)
    assert not any(key.startswith("subgroup/") for key in row)
    # A study with no moderators at all has no Section 2 to drop.
    plain = SC.ScoredDesign(
        outcomes={"trust_post": 100.0},
        control="control",
        conditions=_grid_design().conditions,
    )
    assert "subgroup/pooled/pearson_r" not in SC.leaderboard_row(frame, frame, plain)


def test_the_flat_row_agrees_with_the_pooled_metrics_it_reports():
    frame = _grid_frame(n_per_arm=30, seed=19)
    other = _grid_frame(n_per_arm=30, seed=20)
    design = _grid_design(subgroup_outcomes=("trust_post",))
    report = SC.score_submission(frame, other, design, label="other", subgroups=True)
    direct = pooled_metrics(
        ate_pairs(SC.ate_side(frame, design), SC.ate_side(other, design)),
        include_rmse=True,
    )
    row = report.row()
    for metric in SC.POOLED_METRICS:
        assert np.isclose(row[metric], direct[metric], equal_nan=True), metric


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
