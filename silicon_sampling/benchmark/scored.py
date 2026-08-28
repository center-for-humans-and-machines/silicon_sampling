"""The scored analyses the leaderboard column does not show.

``metrics.py`` covers the pooled ATE row — the seven estimate metrics plus the
calibration line — because that is what the benchmark sorts its table on.  It is
not what the benchmark *scores*.  The preregistration also scores subgroup
recovery from condition x moderator interactions, per-outcome and
per-intervention breakdowns, response-distribution shape overall and within
demographic groups, demographic baseline calibration, the demographic parity gap,
and the stereotyping regressions.  A calibration choice that lifts the pooled
correlation while inflating every group gap is a worse submission, not a better
one, and there is no way to notice that without computing all of it.  So this
module closes the gap: everything the benchmark scores, from respondent-level
frames, in one place.

**Three unit conventions live here and they are not interchangeable.** The pooled
and subgroup metrics run on estimates converted to *percentage points of each
outcome's scale range* (``x 100 / scale_range``), applied once at pair-building,
because pooling raw units would let the 0-100 sliders drown out a dollar
donation.  The distribution metrics run on *raw responses* over a fixed grid
(the outcome's full scale), so that every submission is scored on the identical
grid rather than on its own observed range.  The demographic analyses —
baselines, parity gap, stereotyping coefficients — run on *raw outcome points*,
because a group mean is a level, not an effect, and the benchmark reports it as
one.  Mixing them produces numbers that look sane and are wrong by a factor of
ten.

**Why a design object.** Which outcomes are continuous, which are behavioral,
what the control arm is called, the codebook order of every moderator, and how
many pairs a complete grid has are all study facts, not scoring facts.  They are
gathered into :class:`ScoredDesign` so the scoring code stays study-independent
and so the one fact that silently corrupts everything — the moderator reference
level — is stated once, in the caller's own words, instead of being inferred
alphabetically at each of seventy-eight regression fits.

**Why the grid assertion raises.** Every pair builder joins the submission
against the human reference, and an inner join drops what it cannot match.  A
single mislabeled condition would quietly shrink a submission's test set and its
scores would then be computed on the surviving subset — flattering, and
undetectable in the output.  The benchmark's own pipeline halts there
(``stopifnot``), so :func:`assert_full_grid` raises rather than warns.  Counting
rows catches a mislabeled *treatment* arm; a mislabeled *control* arm never
reaches the count, because it makes the reference level itself missing, and
:func:`silicon_sampling.analysis.ols.design_matrix` halts on that instead.  The
human reference defines the expected count whenever the design cannot state one,
so there is no configuration under which the check quietly does nothing.

**Why a missing coefficient stays missing.** Two things here have no identified
answer and are reported as ``NaN`` rather than as a number: an interaction whose
cell is empty (:func:`aliased_columns` finds it, exactly as R's ``lm`` reports
``NA`` and ``broom::tidy`` drops it) and a moderator value that is missing in the
first place.  R's ``factor()`` keeps ``NA`` as ``NA`` and ``lm`` drops those rows,
so :func:`align_submission_levels` does too — coercing a column with ``astype(str)``
would turn item nonresponse into a demographic group called ``"nan"``, which then
gets scored, correlated and reported like any other.

**Why the binary outcome gets a logistic fit.** ``newsletter_signup`` is scored
through logistic regression and average marginal effects on the probability
scale, not a linear probability model.  Implementing it turned out to settle a
question the plan had left open: in this saturated, covariate-free specification
the two paths agree on the point estimate *and* on the HC2 standard error to
machine precision, not merely on the estimate.  Both fit the cell proportions
exactly, and the delta-method variance of ``p_k - p_0`` collapses to
``p(1-p)/(n-1)`` per cell — which is what HC2 on a saturated linear model already
gives.  So nothing downstream differs, ``se_l`` and ``beta_adj`` included.  The
logistic path is kept because the equality is a fact about *this* specification
(add a covariate and it goes), and because the LPM fallback — taken when a cell
has no signups at all and the logit coefficients diverge — is then a flagged,
visible event rather than the silent default.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from ..analysis.ols import aliased_columns, benjamini_hochberg, design_matrix
from ..analysis.ols import interaction, level_order, ols
from . import distributions as dist
from .metrics import cluster_bootstrap, pooled_metrics, run_calibration_pooled
from .metrics import signed_metrics
from .reference import ate_pairs, treatment_effects

#: The metrics a pooled ATE row carries, in the benchmark's least-to-most-strict
#: order.  Named so the bootstrap and the flat row agree on what to look for.
POOLED_METRICS = (
    "directional_pct",
    "spearman_rho",
    "pearson_r",
    "pearson_within",
    "pearson_adj",
    "rmse",
    "rmse_adj",
)

#: What the per-outcome / per-intervention / per-class breakdowns report.  The
#: within-outcome correlation is left out because it is degenerate inside these
#: cuts: within one outcome it equals the plain Pearson r, and within one
#: intervention every outcome contributes a single pair, so centring annihilates
#: the whole column.  The benchmark's companion tables report exactly this set.
BREAKDOWN_METRICS = tuple(
    metric for metric in POOLED_METRICS if metric != "pearson_within"
)

#: The subgroup section's metrics: the same ladder without RMSE.  Interaction
#: estimates come from small demographic cells, so the absolute distance between
#: two of them is mostly noise in the human estimate.
SUBGROUP_METRICS = ("directional_pct", "spearman_rho", "pearson_r", "pearson_adj")

#: The four distribution-shape metrics, with the variance ratio first because it
#: is the section's headline diagnostic (under-dispersion is the failure mode the
#: literature reports most often).
SHAPE_METRICS = ("variance_ratio", "ovl", "ks", "w1")


@dataclass(frozen=True)
class ScoredDesign:
    """The study facts every scored analysis needs, stated once.

    ``outcomes`` maps every pooled outcome to its scale range; ``moderators`` maps
    each moderator to its **codebook level order**, whose first entry is the
    reference level of every interaction and stereotyping coefficient.
    ``conditions`` is the full arm list with the control first, and it is what
    makes the expected pair count derivable — and therefore the grid assertion
    meaningful.
    """

    outcomes: Mapping[str, float]
    control: str
    moderators: Mapping[str, Sequence[str]] = field(default_factory=dict)
    conditions: Sequence[str] = ()
    binary: Sequence[str] = ()
    continuous: Sequence[str] = ()
    subgroup_outcomes: Sequence[str] = ()
    baseline_outcomes: Sequence[str] = ()
    condition_col: str = "condition"
    min_group_n: int = 30
    n_pairs_expected: int | None = None

    @property
    def continuous_outcomes(self) -> tuple[str, ...]:
        """Outcomes sharing one 0-100 scale: the per-intervention and shape set.

        Defaulted rather than required: an outcome that is neither binary nor
        rescaled is a slider on the common scale, which is exactly the
        benchmark's ``outcomes_continuous``.
        """
        if self.continuous:
            return tuple(self.continuous)
        return tuple(
            name
            for name, scale in self.outcomes.items()
            if name not in set(self.binary) and float(scale) == 100.0
        )

    @property
    def behavioral_outcomes(self) -> tuple[str, ...]:
        """The complement of :attr:`continuous_outcomes`: the class cut's other half."""
        continuous = set(self.continuous_outcomes)
        return tuple(name for name in self.outcomes if name not in continuous)

    @property
    def primary(self) -> str:
        """The first outcome, which the preregistration treats as primary."""
        return next(iter(self.outcomes))

    @property
    def subgroup_set(self) -> tuple[str, ...]:
        """Outcomes entering the subgroup analysis (default: all of them)."""
        return tuple(self.subgroup_outcomes or self.outcomes)

    @property
    def baseline_set(self) -> tuple[str, ...]:
        """Outcomes entering the demographic analyses (default: the primary one)."""
        return tuple(self.baseline_outcomes or (self.primary,))

    def outcome_class(self, outcome: str) -> str:
        """``Self-report`` or ``Behavioral`` — the secondary class cut."""
        return (
            "Self-report" if outcome in set(self.continuous_outcomes) else "Behavioral"
        )

    @property
    def expected_pairs(self) -> int | None:
        """How many condition x outcome pairs a complete grid has, if knowable."""
        if self.n_pairs_expected is not None:
            return int(self.n_pairs_expected)
        if self.conditions:
            # Tolerant of either convention for `conditions` — with the control
            # arm listed or without it — because getting 195 instead of 208 here
            # would turn the grid assertion into a false alarm on every entry.
            arms = {str(name) for name in self.conditions} | {str(self.control)}
            return (len(arms) - 1) * len(self.outcomes)
        return None

    def scale(self, outcome: str) -> float:
        return float(self.outcomes[outcome])


def _median(values) -> float:
    """Median that answers NaN for an all-missing column instead of warning."""
    numeric = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    return float(numeric.median()) if len(numeric) else float("nan")


def _labels(values: pd.Series) -> pd.Series:
    """Level labels as strings, with missing values left missing.

    ``astype(str)`` alone turns ``NaN`` into the string ``"nan"``, which then
    behaves like a real level everywhere downstream: it survives ``dropna()``, it
    earns its own dummy column, and it lands in the output as a demographic group
    or a study arm that does not exist.  R's ``factor()`` leaves ``NA`` alone and
    every model drops those rows, so the labels are built to allow that.
    """
    text = values.astype(str)
    # A hole that has already been through ``astype(str)`` upstream arrives as the
    # *string* ``"nan"`` and would otherwise survive as a level: ICPC's human
    # ``age_band`` carried 44 interaction estimates for a group called ``"nan"``.
    # No real codebook level spells itself this way, so they are holes here too.
    stringified = text.isin({"nan", "NaN", "None", "<NA>", ""})
    return text.where(values.notna() & ~stringified)


def _numeric(values: pd.Series) -> pd.Series:
    """Outcome values as floats, unparseable entries dropped to ``NaN``.

    Used everywhere an outcome column is read, so that a stray ``"N/A"`` in one
    submission cannot make one analysis raise and another silently disagree.
    """
    return pd.to_numeric(values, errors="coerce")


def align_submission_levels(frame: pd.DataFrame, design: ScoredDesign) -> pd.DataFrame:
    """Give every factor the human reference's level order, before any fit.

    Submitted CSVs arrive as character columns, and every estimation function here
    takes the first level as the reference.  Alphabetical coercion picks the wrong
    one almost everywhere — capitalised intervention titles sort before
    ``control``, ``Bachelor's degree`` before ``Less than high school`` — so the
    order is set from the codebook first and the fits inherit it.

    A missing value is not a level.  Item nonresponse in a demographic column is
    normal in human data, and ``astype(str)`` would promote it to a group named
    ``"nan"`` that then gets a dummy, a scored interaction estimate and a row in
    every demographic table.  R keeps ``NA`` as ``NA`` and drops those rows at fit
    time; the factors built here carry the same hole, so ``dropna()`` inside each
    estimator does the dropping.
    """
    out = frame.copy()
    column = design.condition_col
    if column in out.columns:
        observed = _labels(out[column])
        order = level_order(observed.dropna(), [design.control, *design.conditions])
        out[column] = pd.Categorical(observed, categories=order, ordered=False)
    for moderator, levels in design.moderators.items():
        if moderator not in out.columns:
            continue
        observed = _labels(out[moderator])
        order = level_order(observed.dropna(), levels)
        out[moderator] = pd.Categorical(observed, categories=order, ordered=False)
    return out


def assert_full_grid(
    pairs: pd.DataFrame, expected: int | None, label: str = "pairs"
) -> pd.DataFrame:
    """Raise unless the join produced the complete scored grid, with no gaps.

    The benchmark's own pipeline stops here rather than scoring a partial grid, so
    this raises: a submission that cannot be paired completely is a submission
    whose score would be computed on a self-selected subset.  ``expected=None``
    means nobody could say how large the grid is, and then this checks only for
    missing estimates — so the callers here never pass ``None``: a design that
    cannot derive its own pair count falls back to the size of the human
    reference, which is the grid by definition.
    """
    if expected is not None and len(pairs) != expected:
        raise ValueError(
            f"{label}: expected {expected} rows, got {len(pairs)} — a condition or "
            "outcome label failed to join (check for mislabeled conditions)"
        )
    for column in ("estimate_h", "estimate_l"):
        if column in pairs.columns and pairs[column].isna().any():
            missing = int(pairs[column].isna().sum())
            raise ValueError(f"{label}: {missing} missing values in {column}")
    return pairs


def to_pp(frame: pd.DataFrame, design: ScoredDesign) -> pd.DataFrame:
    """Put estimates and standard errors in pp of each outcome's scale range."""
    out = frame.copy()
    factor = 100.0 / out["outcome"].map(design.scale).astype(float)
    for column in ("estimate_h", "se_h", "estimate_l", "se_l", "estimate", "se"):
        if column in out.columns:
            out[column] = out[column] * factor
    return out


# --------------------------------------------------------------------------- #
# Section 1 — average treatment effects
# --------------------------------------------------------------------------- #


def _logit_fit(X: np.ndarray, y: np.ndarray, max_iter: int = 100) -> np.ndarray:
    """IRLS for a logistic regression; returns the coefficients."""
    beta = np.zeros(X.shape[1])
    for _ in range(max_iter):
        eta = X @ beta
        p = 1.0 / (1.0 + np.exp(-eta))
        w = np.clip(p * (1.0 - p), 1e-10, None)
        z = eta + (y - p) / w
        weighted = X * w[:, None]
        step = np.linalg.pinv(X.T @ weighted) @ (weighted.T @ z)
        if np.max(np.abs(step - beta)) < 1e-10:
            beta = step
            break
        beta = step
    return beta


def _logit_vcov_hc2(X: np.ndarray, y: np.ndarray, beta: np.ndarray) -> np.ndarray:
    """``sandwich::vcovHC(type="HC2")`` for a GLM: score residuals over leverage."""
    p = 1.0 / (1.0 + np.exp(-(X @ beta)))
    w = np.clip(p * (1.0 - p), 1e-10, None)
    bread = np.linalg.pinv((X * w[:, None]).T @ X)
    leverage = np.einsum("ij,jk,ik->i", X * w[:, None], bread, X)
    omega = (y - p) ** 2 / np.clip(1.0 - leverage, 1e-12, None)
    return bread @ ((X * omega[:, None]).T @ X) @ bread


def binary_marginal_effects(
    frame: pd.DataFrame,
    outcome: str,
    design: ScoredDesign,
    scale: float | None = None,
) -> pd.DataFrame:
    """ATEs for a 0/1 outcome: logistic fit, marginal effects on the probability scale.

    The preregistered path for ``newsletter_signup``.  With condition as the only
    regressor the average comparison collapses to the difference of two fitted
    cell proportions, so the point estimate equals the linear-probability one; the
    delta-method standard error does not, and that is the whole difference.  A
    cell with no signups at all sends the logit to infinity, so that case falls
    back to the LPM (flagged in ``model``), which is what keeps the estimate
    defined.

    The marginal effects carry their own BH-adjusted ``p_bh``, over this outcome's
    own family of effects, because the benchmark's binary model adjusts its
    p-values too — leaving the column out here made every binary row's ``p_bh``
    ``NaN`` once the tables were concatenated.
    """
    from scipy import stats

    column = design.condition_col
    scale = design.scale(outcome) if scale is None else scale
    data = frame[[column, outcome]].dropna()
    if data.empty or data[column].nunique() < 2:
        return pd.DataFrame()
    y = data[outcome].to_numpy(dtype=float)
    X, names = design_matrix(
        {column: data[column].astype(str)},
        reference={column: design.control},
        levels={column: [design.control, *design.conditions]},
    )
    proportions = data.groupby(column, observed=True)[outcome].mean()
    degenerate = bool(((proportions <= 0) | (proportions >= 1)).any())

    rows: list[dict] = []
    if degenerate:
        fit = ols(X, y, names, robust="HC2")
        for name in names:
            if name == "(Intercept)":
                continue
            term = fit.term(name)
            condition = name[len(column) + 1 : -1]
            rows.append(
                {
                    "outcome": outcome,
                    "condition": condition,
                    "n": int((data[column].astype(str) == condition).sum()),
                    "estimate": term["estimate"] / scale * 100,
                    "se": term["se"] / scale * 100,
                    "estimate_raw": term["estimate"],
                    "p": term["p"],
                    "model": "lpm",
                }
            )
        return _with_bh(pd.DataFrame(rows))

    beta = _logit_fit(X, y)
    cov = _logit_vcov_hc2(X, y, beta)

    def counterfactual(level: str) -> np.ndarray:
        row = np.zeros(len(names))
        row[0] = 1.0
        target = f"{column}[{level}]"
        if target in names:
            row[names.index(target)] = 1.0
        return row

    base = counterfactual(design.control)
    p_base = float(1.0 / (1.0 + np.exp(-(base @ beta))))

    for name in names:
        if name == "(Intercept)":
            continue
        condition = name[len(column) + 1 : -1]
        row = counterfactual(condition)
        p_row = float(1.0 / (1.0 + np.exp(-(row @ beta))))
        estimate = p_row - p_base
        gradient = p_row * (1 - p_row) * row - p_base * (1 - p_base) * base
        se = float(np.sqrt(max(gradient @ cov @ gradient, 0.0)))
        z = estimate / se if se > 0 else np.nan
        rows.append(
            {
                "outcome": outcome,
                "condition": condition,
                "n": int((data[column].astype(str) == condition).sum()),
                "estimate": estimate / scale * 100,
                "se": se / scale * 100,
                "estimate_raw": estimate,
                "p": float(2 * stats.norm.sf(abs(z))) if se > 0 else np.nan,
                "model": "logit",
            }
        )
    return _with_bh(pd.DataFrame(rows))


def _with_bh(table: pd.DataFrame) -> pd.DataFrame:
    """Add BH-adjusted p-values to an effect table, in place of nothing."""
    if not table.empty and "p" in table.columns:
        table["p_bh"] = benjamini_hochberg(table["p"].tolist())
    return table


def ate_side(frame: pd.DataFrame, design: ScoredDesign) -> pd.DataFrame:
    """One side of the ATE grid: every condition x outcome effect, in pp of scale.

    OLS with HC2 errors for the continuous outcomes and the donation, logistic
    marginal effects for a binary one.  Levels are aligned first, so ``control``
    is the reference whatever the CSV's column order was.
    """
    aligned = align_submission_levels(frame, design)
    continuous = {
        name: scale
        for name, scale in design.outcomes.items()
        if name not in set(design.binary)
    }
    parts = []
    if continuous:
        parts.append(
            treatment_effects(
                aligned,
                continuous,
                design.control,
                condition_col=design.condition_col,
                robust="HC2",
            ).assign(model="ols")
        )
    for outcome in design.binary:
        part = binary_marginal_effects(aligned, outcome, design)
        if len(part):
            parts.append(part)
    usable = [part for part in parts if len(part)]
    if not usable:
        return pd.DataFrame()
    table = pd.concat(usable, ignore_index=True)
    order = {name: index for index, name in enumerate(design.outcomes)}
    table["_order"] = table["outcome"].map(order)
    return (
        table.sort_values(["_order", "condition"])
        .drop(columns="_order")
        .reset_index(drop=True)
    )


def build_ate_pairs(
    human_side: pd.DataFrame, prediction: pd.DataFrame, design: ScoredDesign
) -> pd.DataFrame:
    """Human-vs-prediction ATE pairs on the full grid, in pp of scale range.

    ``human_side`` is a fitted effect table (:func:`ate_side`), ``prediction`` a
    respondent-level frame whose effects are refit here — the asymmetry is the
    benchmark's, which fits the reference once and every submission separately.
    """
    if not len(human_side):
        raise ValueError(
            "ATE pairs: the human reference produced no effects — there is no grid "
            "to score against"
        )
    expected = design.expected_pairs
    if expected is None:
        expected = len(human_side)
    pairs = ate_pairs(human_side, ate_side(prediction, design))
    return assert_full_grid(pairs, expected, "ATE pairs")


def metrics_by_group(
    pairs: pd.DataFrame,
    group: str,
    statistic: Callable[[pd.DataFrame], dict] = pooled_metrics,
) -> pd.DataFrame:
    """Score within each level of ``group`` instead of over everything at once.

    ``group="outcome"`` says which outcomes an approach can predict at all;
    ``group="condition"`` says which interventions it understands.  The two answer
    different questions and the benchmark reports both.
    """
    rows = []
    for level, part in pairs.groupby(group, sort=True, observed=True):
        rows.append({group: level, **statistic(part)})
    return pd.DataFrame(rows)


def breakdowns(pairs: pd.DataFrame, design: ScoredDesign) -> dict[str, pd.DataFrame]:
    """The three reported cuts of the pooled row: outcome, intervention, class.

    The per-intervention cut runs on the continuous outcomes only.  The pp
    conversion removes the unit mix, but a correlation across all thirteen
    outcomes within one intervention would still mix attitude sliders with
    behaviour, so the benchmark keeps the conservative restriction.
    """
    continuous = set(design.continuous_outcomes)
    per_intervention = pairs[pairs["outcome"].isin(continuous)]
    classed = pairs.assign(outcome_class=pairs["outcome"].map(design.outcome_class))
    return {
        "by_outcome": metrics_by_group(pairs, "outcome"),
        "by_intervention": (
            metrics_by_group(per_intervention, "condition")
            if len(per_intervention)
            else pd.DataFrame()
        ),
        "by_class": metrics_by_group(classed, "outcome_class"),
    }


# --------------------------------------------------------------------------- #
# Section 2 — subgroup effects (condition x moderator interactions)
# --------------------------------------------------------------------------- #


def _split_interaction(
    name: str, condition_col: str, moderator: str
) -> tuple[str, str]:
    """``condition[X]:moderator[Y]`` -> ``("X", "Y")``."""
    left, right = name.split(":", 1)
    condition = left[len(condition_col) + 1 : -1]
    level = right[len(moderator) + 1 : -1]
    return condition, level


def run_moderator_model(
    frame: pd.DataFrame,
    outcome: str,
    moderator: str,
    design: ScoredDesign,
    robust: str = "HC2",
) -> pd.DataFrame:
    """The condition x moderator interaction coefficients, in raw outcome units.

    ``lm(outcome ~ condition * moderator)`` with HC2 errors, keeping only the
    interaction terms — the benchmark's Section 2 estimand.  Each coefficient is a
    difference in differences: how much more (or less) an intervention moved this
    demographic group than it moved the moderator's reference group.  A binary
    outcome runs through the same OLS as a linear probability model, which in this
    saturated specification *is* the difference-in-differences of cell
    proportions.

    Note what this is not: the ATE *within* a moderator level
    (:func:`silicon_sampling.analysis.moderators.subgroup_effects`).  That is a
    readable table; this is the scored quantity.
    """
    column = design.condition_col
    data = frame[[column, moderator, outcome]].dropna()
    if data.empty or data[column].nunique() < 2 or data[moderator].nunique() < 2:
        return pd.DataFrame()
    y = data[outcome].to_numpy(dtype=float)
    cond_X, cond_names = design_matrix(
        {column: data[column].astype(str)},
        reference={column: design.control},
        levels={column: [design.control, *design.conditions]},
    )
    mod_X, mod_names = design_matrix(
        {moderator: data[moderator].astype(str)},
        levels={moderator: design.moderators.get(moderator)},
    )
    inter_X, inter_names = interaction(cond_X, cond_names, mod_X, mod_names)
    if not inter_names:
        return pd.DataFrame()
    X = np.hstack([cond_X, mod_X[:, 1:], inter_X])
    names = cond_names + mod_names[1:] + inter_names

    # An empty cell leaves an interaction with no identified answer, and there are
    # two shapes of it.  An empty *treatment* x level cell zeroes the interaction
    # column outright.  An empty *control* x level cell is the dangerous one: the
    # level's main effect and its interaction become the same non-zero column, so
    # `pinv` splits the effect between them and reports both with a small standard
    # error — a confident estimate of a quantity the data cannot identify.  It is
    # also the plausible one at this scale, where `gender[Other]` can be absent
    # from a thousand-respondent control arm.  R drops the aliased coefficient and
    # `broom::tidy` omits the row, so both shapes are found here and reported
    # missing rather than estimated.
    dropped = set(aliased_columns(X))
    keep = [index for index in range(X.shape[1]) if index not in dropped]
    fitted_names = [names[index] for index in keep]
    fit = ols(X[:, keep], y, fitted_names, robust=robust)

    rows = []
    for name in inter_names:
        condition, level = _split_interaction(name, column, moderator)
        record = {
            "outcome": outcome,
            "moderator": moderator,
            "condition": condition,
            "moderator_level": level,
            "baseline": design.control,
            "reference_level": (
                design.moderators.get(moderator)
                or level_order(_labels(data[moderator]).dropna())
            )[0],
        }
        if name in fitted_names:
            term = fit.term(name)
            record.update(
                {
                    "estimate": term["estimate"],
                    "se": term["se"],
                    "conf_low": term["conf_low"],
                    "conf_high": term["conf_high"],
                    "p": term["p"],
                }
            )
        else:
            record.update(
                {
                    "estimate": np.nan,
                    "se": np.nan,
                    "conf_low": np.nan,
                    "conf_high": np.nan,
                    "p": np.nan,
                }
            )
        rows.append(record)
    return pd.DataFrame(rows)


def subgroup_side(
    frame: pd.DataFrame, design: ScoredDesign, outcomes: Sequence[str] | None = None
) -> pd.DataFrame:
    """Every interaction coefficient of one dataset: outcomes x moderators."""
    aligned = align_submission_levels(frame, design)
    parts = []
    for outcome in outcomes or design.subgroup_set:
        for moderator in design.moderators:
            if moderator not in aligned.columns:
                continue
            part = run_moderator_model(aligned, outcome, moderator, design)
            if len(part):
                parts.append(part)
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


#: The columns that identify one interaction estimate, and so the join keys of
#: the subgroup pairs.
SUBGROUP_KEYS = ("outcome", "moderator", "condition", "moderator_level")


def build_subgroup_pairs(
    human_side: pd.DataFrame, prediction: pd.DataFrame, design: ScoredDesign
) -> pd.DataFrame:
    """Paired human-vs-prediction interaction estimates, in pp of scale range.

    The human side defines the grid: every human interaction estimate must find
    its counterpart in the submission's refit, or the join has silently dropped
    part of the test set and this raises.
    """
    predicted = subgroup_side(
        prediction, design, outcomes=sorted(set(human_side["outcome"].unique()))
    )
    if predicted.empty:
        raise ValueError("subgroup pairs: the submission produced no interaction terms")
    left = human_side.rename(columns={"estimate": "estimate_h", "se": "se_h"})[
        [*SUBGROUP_KEYS, "estimate_h", "se_h"]
    ]
    right = predicted.rename(columns={"estimate": "estimate_l", "se": "se_l"})[
        [*SUBGROUP_KEYS, "estimate_l", "se_l"]
    ]
    pairs = left.merge(right, on=list(SUBGROUP_KEYS), how="inner")
    if len(pairs) != len(left):
        raise ValueError(
            f"subgroup pairs: {len(left)} human interaction estimates, "
            f"{len(pairs)} matched — check moderator level spelling"
        )
    return to_pp(pairs, design)


def subgroup_metrics(pairs: pd.DataFrame) -> dict:
    """Pooled subgroup metrics: the ATE ladder without RMSE."""
    return signed_metrics(pairs)


# --------------------------------------------------------------------------- #
# Section 3 — response distributions and demographic baselines
# --------------------------------------------------------------------------- #


def response_distributions(
    human: pd.DataFrame,
    prediction: pd.DataFrame,
    design: ScoredDesign,
    outcomes: Sequence[str] | None = None,
) -> pd.DataFrame:
    """The four shape metrics per continuous outcome, in the control condition.

    The OVL/W1 grid is fixed to the outcome's full scale rather than the observed
    range, so two submissions are integrated over the same interval and their
    numbers are comparable.
    """
    column = design.condition_col
    rows = []
    for outcome in outcomes or design.continuous_outcomes:
        left = _numeric(
            human.loc[human[column].astype(str) == design.control, outcome]
        ).dropna()
        right = _numeric(
            prediction.loc[prediction[column].astype(str) == design.control, outcome]
        ).dropna()
        if len(left) < 2 or len(right) < 2:
            continue
        rows.append(
            {
                "outcome": outcome,
                "condition": design.control,
                **dist.compare_distributions(
                    left, right, lo=0.0, hi=design.scale(outcome)
                ),
            }
        )
    return pd.DataFrame(rows)


def subgroup_distributions(
    human: pd.DataFrame,
    prediction: pd.DataFrame,
    design: ScoredDesign,
    outcomes: Sequence[str] | None = None,
    min_n: int | None = None,
) -> pd.DataFrame:
    """Shape metrics within each demographic group, in the control condition.

    Catches an approach that reproduces the overall response distribution while
    miscalibrating the spread inside groups.  Groups under ``min_n`` on either
    side are skipped, because a kernel density over a dozen respondents is not a
    distribution estimate — but they are *reported* as skipped, with their sizes,
    since the smallest groups are the population this metric exists for.
    """
    column = design.condition_col
    threshold = design.min_group_n if min_n is None else min_n
    human_control = human[human[column].astype(str) == design.control]
    pred_control = prediction[prediction[column].astype(str) == design.control]
    rows = []
    for moderator, levels in design.moderators.items():
        if moderator not in human.columns or moderator not in prediction.columns:
            continue
        order = levels or level_order(_labels(human_control[moderator]).dropna())
        for level in order:
            left_cell = human_control[_labels(human_control[moderator]) == level]
            right_cell = pred_control[_labels(pred_control[moderator]) == level]
            for outcome in outcomes or design.continuous_outcomes:
                left = _numeric(left_cell[outcome]).dropna()
                right = _numeric(right_cell[outcome]).dropna()
                record = {
                    "moderator": moderator,
                    "level": level,
                    "outcome": outcome,
                    "n_human": len(left),
                    "n_prediction": len(right),
                }
                if len(left) < threshold or len(right) < threshold:
                    rows.append(
                        {
                            **record,
                            "skipped": True,
                            **{metric: np.nan for metric in SHAPE_METRICS},
                        }
                    )
                    continue
                shape = dist.compare_distributions(
                    left, right, lo=0.0, hi=design.scale(outcome)
                )
                rows.append(
                    {
                        **record,
                        "skipped": False,
                        **{metric: shape[metric] for metric in SHAPE_METRICS},
                    }
                )
    return pd.DataFrame(rows)


def compare_demographic_baselines(
    human: pd.DataFrame,
    prediction: pd.DataFrame,
    outcome: str,
    design: ScoredDesign,
    min_cells_r: int = 0,
) -> pd.DataFrame:
    """Per moderator, how well the control-condition group means line up.

    In **raw outcome points**, not pp: a group mean is a response level, and the
    question is whether the approach starts each demographic group off at the
    right level before any intervention.  The correlation is reported but
    de-emphasised — over three to six group means it is unstable and blind to
    level and scale, which is why the benchmark rests on the RMSE.  It comes back
    ``NaN`` when fewer than two cells survive or one side is constant, which is
    what R's ``cor()`` answers there rather than a deviation from it;
    ``min_cells_r`` can only raise that floor.
    """
    column = design.condition_col
    rows = []
    left_control = human[human[column].astype(str) == design.control]
    right_control = prediction[prediction[column].astype(str) == design.control]
    for moderator in design.moderators:
        if moderator not in human.columns or moderator not in prediction.columns:
            continue
        left = (
            _numeric(left_control[outcome])
            .groupby(_labels(left_control[moderator]))
            .mean()
        )
        right = (
            _numeric(right_control[outcome])
            .groupby(_labels(right_control[moderator]))
            .mean()
        )
        cells = pd.concat(
            [left.rename("mean_h"), right.rename("mean_l")], axis=1, join="inner"
        )
        cells = cells.dropna()
        if cells.empty:
            continue
        correlation = np.nan
        if len(cells) >= max(min_cells_r, 2) and len(cells) >= 2:
            if cells["mean_h"].std(ddof=1) > 0 and cells["mean_l"].std(ddof=1) > 0:
                correlation = float(cells["mean_h"].corr(cells["mean_l"]))
        rows.append(
            {
                "outcome": outcome,
                "moderator": moderator,
                "r": correlation,
                "rmse": float(
                    np.sqrt(np.mean((cells["mean_h"] - cells["mean_l"]) ** 2))
                ),
                "n_cells": int(len(cells)),
            }
        )
    return pd.DataFrame(rows)


def demographic_parity_gap(
    human: pd.DataFrame,
    prediction: pd.DataFrame,
    outcome: str,
    design: ScoredDesign,
    min_n: int | None = None,
) -> pd.DataFrame:
    """Worst- minus best-served demographic group, per moderator, in raw points.

    The baseline RMSE averages over groups, and an average hides an approach that
    serves most groups well and one group badly.  ``dpd`` is the gap between the
    largest and smallest absolute group-mean error; ``worst_abs_err`` rides along
    because a zero gap can also mean every group is served equally badly.  Groups
    under ``min_n`` on either side are skipped and named — the max-over-groups
    error is a max-of-noise statistic, mechanically positive even for a perfect
    approach, and the smallest cells are where that bites.
    """
    column = design.condition_col
    threshold = design.min_group_n if min_n is None else min_n
    human_control = human[human[column].astype(str) == design.control]
    pred_control = prediction[prediction[column].astype(str) == design.control]
    rows = []
    for moderator, levels in design.moderators.items():
        if moderator not in human.columns or moderator not in prediction.columns:
            continue
        order = list(levels or level_order(_labels(human_control[moderator]).dropna()))
        errors: list[float] = []
        skipped: list[str] = []
        detail: list[str] = []
        for level in order:
            left = _numeric(
                human_control.loc[_labels(human_control[moderator]) == level, outcome]
            ).dropna()
            right = _numeric(
                pred_control.loc[_labels(pred_control[moderator]) == level, outcome]
            ).dropna()
            if len(left) < threshold or len(right) < threshold:
                errors.append(np.nan)
                skipped.append(level)
                detail.append(f"{level} (human {len(left)}, prediction {len(right)})")
                continue
            errors.append(abs(float(right.mean()) - float(left.mean())))
        values = np.asarray(errors, dtype=float)
        record = {
            "outcome": outcome,
            "moderator": moderator,
            "n_skipped": len(skipped),
            "groups_skipped": ", ".join(skipped) if skipped else None,
            "groups_skipped_detail": ", ".join(detail) if detail else None,
        }
        if np.all(np.isnan(values)):
            rows.append(
                {
                    **record,
                    "dpd": np.nan,
                    "worst_abs_err": np.nan,
                    "worst_group": None,
                    "best_group": None,
                    "n_groups": 0,
                }
            )
            continue
        rows.append(
            {
                **record,
                "dpd": float(np.nanmax(values) - np.nanmin(values)),
                "worst_abs_err": float(np.nanmax(values)),
                "worst_group": order[int(np.nanargmax(values))],
                "best_group": order[int(np.nanargmin(values))],
                "n_groups": int(np.sum(~np.isnan(values))),
            }
        )
    return pd.DataFrame(rows)


def compare_demographic_predictability(
    human: pd.DataFrame,
    prediction: pd.DataFrame,
    outcome: str,
    design: ScoredDesign,
) -> dict[str, pd.DataFrame]:
    """The stereotyping diagnostic: ``outcome ~ moderator + condition``, both sides.

    Two quantities with two jobs.  The **dummy-coded coefficients are the primary
    one**: each is a group's gap to the moderator's reference level, and putting
    the submission's gaps beside the human ones tests directly whether the
    approach exaggerates demographic differences.  The R-squared pair is the
    summary, and it is ambiguous by construction — a higher synthetic R-squared
    means either exaggerated gaps or synthetic respondents answering too much
    alike, and only the coefficients (and the variance ratio) separate the two.

    Standard errors are the textbook ones, because the preregistered function is a
    plain ``lm()`` and its intervals are what the benchmark will print.
    """
    column = design.condition_col
    r2_rows, coefficient_rows = [], []
    for moderator, levels in design.moderators.items():
        if moderator not in human.columns or moderator not in prediction.columns:
            continue
        fits = {}
        for side, frame in (("h", human), ("l", prediction)):
            data = frame[[column, moderator, outcome]].dropna()
            if data.empty or data[moderator].nunique() < 2:
                fits[side] = None
                continue
            X, names = design_matrix(
                {
                    moderator: data[moderator].astype(str),
                    column: data[column].astype(str),
                },
                reference={column: design.control},
                levels={
                    moderator: levels,
                    column: [design.control, *design.conditions],
                },
            )
            fits[side] = ols(
                X, data[outcome].to_numpy(float), names, robust="classical"
            )
        if fits["h"] is None or fits["l"] is None:
            continue
        reference = (levels or level_order(_labels(human[moderator]).dropna()))[0]
        r2_rows.append(
            {
                "outcome": outcome,
                "moderator": moderator,
                "reference_level": reference,
                "r_squared_h": fits["h"].r2,
                "r_squared_l": fits["l"].r2,
                "r_squared_gap": fits["l"].r2 - fits["h"].r2,
            }
        )
        shared = [
            name
            for name in fits["h"].names
            if name.startswith(f"{moderator}[") and name in fits["l"].names
        ]
        for name in shared:
            left = fits["h"].term(name)
            right = fits["l"].term(name)
            coefficient_rows.append(
                {
                    "outcome": outcome,
                    "moderator": moderator,
                    "term": name,
                    "level": name[len(moderator) + 1 : -1],
                    "reference_level": reference,
                    "est_h": left["estimate"],
                    "lo_h": left["conf_low"],
                    "hi_h": left["conf_high"],
                    "est_l": right["estimate"],
                    "lo_l": right["conf_low"],
                    "hi_l": right["conf_high"],
                    "diff": right["estimate"] - left["estimate"],
                }
            )
    return {
        "r_squared": pd.DataFrame(r2_rows),
        "coefficients": pd.DataFrame(coefficient_rows),
    }


def summarise_field(
    field_long: pd.DataFrame, ceiling_label: str = "Human replication"
) -> pd.DataFrame:
    """Centre and spread of one metric across submissions, reference row excluded.

    Expects long input with ``submission``, ``metric``, ``value``.  The human
    replication row is dropped because it is the yardstick, not a competitor.
    """
    field = field_long[field_long["submission"] != ceiling_label]
    grouped = field.groupby("metric", sort=True)["value"]
    return pd.DataFrame(
        {
            "mean": grouped.mean(),
            "median": grouped.median(),
            "sd": grouped.std(ddof=1),
            "min": grouped.min(),
            "max": grouped.max(),
        }
    ).reset_index()


# --------------------------------------------------------------------------- #
# The assembler
# --------------------------------------------------------------------------- #


@dataclass
class ReferenceSides:
    """The human reference, fitted once and reused for every submission.

    Refitting seventy-eight moderator models per scored entry is the expensive
    part of this pipeline, and the human side does not change between entries.
    """

    ate: pd.DataFrame
    subgroup: pd.DataFrame


def reference_sides(
    human: pd.DataFrame, design: ScoredDesign, subgroup: bool = True
) -> ReferenceSides:
    """Fit the human reference's ATE and interaction estimates."""
    return ReferenceSides(
        ate=ate_side(human, design),
        subgroup=subgroup_side(human, design) if subgroup else pd.DataFrame(),
    )


@dataclass
class ScoredReport:
    """Every scored analysis for one submission, as tables plus one flat row.

    The tables are what you read when a number looks wrong; :meth:`row` is what
    you feed to a calibration search, which needs every scored quantity of one
    submission side by side in a single flat namespace.
    """

    label: str
    design: ScoredDesign
    pairs: pd.DataFrame
    pooled: dict
    calibration: dict
    intervals: dict
    by_outcome: pd.DataFrame
    by_intervention: pd.DataFrame
    by_class: pd.DataFrame
    subgroup_pairs: pd.DataFrame
    subgroup_pooled: dict
    subgroup_by_moderator: pd.DataFrame
    distributions: pd.DataFrame
    subgroup_shape: pd.DataFrame
    baselines: pd.DataFrame
    parity: pd.DataFrame
    stereotyping_r2: pd.DataFrame
    stereotyping_coefficients: pd.DataFrame

    def row(self) -> dict:
        """Every scored number of this submission, flat, with self-describing keys.

        Key grammar, one scope per prefix, ``/`` separating the scope from what it
        localises:

        - bare names (``pearson_r``, ``beta``) — the pooled ATE row and its
          calibration line, in pp of scale range; ``*_lo`` / ``*_hi`` are cluster
          bootstrap bounds when a bootstrap was run.
        - ``outcome/<outcome>/<metric>``, ``intervention/<condition>/<metric>``,
          ``class/<Self-report|Behavioral>/<metric>`` — the reported breakdowns.
        - ``subgroup/pooled/<metric>``, ``subgroup/<moderator>/<metric>`` —
          interaction-estimate recovery, pp, no RMSE.
        - ``shape/<outcome>/<metric>`` and ``shape/<moderator>/<metric>_*`` —
          control-condition distribution shape, raw response units.
        - ``baseline/<outcome>/<moderator>/<r|rmse|n_cells>``,
          ``parity/<outcome>/<moderator>/<dpd|worst_abs_err|...>``,
          ``stereo/<outcome>/<moderator>/...`` — the demographic analyses, in raw
          outcome points.
        - ``median/...`` — the median across the levels of a breakdown, so a
          search has one number per analysis as well as the detail.
        """
        row: dict = {"submission": self.label}
        row.update({key: value for key, value in self.pooled.items()})
        row.update(self.calibration)
        row.update(self.intervals)

        for scope, table, key in (
            ("outcome", self.by_outcome, "outcome"),
            ("intervention", self.by_intervention, "condition"),
            ("class", self.by_class, "outcome_class"),
        ):
            if table.empty:
                continue
            for _, record in table.iterrows():
                for metric in BREAKDOWN_METRICS:
                    if metric in table.columns:
                        row[f"{scope}/{record[key]}/{metric}"] = record[metric]
            for metric in BREAKDOWN_METRICS:
                if metric in table.columns:
                    row[f"median/{scope}/{metric}"] = _median(table[metric])

        if "n_pairs" in self.subgroup_pooled:
            row["subgroup/pooled/n_pairs"] = self.subgroup_pooled["n_pairs"]
        for metric in SUBGROUP_METRICS:
            if metric in self.subgroup_pooled:
                row[f"subgroup/pooled/{metric}"] = self.subgroup_pooled[metric]
        for _, record in self.subgroup_by_moderator.iterrows():
            for metric in SUBGROUP_METRICS:
                if metric in self.subgroup_by_moderator.columns:
                    row[f"subgroup/{record['moderator']}/{metric}"] = record[metric]

        for _, record in self.distributions.iterrows():
            for metric in SHAPE_METRICS:
                row[f"shape/{record['outcome']}/{metric}"] = record[metric]
        for metric in SHAPE_METRICS:
            if len(self.distributions):
                row[f"median/shape/{metric}"] = _median(self.distributions[metric])

        if len(self.subgroup_shape):
            scored = self.subgroup_shape[~self.subgroup_shape["skipped"]]
            for moderator, part in scored.groupby("moderator", sort=True):
                row[f"shape/{moderator}/variance_ratio_median"] = _median(
                    part["variance_ratio"]
                )
                row[f"shape/{moderator}/variance_ratio_min"] = float(
                    part["variance_ratio"].min()
                )
                row[f"shape/{moderator}/w1_worst"] = float(part["w1"].max())
            for moderator, part in self.subgroup_shape.groupby("moderator", sort=True):
                row[f"shape/{moderator}/n_skipped"] = int(part["skipped"].sum())

        for _, record in self.baselines.iterrows():
            stem = f"baseline/{record['outcome']}/{record['moderator']}"
            row[f"{stem}/r"] = record["r"]
            row[f"{stem}/rmse"] = record["rmse"]
            row[f"{stem}/n_cells"] = record["n_cells"]
        if len(self.baselines):
            row["median/baseline/rmse"] = _median(self.baselines["rmse"])

        for _, record in self.parity.iterrows():
            stem = f"parity/{record['outcome']}/{record['moderator']}"
            row[f"{stem}/dpd"] = record["dpd"]
            row[f"{stem}/worst_abs_err"] = record["worst_abs_err"]
            row[f"{stem}/worst_group"] = record["worst_group"]
            row[f"{stem}/best_group"] = record["best_group"]
            row[f"{stem}/n_skipped"] = record["n_skipped"]
        if len(self.parity):
            row["median/parity/dpd"] = _median(self.parity["dpd"])
            row["max/parity/worst_abs_err"] = float(
                pd.to_numeric(self.parity["worst_abs_err"], errors="coerce").max()
            )

        for _, record in self.stereotyping_r2.iterrows():
            stem = f"stereo/{record['outcome']}/{record['moderator']}"
            row[f"{stem}/r2_h"] = record["r_squared_h"]
            row[f"{stem}/r2_l"] = record["r_squared_l"]
            row[f"{stem}/r2_gap"] = record["r_squared_gap"]
        coefficients = self.stereotyping_coefficients
        for _, record in coefficients.iterrows():
            stem = f"stereo/{record['outcome']}/{record['moderator']}/{record['level']}"
            row[f"{stem}/est_h"] = record["est_h"]
            row[f"{stem}/est_l"] = record["est_l"]
            row[f"{stem}/diff"] = record["diff"]
        if len(coefficients):
            for moderator, part in coefficients.groupby("moderator", sort=True):
                row[f"stereo/{moderator}/coef_rmse"] = float(
                    np.sqrt(np.mean(part["diff"] ** 2))
                )
                row[f"stereo/{moderator}/coef_max_abs_diff"] = float(
                    part["diff"].abs().max()
                )
            row["median/stereo/coef_rmse"] = _median(
                [
                    np.sqrt(np.mean(part["diff"] ** 2))
                    for _, part in coefficients.groupby("moderator")
                ]
            )
            row["median/stereo/r2_gap"] = _median(self.stereotyping_r2["r_squared_gap"])
        return row


def score_submission(
    human: pd.DataFrame,
    prediction: pd.DataFrame,
    design: ScoredDesign,
    label: str = "submission",
    sides: ReferenceSides | None = None,
    bootstrap: int = 0,
    seed: int = 2026,
    subgroups: bool = True,
) -> ScoredReport:
    """Run every scored analysis of the benchmark on one submission.

    ``sides`` passes in a human reference fitted earlier — worth doing whenever
    more than one submission is scored, since the human interaction models are the
    expensive part.  ``bootstrap`` is the number of cluster-bootstrap draws for
    the pooled row (the preregistered value is 2000; 0 skips it).  ``subgroups``
    has to be told, not inferred: asking for Section 2 with a reference that was
    fitted without it raises rather than returning a row that is quietly fifty
    keys short.
    """
    sides = sides or reference_sides(human, design, subgroup=subgroups)
    pairs = build_ate_pairs(sides.ate, prediction, design)
    pooled = pooled_metrics(pairs, include_rmse=True)
    calibration = run_calibration_pooled(pairs)
    intervals: dict = {}
    if bootstrap:
        interval = cluster_bootstrap(
            pairs,
            lambda part: {
                **pooled_metrics(part, include_rmse=True),
                **run_calibration_pooled(part),
            },
            cluster=design.condition_col,
            draws=bootstrap,
            seed=seed,
        )
        # A bootstrap interval on the *pair count* is not a metric; drop it so the
        # row's interval keys line up one-to-one with scored quantities.
        intervals = {
            key: value
            for key, value in interval.items()
            if key.endswith(("_lo", "_hi")) and not key.startswith("n_pairs")
        }

    cuts = breakdowns(pairs, design)

    subgroup_pairs = pd.DataFrame()
    subgroup_pooled: dict = {}
    subgroup_by_moderator = pd.DataFrame()
    if subgroups and design.moderators and not len(sides.subgroup):
        # Fifty-odd keys of Section 2 would otherwise vanish from the row with no
        # sign that they were ever expected, which reads as a submission that
        # simply was not scored on subgroups.
        raise ValueError(
            "subgroups=True but the reference sides carry no interaction "
            "estimates — either the ReferenceSides was built with subgroup=False "
            "or the human frame is missing every moderator column; pass "
            "subgroups=False to score without Section 2 deliberately"
        )
    if subgroups and len(sides.subgroup):
        subgroup_pairs = build_subgroup_pairs(sides.subgroup, prediction, design)
        subgroup_pooled = subgroup_metrics(subgroup_pairs)
        subgroup_by_moderator = metrics_by_group(
            subgroup_pairs, "moderator", statistic=signed_metrics
        )

    aligned_human = align_submission_levels(human, design)
    aligned_prediction = align_submission_levels(prediction, design)
    baselines, parity, r2, coefficients = [], [], [], []
    for outcome in design.baseline_set:
        baselines.append(
            compare_demographic_baselines(
                aligned_human, aligned_prediction, outcome, design
            )
        )
        parity.append(
            demographic_parity_gap(aligned_human, aligned_prediction, outcome, design)
        )
        stereotyping = compare_demographic_predictability(
            aligned_human, aligned_prediction, outcome, design
        )
        r2.append(stereotyping["r_squared"])
        coefficients.append(stereotyping["coefficients"])

    def stack(parts: list[pd.DataFrame]) -> pd.DataFrame:
        usable = [part for part in parts if len(part)]
        return pd.concat(usable, ignore_index=True) if usable else pd.DataFrame()

    return ScoredReport(
        label=label,
        design=design,
        pairs=pairs,
        pooled=pooled,
        calibration=calibration,
        intervals=intervals,
        by_outcome=cuts["by_outcome"],
        by_intervention=cuts["by_intervention"],
        by_class=cuts["by_class"],
        subgroup_pairs=subgroup_pairs,
        subgroup_pooled=subgroup_pooled,
        subgroup_by_moderator=subgroup_by_moderator,
        distributions=response_distributions(aligned_human, aligned_prediction, design),
        subgroup_shape=(
            subgroup_distributions(aligned_human, aligned_prediction, design)
            if subgroups
            else pd.DataFrame()
        ),
        baselines=stack(baselines),
        parity=stack(parity),
        stereotyping_r2=stack(r2),
        stereotyping_coefficients=stack(coefficients),
    )


def leaderboard_row(
    human: pd.DataFrame,
    prediction: pd.DataFrame,
    design: ScoredDesign,
    label: str = "submission",
    sides: ReferenceSides | None = None,
    bootstrap: int = 0,
    seed: int = 2026,
    subgroups: bool = True,
) -> dict:
    """Every scored number for one submission, in one flat dict.

    The point of the flat shape: a calibration is a choice made against *all* the
    scored quantities at once, and comparing two candidate calibrations means
    diffing two of these dicts.  See :meth:`ScoredReport.row` for the key grammar.
    """
    return score_submission(
        human,
        prediction,
        design,
        label=label,
        sides=sides,
        bootstrap=bootstrap,
        seed=seed,
        subgroups=subgroups,
    ).row()
