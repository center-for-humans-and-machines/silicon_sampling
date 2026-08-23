"""Does who the respondent is change what they answer, or how they react?

Three different questions, kept separate because they fail differently:

**Baselines** — do demographic groups differ in the control condition?  A synthetic
sample can get the average right while placing every group on top of it.

**Moderation** — does an intervention work differently for different groups?  This
is the condition x moderator interaction.

**Predictability** — how much of the variance is demographics alone?  A model that
answers from a stereotype produces a sample where knowing someone's party tells
you their answer almost exactly; real people are far noisier than that.  This is
the benchmark's stereotyping diagnostic.

These are *descriptive* views of one sample, and two of them are deliberately
not the benchmark's scored estimands, which live in
:mod:`silicon_sampling.benchmark.scored`.  :func:`subgroup_effects` gives the ATE
*within* each moderator level, which reads well in a table but is not what the
benchmark scores — that is the condition x moderator interaction coefficient, a
difference of two such ATEs, with its own standard error.  :func:`parity_gap`
here is the spread of group means inside one sample; the benchmark's demographic
parity gap is the spread of *human-vs-prediction errors* across groups, and a
sample can look flat on one while failing badly on the other.  Keeping both is
the point: the descriptive view explains a scored number once it goes wrong.

The two regression views use **HC2** errors, which is what the Voelkel study and
the benchmark preregistration specify and what matters here: the control arm is
much larger than any single treatment arm, and HC2's leverage correction is
exactly the adjustment uneven cell sizes call for.  It is a keyword rather than a
constant because the choice moves published numbers — switching the default from
HC1 moved every ``moderation_test`` chi-square on the 18,000-row Pfänder sample
(median 0.49, up to 8.14, so the p-values and their ordering move with it), while
leaving :func:`subgroup_effects`' point estimates untouched, as any covariance
choice must.  Anything that reproduces an older table has to say so.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from scipy import stats

from .ols import design_matrix, interaction, ols, wald


def baseline_means(
    frame: pd.DataFrame,
    moderator: str,
    outcomes,
    condition: str = "control",
    condition_col: str = "condition",
) -> pd.DataFrame:
    """Cell means per moderator level, within one condition."""
    subset = frame[frame[condition_col] == condition]
    rows = []
    for outcome in outcomes:
        for level, group in subset.groupby(moderator, sort=True):
            values = group[outcome].dropna().to_numpy(dtype=float)
            if len(values) == 0:
                continue
            sem = (
                values.std(ddof=1) / np.sqrt(len(values))
                if len(values) > 1
                else float("nan")
            )
            crit = stats.t.ppf(0.975, df=max(len(values) - 1, 1))
            rows.append(
                {
                    "moderator": moderator,
                    "level": level,
                    "outcome": outcome,
                    "n": len(values),
                    "mean": float(values.mean()),
                    "sd": (
                        float(values.std(ddof=1)) if len(values) > 1 else float("nan")
                    ),
                    "conf_low": float(values.mean() - crit * sem),
                    "conf_high": float(values.mean() + crit * sem),
                }
            )
    return pd.DataFrame(rows)


def moderation_test(
    frame: pd.DataFrame,
    moderator: str,
    outcome: str,
    condition_col: str = "condition",
    control: str = "control",
    levels: Sequence[str] | None = None,
    robust: str = "HC2",
) -> dict:
    """Saturated ``outcome ~ condition * moderator``; joint test on the interaction.

    ``robust`` reaches the Wald statistic through the covariance matrix, so it is
    the whole difference between one recorded chi-square and another.
    """
    data = frame[[condition_col, moderator, outcome]].dropna()
    y = data[outcome].to_numpy(dtype=float)
    cond_X, cond_names = design_matrix(
        {condition_col: data[condition_col].tolist()},
        reference={condition_col: control},
    )
    mod_X, mod_names = design_matrix(
        {moderator: data[moderator].astype(str).tolist()},
        levels={moderator: levels} if levels else None,
    )
    inter_X, inter_names = interaction(cond_X, cond_names, mod_X, mod_names)
    X = np.hstack([cond_X, mod_X[:, 1:], inter_X])
    names = cond_names + mod_names[1:] + inter_names
    fit = ols(X, y, names, robust=robust)
    test = wald(fit, inter_names)
    return {
        "moderator": moderator,
        "outcome": outcome,
        "n": fit.n,
        "r2": fit.r2,
        "interaction_terms": len(inter_names),
        "chi2": test["chi2"],
        "df": test["df"],
        "p": test["p"],
    }


#: Kept explicit so a moderator with no usable level still yields a frame that
#: downstream code can filter, rather than an empty one with no columns.
SUBGROUP_COLUMNS = (
    "moderator",
    "level",
    "outcome",
    "condition",
    "estimate",
    "se",
    "conf_low",
    "conf_high",
    "p",
)


def subgroup_effects(
    frame: pd.DataFrame,
    moderator: str,
    outcome: str,
    condition_col: str = "condition",
    control: str = "control",
    robust: str = "HC2",
) -> pd.DataFrame:
    """The ATE of each intervention within each moderator level.

    ``robust`` changes the intervals only: the estimates are cell-mean differences
    and no covariance choice can move them.
    """
    rows = []
    for level, group in frame.groupby(moderator, sort=True):
        data = group[[condition_col, outcome]].dropna()
        if (
            data[condition_col].nunique() < 2
            or (data[condition_col] == control).sum() < 5
        ):
            continue
        y = data[outcome].to_numpy(dtype=float)
        X, names = design_matrix(
            {condition_col: data[condition_col].tolist()},
            reference={condition_col: control},
        )
        fit = ols(X, y, names, robust=robust)
        for name in names:
            if name == "(Intercept)":
                continue
            term = fit.term(name)
            rows.append(
                {
                    "moderator": moderator,
                    "level": level,
                    "outcome": outcome,
                    "condition": name[len(condition_col) + 1 : -1],
                    "estimate": term["estimate"],
                    "se": term["se"],
                    "conf_low": term["conf_low"],
                    "conf_high": term["conf_high"],
                    "p": term["p"],
                }
            )
    return pd.DataFrame(rows, columns=SUBGROUP_COLUMNS)


def predictability(
    frame: pd.DataFrame, moderators, outcomes, condition_col: str = "condition"
) -> pd.DataFrame:
    """R-squared of ``outcome ~ moderator + condition``, per moderator x outcome."""
    rows = []
    for moderator in moderators:
        for outcome in outcomes:
            data = frame[[condition_col, moderator, outcome]].dropna()
            if data.empty:
                continue
            y = data[outcome].to_numpy(dtype=float)
            X, names = design_matrix(
                {
                    moderator: data[moderator].astype(str).tolist(),
                    condition_col: data[condition_col].tolist(),
                }
            )
            fit = ols(X, y, names)
            cond_only_X, cond_only_names = design_matrix(
                {condition_col: data[condition_col].tolist()}
            )
            cond_only = ols(cond_only_X, y, cond_only_names)
            rows.append(
                {
                    "moderator": moderator,
                    "outcome": outcome,
                    "n": fit.n,
                    "r2_full": fit.r2,
                    "r2_condition_only": cond_only.r2,
                    # What the moderator adds once condition is already in.
                    "r2_moderator": max(fit.r2 - cond_only.r2, 0.0),
                }
            )
    return pd.DataFrame(rows)


def parity_gap(frame: pd.DataFrame, moderators, outcomes) -> pd.DataFrame:
    """Largest gap between any two moderator cells, per moderator x outcome."""
    rows = []
    for moderator in moderators:
        for outcome in outcomes:
            means = frame.groupby(moderator)[outcome].mean().dropna()
            if len(means) < 2:
                continue
            rows.append(
                {
                    "moderator": moderator,
                    "outcome": outcome,
                    "gap": float(means.max() - means.min()),
                    "worst_level": str(means.idxmin()),
                    "best_level": str(means.idxmax()),
                }
            )
    return pd.DataFrame(rows)
