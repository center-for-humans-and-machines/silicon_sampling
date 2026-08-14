"""Average treatment effects: every intervention against the shared control.

One OLS per outcome, with condition dummy-coded and control as the reference
level, gives all 16 contrasts at once with HC1 standard errors.  Effects are
reported three ways: in the outcome's own units, as Cohen's *d*, and in
percentage points of the outcome's scale range — the last being the unit the
benchmark scores on, so that a 5-point move on a 0-100 slider and a $0.50 move on
the donation item are comparable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .ols import benjamini_hochberg, design_matrix, holm, ols


def _pooled_sd(treated: np.ndarray, control: np.ndarray) -> float:
    n1, n0 = len(treated), len(control)
    if n1 < 2 or n0 < 2:
        return float("nan")
    var = ((n1 - 1) * treated.var(ddof=1) + (n0 - 1) * control.var(ddof=1)) / (
        n1 + n0 - 2
    )
    return float(np.sqrt(var))


def ate_table(
    frame: pd.DataFrame,
    outcomes: dict[str, float],
    condition_col: str = "condition",
    control: str = "control",
) -> pd.DataFrame:
    """One row per intervention x outcome."""
    rows = []
    for outcome, scale in outcomes.items():
        data = frame[[condition_col, outcome]].dropna()
        y = data[outcome].to_numpy(dtype=float)
        X, names = design_matrix(
            {condition_col: data[condition_col].tolist()},
            reference={condition_col: control},
        )
        fit = ols(X, y, names)
        control_values = y[data[condition_col].to_numpy() == control]
        for name in names:
            if name == "(Intercept)":
                continue
            condition = name[len(condition_col) + 1 : -1]
            term = fit.term(name)
            treated = y[data[condition_col].to_numpy() == condition]
            rows.append(
                {
                    "outcome": outcome,
                    "condition": condition,
                    "n_treated": len(treated),
                    "n_control": len(control_values),
                    "control_mean": float(control_values.mean()),
                    "treated_mean": float(treated.mean()),
                    "estimate": term["estimate"],
                    "se": term["se"],
                    "conf_low": term["conf_low"],
                    "conf_high": term["conf_high"],
                    "p": term["p"],
                    "cohens_d": term["estimate"] / _pooled_sd(treated, control_values),
                    "pp_scale": term["estimate"] / scale * 100,
                    "pp_se": term["se"] / scale * 100,
                }
            )
    table = pd.DataFrame(rows)
    table["p_holm"] = holm(table["p"].tolist())
    table["p_bh"] = benjamini_hochberg(table["p"].tolist())
    return table


def condition_means(
    frame: pd.DataFrame, outcomes, condition_col: str = "condition"
) -> pd.DataFrame:
    """Mean, SD and n per condition x outcome."""
    rows = []
    for outcome in outcomes:
        for condition, group in frame.groupby(condition_col, sort=True):
            values = group[outcome].dropna().to_numpy(dtype=float)
            rows.append(
                {
                    "outcome": outcome,
                    "condition": condition,
                    "n": len(values),
                    "mean": float(values.mean()) if len(values) else float("nan"),
                    "sd": (
                        float(values.std(ddof=1)) if len(values) > 1 else float("nan")
                    ),
                    "median": float(np.median(values)) if len(values) else float("nan"),
                }
            )
    return pd.DataFrame(rows)


def spread_of_effects(table: pd.DataFrame) -> pd.DataFrame:
    """Per outcome: how much the 16 messages differ from each other.

    The benchmark separates two kinds of prediction skill — knowing which
    outcomes move at all, and knowing which message moves them.  The second is
    only meaningful where the effects actually spread out, so the spread is
    reported next to the average.
    """
    rows = []
    for outcome, group in table.groupby("outcome", sort=False):
        estimates = group["pp_scale"].to_numpy(dtype=float)
        errors = group["pp_se"].to_numpy(dtype=float)
        observed = float(estimates.var(ddof=1))
        noise = float(np.mean(errors**2))
        rows.append(
            {
                "outcome": outcome,
                "mean_effect_pp": float(estimates.mean()),
                "min_effect_pp": float(estimates.min()),
                "max_effect_pp": float(estimates.max()),
                "sd_effect_pp": float(np.sqrt(observed)),
                # Observed spread minus the average sampling variance: what is
                # left is real between-message variation, not estimation noise.
                "true_sd_effect_pp": float(np.sqrt(max(observed - noise, 0.0))),
                "n_positive": int((estimates > 0).sum()),
                "n_significant_holm": int((group["p_holm"] < 0.05).sum()),
            }
        )
    return pd.DataFrame(rows)
