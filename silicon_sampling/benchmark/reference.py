"""The scoring frame: split the humans, estimate effects, pair them up.

The benchmark scores every submission against **Human 1**, one half of the human
sample drawn on a fixed seed.  **Human 2**, the other half, predicts Human 1
exactly as a submission would, and its score is the *human replication
reference* — what a fresh human sample of that size achieves.  That row is what
makes an absolute correlation readable: on its own, "r = 0.4" says nothing; next
to a real replication's r it says a great deal.

The reference is not a ceiling.  Human 2 carries the sampling noise of a half
sample, so a synthetic sample built from many more respondents can legitimately
score above it — and if it does, that is a result, not an artefact.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..analysis.ols import benjamini_hochberg, design_matrix, ols


def half_split(
    frame: pd.DataFrame, seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split respondents into the reference half and the replication half."""
    rng = np.random.default_rng(seed)
    first = rng.random(len(frame)) < 0.5
    return frame.loc[first].copy(), frame.loc[~first].copy()


def treatment_effects(
    frame: pd.DataFrame,
    outcomes: dict[str, float],
    control: str,
    condition_col: str = "condition",
    robust: str = "HC2",
) -> pd.DataFrame:
    """ATE of each condition against the shared control, per outcome.

    ``outcomes`` maps an outcome name to its scale range, so every estimate can be
    reported in percentage points of that range — the unit the benchmark pools
    over. Effects are converted at this point, not later, so that everything
    downstream is already on one footing.
    """
    rows = []
    for outcome, scale in outcomes.items():
        data = frame[[condition_col, outcome]].dropna()
        if data.empty or data[condition_col].nunique() < 2:
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
            condition = name[len(condition_col) + 1 : -1]
            rows.append(
                {
                    "outcome": outcome,
                    "condition": condition,
                    "n": int((data[condition_col] == condition).sum()),
                    "estimate": term["estimate"] / scale * 100,
                    "se": term["se"] / scale * 100,
                    "estimate_raw": term["estimate"],
                    "p": term["p"],
                }
            )
    table = pd.DataFrame(rows)
    if not table.empty:
        table["p_bh"] = benjamini_hochberg(table["p"].tolist())
    return table


def ate_pairs(reference: pd.DataFrame, prediction: pd.DataFrame) -> pd.DataFrame:
    """Join reference and predicted effects into the frame every metric consumes.

    Column names follow the benchmark's: ``_h`` is the human reference side,
    ``_l`` the prediction being scored.
    """
    left = reference.rename(
        columns={"estimate": "estimate_h", "se": "se_h", "n": "n_h"}
    )
    right = prediction.rename(
        columns={"estimate": "estimate_l", "se": "se_l", "n": "n_l"}
    )
    keep_left = ["outcome", "condition", "estimate_h", "se_h", "n_h"]
    keep_right = ["outcome", "condition", "estimate_l", "se_l", "n_l"]
    return left[keep_left].merge(
        right[keep_right], on=["outcome", "condition"], how="inner"
    )


def null_prediction(reference: pd.DataFrame, value: float = 0.0) -> pd.DataFrame:
    """The "no effect" baseline: every effect predicted as zero."""
    out = reference.copy()
    out["estimate"] = value
    out["se"] = np.nan
    return out


def all_positive_prediction(
    reference: pd.DataFrame, value: float = 1.0
) -> pd.DataFrame:
    """The "all positive" baseline.

    Worth carrying because most megastudy effects come out positive, so a
    predictor that says "everything works a bit" beats a coin flip on directional
    agreement without knowing anything.
    """
    out = reference.copy()
    out["estimate"] = value
    out["se"] = np.nan
    return out
