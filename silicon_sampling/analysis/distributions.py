"""Shape of the responses, and whether they look like people answered them.

The characteristic failure of a silicon sample is not a wrong mean — it is a
degenerate distribution: every respondent picking 50, or every item within a
battery getting the same number.  These diagnostics are aimed squarely at that,
because a sample can reproduce every average in the study and still be worthless
for anything distributional.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def summary(frame: pd.DataFrame, columns) -> pd.DataFrame:
    """Mean, spread and shape per column."""
    from scipy import stats

    rows = []
    for column in columns:
        values = (
            pd.to_numeric(frame[column], errors="coerce").dropna().to_numpy(dtype=float)
        )
        if len(values) == 0:
            continue
        rows.append(
            {
                "variable": column,
                "n": len(values),
                "mean": float(values.mean()),
                "sd": float(values.std(ddof=1)) if len(values) > 1 else float("nan"),
                "median": float(np.median(values)),
                "q25": float(np.percentile(values, 25)),
                "q75": float(np.percentile(values, 75)),
                "min": float(values.min()),
                "max": float(values.max()),
                "skew": float(stats.skew(values)),
                "kurtosis": float(stats.kurtosis(values)),
            }
        )
    return pd.DataFrame(rows)


def degeneracy(frame: pd.DataFrame, columns, scale_max: float = 100.0) -> pd.DataFrame:
    """How concentrated and how round the answers are."""
    rows = []
    for column in columns:
        values = (
            pd.to_numeric(frame[column], errors="coerce").dropna().to_numpy(dtype=float)
        )
        if len(values) == 0:
            continue
        counts = pd.Series(values).value_counts()
        rows.append(
            {
                "variable": column,
                "n": len(values),
                "distinct": int(len(counts)),
                "modal_value": float(counts.index[0]),
                "modal_share": float(counts.iloc[0] / len(values)),
                "share_at_min": float((values == 0).mean()),
                "share_at_mid": float((values == scale_max / 2).mean()),
                "share_at_max": float((values == scale_max).mean()),
                "share_multiple_of_10": float((values % 10 == 0).mean()),
                "share_multiple_of_5": float((values % 5 == 0).mean()),
            }
        )
    return pd.DataFrame(rows)


def cronbach_alpha(frame: pd.DataFrame, items) -> float:
    """Internal consistency of a multi-item scale."""
    data = frame[list(items)].apply(pd.to_numeric, errors="coerce").dropna()
    if data.shape[0] < 3 or data.shape[1] < 2:
        return float("nan")
    k = data.shape[1]
    item_var = data.var(axis=0, ddof=1).sum()
    total_var = data.sum(axis=1).var(ddof=1)
    if total_var == 0:
        return float("nan")
    return float(k / (k - 1) * (1 - item_var / total_var))


def scale_reliabilities(frame: pd.DataFrame, scales: dict) -> pd.DataFrame:
    """Alpha and mean inter-item correlation for every multi-item scale."""
    rows = []
    for name, items in scales.items():
        present = [item for item in items if item in frame.columns]
        if len(present) < 2:
            continue
        data = frame[present].apply(pd.to_numeric, errors="coerce").dropna()
        corr = data.corr().to_numpy()
        off_diagonal = corr[~np.eye(len(present), dtype=bool)]
        rows.append(
            {
                "scale": name,
                "items": len(present),
                "alpha": cronbach_alpha(frame, present),
                "mean_inter_item_r": float(np.nanmean(off_diagonal)),
                "min_inter_item_r": float(np.nanmin(off_diagonal)),
            }
        )
    return pd.DataFrame(rows)


def straightlining(frame: pd.DataFrame, batteries: dict) -> pd.DataFrame:
    """Within-battery variation per respondent.

    A within-respondent SD of zero across a battery means every item in it got
    the same answer.  Some of that is real (people do have flat profiles); a lot
    of it is a model copying its previous line.
    """
    rows = []
    for name, items in batteries.items():
        present = [item for item in items if item in frame.columns]
        if len(present) < 2:
            continue
        data = frame[present].apply(pd.to_numeric, errors="coerce")
        within = data.std(axis=1, ddof=1)
        rows.append(
            {
                "battery": name,
                "items": len(present),
                "mean_within_sd": float(within.mean()),
                "median_within_sd": float(within.median()),
                "share_flat": float((within == 0).mean()),
            }
        )
    return pd.DataFrame(rows)


def correlations(frame: pd.DataFrame, columns) -> pd.DataFrame:
    """Pearson correlation matrix over the given columns."""
    data = frame[list(columns)].apply(pd.to_numeric, errors="coerce")
    return data.corr()


def position_effects(frame: pd.DataFrame, batteries: dict) -> pd.DataFrame:
    """Does the answer drift with an item's position inside its battery?

    A transcript format can induce this even when the content does not: the model
    sees its own earlier answers and may anchor or drift across a run of similar
    questions.  A real battery should show no systematic slope once the items
    themselves are accounted for.
    """
    from scipy import stats

    rows = []
    for name, items in batteries.items():
        present = [item for item in items if item in frame.columns]
        if len(present) < 3:
            continue
        means = [pd.to_numeric(frame[item], errors="coerce").mean() for item in present]
        positions = np.arange(len(present), dtype=float)
        slope, intercept, r, p, stderr = stats.linregress(positions, means)
        rows.append(
            {
                "battery": name,
                "items": len(present),
                "slope_per_item": float(slope),
                "r": float(r),
                "p": float(p),
                "first_item_mean": float(means[0]),
                "last_item_mean": float(means[-1]),
            }
        )
    return pd.DataFrame(rows)
