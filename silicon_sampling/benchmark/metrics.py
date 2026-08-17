"""The Silicon Sample Benchmark's scoring metrics, in Python.

Translated from ``R/functions/statistics.R`` of the benchmark repository, which
is the preregistered definition.  The translation is deliberately literal — same
guards, same edge cases, same names — because the whole point of computing these
is that the numbers are comparable to what the benchmark will report.

Two ideas recur and are worth stating once.

**Noise correction.** The human reference is itself an estimate, so a perfect
prediction still correlates imperfectly with it.  Subtracting the reference's
mean sampling variance from its observed spread gives the variance of the *true*
effects, and correlating against that removes the deflation.  The same trick in
reverse corrects the calibration slope for noise in the *predictions*.

**Half credit for zeros.** A predicted effect of exactly zero makes no
directional claim, so it scores 0.5 rather than being dropped.  Dropping zeros
would let a submission shrink its own denominator to the cases it is confident
about; scoring them at chance keeps every pair in play and makes an all-zero
predictor read 50%, which is what a predictor with no directional information
deserves.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def directional_score(estimate_h, estimate_l) -> float:
    """Percentage of pairs whose predicted sign matches, zeros at half credit."""
    h = np.asarray(estimate_h, dtype=float)
    lo = np.asarray(estimate_l, dtype=float)
    keep = ~(np.isnan(h) | np.isnan(lo))
    if not keep.any():
        return float("nan")
    h, lo = h[keep], lo[keep]
    score = np.where(lo == 0, 0.5, (np.sign(h) == np.sign(lo)).astype(float))
    return float(score.mean() * 100)


def pearson(x, y) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    keep = ~(np.isnan(x) | np.isnan(y))
    if keep.sum() < 3 or np.std(x[keep]) == 0 or np.std(y[keep]) == 0:
        return float("nan")
    return float(np.corrcoef(x[keep], y[keep])[0, 1])


def spearman(x, y) -> float:
    from scipy import stats

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    keep = ~(np.isnan(x) | np.isnan(y))
    if keep.sum() < 3:
        return float("nan")
    return float(stats.spearmanr(x[keep], y[keep]).statistic)


def pearson_within_outcome(pairs: pd.DataFrame) -> float:
    """Correlation after centring both sides on their per-outcome means.

    The pooled correlation credits two different things at once: knowing which
    outcomes move at all, and knowing which message moves them.  Centring within
    outcome removes the first, so what survives is message-level skill, and the
    gap between the two says how much of the pooled score came from knowing the
    outcomes rather than the messages.
    """
    if "outcome" not in pairs.columns:
        return float("nan")
    frame = pairs.dropna(subset=["estimate_h", "estimate_l"]).copy()
    if len(frame) < 3:
        return float("nan")
    frame["h_c"] = frame["estimate_h"] - frame.groupby("outcome")[
        "estimate_h"
    ].transform("mean")
    frame["l_c"] = frame["estimate_l"] - frame.groupby("outcome")[
        "estimate_l"
    ].transform("mean")
    if frame["h_c"].std() == 0 or frame["l_c"].std() == 0:
        return float("nan")
    return pearson(frame["h_c"], frame["l_c"])


def adjusted_metrics(pairs: pd.DataFrame) -> dict:
    """Correlation and RMSE with the reference's own sampling noise removed."""
    needed = {"estimate_h", "estimate_l", "se_h"}
    if not needed <= set(pairs.columns):
        return {
            "pearson_adj": float("nan"),
            "rmse_adj": float("nan"),
            "rmse_adj_at_floor": False,
        }
    frame = pairs.dropna(subset=["estimate_h", "estimate_l", "se_h"])
    if len(frame) < 3:
        return {
            "pearson_adj": float("nan"),
            "rmse_adj": float("nan"),
            "rmse_adj_at_floor": False,
        }

    h = frame["estimate_h"].to_numpy(float)
    lo = frame["estimate_l"].to_numpy(float)
    se = frame["se_h"].to_numpy(float)
    var_true = h.var(ddof=1) - np.mean(se**2)
    mse_true = np.mean((h - lo) ** 2) - np.mean(se**2)

    if var_true > 0 and lo.std(ddof=1) > 0:
        adjusted = float(
            np.cov(lo, h, ddof=1)[0, 1] / (lo.std(ddof=1) * np.sqrt(var_true))
        )
        adjusted = max(-1.0, min(1.0, adjusted))
    else:
        adjusted = float("nan")
    return {
        "pearson_adj": adjusted,
        "rmse_adj": float(np.sqrt(max(mse_true, 0.0))),
        "rmse_adj_at_floor": bool(mse_true <= 0),
    }


def pooled_metrics(pairs: pd.DataFrame, include_rmse: bool = True) -> dict:
    """Estimate-based metrics pooled across outcomes."""
    adjusted = adjusted_metrics(pairs)
    out = {
        "n_pairs": int(pairs[["estimate_h", "estimate_l"]].dropna().shape[0]),
        "directional_pct": directional_score(pairs["estimate_h"], pairs["estimate_l"]),
        "spearman_rho": spearman(pairs["estimate_h"], pairs["estimate_l"]),
        "pearson_r": pearson(pairs["estimate_h"], pairs["estimate_l"]),
        "pearson_within": pearson_within_outcome(pairs),
        "pearson_adj": adjusted["pearson_adj"],
    }
    if include_rmse:
        frame = pairs.dropna(subset=["estimate_h", "estimate_l"])
        out["rmse"] = float(
            np.sqrt(np.mean((frame["estimate_h"] - frame["estimate_l"]) ** 2))
        )
        out["rmse_adj"] = adjusted["rmse_adj"]
        out["rmse_adj_at_floor"] = adjusted["rmse_adj_at_floor"]
    return out


def run_calibration_pooled(pairs: pd.DataFrame) -> dict:
    """``ATE_human = alpha + beta * ATE_predicted``, pooled across outcomes.

    ``beta_adj`` corrects the slope for sampling noise in the *predictions*.
    Noise there — never in the reference, which only widens the residuals — drags
    the slope toward zero by the reliability of the predictions, so a noisy but
    well-calibrated submission prints beta < 1 without exaggerating anything.
    """
    frame = pairs.dropna(subset=["estimate_h", "estimate_l"])
    if len(frame) < 3:
        return {"alpha": float("nan"), "beta": float("nan"), "beta_adj": float("nan")}
    h = frame["estimate_h"].to_numpy(float)
    lo = frame["estimate_l"].to_numpy(float)
    design = np.column_stack([np.ones_like(lo), lo])
    alpha, beta = np.linalg.lstsq(design, h, rcond=None)[0]

    beta_adj = float("nan")
    if "se_l" in frame.columns and frame["se_l"].notna().any():
        variance = np.nanvar(lo, ddof=1)
        reliability = (
            1 - np.nanmean(frame["se_l"].to_numpy(float) ** 2) / variance
            if variance > 0
            else 0.0
        )
        if reliability > 0:
            beta_adj = float(beta / reliability)
    return {"alpha": float(alpha), "beta": float(beta), "beta_adj": beta_adj}


def signed_metrics(pairs: pd.DataFrame) -> dict:
    """Directional agreement, Spearman and Pearson for estimate-only pairs."""
    out = {
        "n_pairs": int(pairs[["estimate_h", "estimate_l"]].dropna().shape[0]),
        "directional_pct": directional_score(pairs["estimate_h"], pairs["estimate_l"]),
        "spearman_rho": spearman(pairs["estimate_h"], pairs["estimate_l"]),
        "pearson_r": pearson(pairs["estimate_h"], pairs["estimate_l"]),
    }
    if "se_h" in pairs.columns:
        out["pearson_adj"] = adjusted_metrics(pairs)["pearson_adj"]
    return out


def paired_cluster_bootstrap(
    baseline: pd.DataFrame,
    contender: pd.DataFrame,
    statistic,
    cluster: str = "condition",
    draws: int = 2000,
    seed: int = 42,
) -> dict:
    """Interval on ``statistic(contender) - statistic(baseline)``, cluster-paired.

    Two models scored against the same humans do not have independent errors:
    they answered the *same* instrument about the *same* interventions, so most of
    what moves one score moves the other.  Bootstrapping each separately and
    eyeballing whether the intervals overlap throws that pairing away and will
    call a real difference inconclusive — the difference is estimated far more
    precisely than either level.  So each draw resamples one set of clusters and
    scores *both* submissions on it.

    Returns ``<metric>_delta`` (the point difference), its interval, and
    ``<metric>_p_gt0``, the share of draws in which the contender scored higher —
    a one-sided bootstrap p-value in either tail.
    """
    rng = np.random.default_rng(seed)
    shared = sorted(set(baseline[cluster].unique()) & set(contender[cluster].unique()))
    if len(shared) < 2:
        return {}
    left = {name: group for name, group in baseline.groupby(cluster) if name in shared}
    right = {
        name: group for name, group in contender.groupby(cluster) if name in shared
    }

    deltas: list[dict] = []
    for _ in range(draws):
        picked = [shared[i] for i in rng.choice(len(shared), len(shared), replace=True)]
        try:
            before = statistic(pd.concat([left[k] for k in picked], ignore_index=True))
            after = statistic(pd.concat([right[k] for k in picked], ignore_index=True))
        except Exception:  # pragma: no cover - a degenerate resample
            continue
        deltas.append(
            {
                key: after[key] - before[key]
                for key in before
                if isinstance(before.get(key), (int, float))
                and not isinstance(before.get(key), bool)
                and isinstance(after.get(key), (int, float))
            }
        )
    if not deltas:
        return {}

    point_before, point_after = statistic(baseline), statistic(contender)
    frame = pd.DataFrame(deltas)
    out: dict = {"n_clusters": len(shared)}
    for column in frame.columns:
        values = pd.to_numeric(frame[column], errors="coerce").dropna()
        if len(values) < 10:
            continue
        out[f"{column}_delta"] = float(point_after[column] - point_before[column])
        out[f"{column}_delta_lo"] = float(values.quantile(0.025))
        out[f"{column}_delta_hi"] = float(values.quantile(0.975))
        out[f"{column}_p_gt0"] = float((values > 0).mean())
    return out


def cluster_bootstrap(
    pairs: pd.DataFrame,
    statistic,
    cluster: str = "condition",
    draws: int = 2000,
    seed: int = 42,
) -> dict:
    """95% intervals by resampling *clusters*, not rows.

    The outcomes within one intervention share that intervention's draw, so
    resampling rows would treat correlated pairs as independent and produce
    intervals that are far too narrow.
    """
    rng = np.random.default_rng(seed)
    clusters = pairs[cluster].unique()
    if len(clusters) < 2:
        return {}
    by_cluster = {name: group for name, group in pairs.groupby(cluster)}
    keys = list(by_cluster)
    samples: list[dict] = []
    for _ in range(draws):
        picked = rng.choice(len(keys), size=len(keys), replace=True)
        resampled = pd.concat([by_cluster[keys[i]] for i in picked], ignore_index=True)
        try:
            samples.append(statistic(resampled))
        except Exception:  # pragma: no cover - a degenerate resample
            continue
    if not samples:
        return {}
    frame = pd.DataFrame(samples)
    out = {}
    for column in frame.columns:
        # Flags like rmse_adj_at_floor ride along in the statistic dict; a
        # quantile of a boolean is meaningless (and numpy refuses it).
        if pd.api.types.is_bool_dtype(frame[column]):
            continue
        values = pd.to_numeric(frame[column], errors="coerce").dropna()
        if len(values) < 10:
            continue
        out[f"{column}_lo"] = float(values.quantile(0.025))
        out[f"{column}_hi"] = float(values.quantile(0.975))
    out["n_clusters"] = int(len(clusters))
    return out
