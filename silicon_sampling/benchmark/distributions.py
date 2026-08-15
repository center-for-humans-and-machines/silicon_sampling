"""Distribution-shape metrics: does the sample reproduce the spread, not just the mean?

A synthetic sample can land every average in a study and still be useless, because
the respondents are all the same person.  These four metrics are what the
benchmark uses to catch that, and they disagree with each other usefully: the
variance ratio is a single moment, OVL is a smoothed area, KS is the worst single
point of the CDF gap, and W1 is the whole CDF gap integrated.

``compute_ovl`` reproduces R's ``density()`` rather than using SciPy's default,
because the two choose bandwidth differently and OVL is sensitive to it.  R uses
Silverman's rule of thumb (``bw.nrd0``); SciPy uses Scott's.
"""

from __future__ import annotations

import numpy as np


def bw_nrd0(x: np.ndarray) -> float:
    """R's ``bw.nrd0``: the bandwidth ``density()`` uses by default."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 2:
        return 1.0
    sd = np.std(x, ddof=1)
    iqr = np.subtract(*np.percentile(x, [75, 25]))
    spread = min(sd, iqr / 1.349) if iqr > 0 else sd
    if spread <= 0:
        spread = abs(x[0]) if x[0] != 0 else 1.0
    return float(0.9 * spread * n ** (-0.2))


def _density(x: np.ndarray, grid: np.ndarray) -> np.ndarray:
    bandwidth = bw_nrd0(x)
    if bandwidth <= 0:
        return np.zeros_like(grid)
    z = (grid[:, None] - x[None, :]) / bandwidth
    return np.exp(-0.5 * z**2).sum(axis=1) / (len(x) * bandwidth * np.sqrt(2 * np.pi))


def compute_ovl(
    x, y, n_grid: int = 512, lo: float | None = None, hi: float | None = None
) -> float:
    """Overlapping coefficient: the area shared by two kernel density estimates."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 2 or len(y) < 2:
        return float("nan")
    low = min(x.min(), y.min()) if lo is None else lo
    high = max(x.max(), y.max()) if hi is None else hi
    if low == high:
        return float("nan")
    grid = np.linspace(low, high, n_grid)
    return float(
        np.minimum(_density(x, grid), _density(y, grid)).sum() * (high - low) / n_grid
    )


def _ecdf(sample: np.ndarray, grid: np.ndarray) -> np.ndarray:
    return np.searchsorted(np.sort(sample), grid, side="right") / len(sample)


def compute_w1(
    x, y, n_grid: int = 512, lo: float | None = None, hi: float | None = None
) -> float:
    """Wasserstein-1 distance, in scale points.

    Bandwidth-free companion to OVL: the integral of the absolute difference
    between the two ECDFs, which unlike a kernel estimate cannot leak density
    past the ends of the scale.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) == 0 or len(y) == 0:
        return float("nan")
    low = min(x.min(), y.min()) if lo is None else lo
    high = max(x.max(), y.max()) if hi is None else hi
    if low == high:
        return 0.0
    grid = np.linspace(low, high, n_grid)
    return float(np.abs(_ecdf(x, grid) - _ecdf(y, grid)).mean() * (high - low))


def compute_ks(x, y) -> float:
    """Kolmogorov-Smirnov statistic: the largest single gap between the ECDFs."""
    from scipy import stats

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 2 or len(y) < 2:
        return float("nan")
    return float(stats.ks_2samp(x, y).statistic)


def variance_ratio(x, y) -> float:
    """Predicted variance over reference variance; 1 is perfect."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 2 or len(y) < 2:
        return float("nan")
    reference = x.var(ddof=1)
    return float(y.var(ddof=1) / reference) if reference > 0 else float("nan")


def compare_distributions(human, synthetic, lo: float = 0.0, hi: float = 100.0) -> dict:
    """All four shape metrics for one condition x outcome cell."""
    human = np.asarray([v for v in human if not np.isnan(v)], dtype=float)
    synthetic = np.asarray([v for v in synthetic if not np.isnan(v)], dtype=float)
    return {
        "n_human": len(human),
        "n_synthetic": len(synthetic),
        "mean_human": float(human.mean()) if len(human) else float("nan"),
        "mean_synthetic": float(synthetic.mean()) if len(synthetic) else float("nan"),
        "sd_human": float(human.std(ddof=1)) if len(human) > 1 else float("nan"),
        "sd_synthetic": (
            float(synthetic.std(ddof=1)) if len(synthetic) > 1 else float("nan")
        ),
        "variance_ratio": variance_ratio(human, synthetic),
        "ovl": compute_ovl(human, synthetic, lo=lo, hi=hi),
        "ks": compute_ks(human, synthetic),
        "w1": compute_w1(human, synthetic, lo=lo, hi=hi),
    }
