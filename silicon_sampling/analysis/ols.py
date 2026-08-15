"""Least squares with heteroskedasticity-consistent standard errors.

HC1 is the default because treatment groups in this design can differ in variance
as well as in mean — a message that polarises raises the variance even when it
barely moves the mean — and classical errors would misstate the precision of
exactly those effects.

    cov_HC1 = (X'X)^-1 X' diag(e_i^2) X (X'X)^-1 . n / (n - k)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy import stats


@dataclass
class Fit:
    """A fitted linear model."""

    names: list[str]
    beta: np.ndarray
    cov: np.ndarray
    n: int
    k: int
    resid: np.ndarray
    r2: float

    @property
    def se(self) -> np.ndarray:
        return np.sqrt(np.diag(self.cov))

    @property
    def t(self) -> np.ndarray:
        with np.errstate(divide="ignore", invalid="ignore"):
            return self.beta / self.se

    @property
    def p(self) -> np.ndarray:
        return 2 * stats.t.sf(np.abs(self.t), df=max(self.n - self.k, 1))

    def ci(self, level: float = 0.95) -> tuple[np.ndarray, np.ndarray]:
        crit = stats.t.ppf(0.5 + level / 2, df=max(self.n - self.k, 1))
        return self.beta - crit * self.se, self.beta + crit * self.se

    def index(self, name: str) -> int:
        return self.names.index(name)

    def term(self, name: str) -> dict:
        """Everything about one coefficient, as a plain dict."""
        i = self.index(name)
        low, high = self.ci()
        return {
            "term": name,
            "estimate": float(self.beta[i]),
            "se": float(self.se[i]),
            "t": float(self.t[i]),
            "p": float(self.p[i]),
            "conf_low": float(low[i]),
            "conf_high": float(high[i]),
        }


def ols(X: np.ndarray, y: np.ndarray, names: Sequence[str], robust: str = "HC1") -> Fit:
    """Fit ``y ~ X`` (X must already contain an intercept column).

    ``robust`` picks the heteroskedasticity-consistent variant: ``HC1`` scales the
    squared residuals by ``n/(n-k)``, ``HC2`` divides each by ``1 - h_ii``, its own
    leverage. HC2 is what the Voelkel study and the benchmark preregistration use,
    and it matters where cell sizes are uneven — a control arm five times the size
    of each treatment arm, as here.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n, k = X.shape
    xtx_inv = np.linalg.pinv(X.T @ X)
    beta = xtx_inv @ X.T @ y
    resid = y - X @ beta
    if robust.upper() == "HC2":
        leverage = np.einsum("ij,jk,ik->i", X, xtx_inv, X)
        weights = resid**2 / np.clip(1 - leverage, 1e-12, None)
        cov = xtx_inv @ ((X * weights[:, None]).T @ X) @ xtx_inv
    else:
        meat = (X * (resid**2)[:, None]).T @ X
        cov = xtx_inv @ meat @ xtx_inv * (n / max(n - k, 1))
    centred = y - y.mean()
    ss_tot = float(centred @ centred)
    r2 = 1.0 - float(resid @ resid) / ss_tot if ss_tot > 0 else 0.0
    return Fit(names=list(names), beta=beta, cov=cov, n=n, k=k, resid=resid, r2=r2)


def design_matrix(
    columns: dict[str, Sequence], reference: dict[str, str] | None = None
) -> tuple[np.ndarray, list[str]]:
    """Intercept plus dummy-coded categoricals.

    ``columns`` maps a variable name to its per-row values; ``reference`` picks the
    omitted level per variable (default: the first level seen, sorted).
    """
    reference = reference or {}
    n = len(next(iter(columns.values())))
    blocks = [np.ones((n, 1))]
    names = ["(Intercept)"]
    for variable, values in columns.items():
        values = list(values)
        levels = sorted({str(value) for value in values})
        base = reference.get(variable, levels[0])
        for level in levels:
            if level == base:
                continue
            blocks.append(
                np.array([[1.0 if str(value) == level else 0.0] for value in values])
            )
            names.append(f"{variable}[{level}]")
    return np.hstack(blocks), names


def interaction(
    left: np.ndarray,
    left_names: Sequence[str],
    right: np.ndarray,
    right_names: Sequence[str],
) -> tuple[np.ndarray, list[str]]:
    """Products of two dummy blocks, skipping each block's intercept column."""
    blocks, names = [], []
    for i, left_name in enumerate(left_names):
        if left_name == "(Intercept)":
            continue
        for j, right_name in enumerate(right_names):
            if right_name == "(Intercept)":
                continue
            blocks.append((left[:, i] * right[:, j])[:, None])
            names.append(f"{left_name}:{right_name}")
    if not blocks:
        return np.zeros((left.shape[0], 0)), []
    return np.hstack(blocks), names


def wald(fit: Fit, terms: Sequence[str]) -> dict:
    """Joint test that every coefficient in ``terms`` is zero."""
    idx = [fit.index(name) for name in terms if name in fit.names]
    if not idx:
        return {"chi2": 0.0, "df": 0, "p": 1.0}
    beta = fit.beta[idx]
    cov = fit.cov[np.ix_(idx, idx)]
    chi2 = float(beta @ np.linalg.pinv(cov) @ beta)
    df = len(idx)
    return {"chi2": chi2, "df": df, "p": float(stats.chi2.sf(chi2, df))}


def holm(pvalues: Sequence[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values, in the input order."""
    order = sorted(range(len(pvalues)), key=lambda i: pvalues[i])
    m = len(pvalues)
    adjusted = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * pvalues[i])
        adjusted[i] = min(1.0, running)
    return adjusted


def benjamini_hochberg(pvalues: Sequence[float]) -> list[float]:
    """BH-adjusted p-values (FDR), in the input order."""
    m = len(pvalues)
    order = sorted(range(m), key=lambda i: pvalues[i], reverse=True)
    adjusted = [0.0] * m
    running = 1.0
    for rank, i in enumerate(order):
        running = min(running, pvalues[i] * m / (m - rank))
        adjusted[i] = min(1.0, running)
    return adjusted
