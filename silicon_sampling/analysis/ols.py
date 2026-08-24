"""Least squares with heteroskedasticity-consistent standard errors.

HC1 is the default because treatment groups in this design can differ in variance
as well as in mean — a message that polarises raises the variance even when it
barely moves the mean — and classical errors would misstate the precision of
exactly those effects.

    cov_HC1 = (X'X)^-1 X' diag(e_i^2) X (X'X)^-1 . n / (n - k)

``classical`` is kept alongside the robust variants only because two of the
benchmark's preregistered functions are plain ``lm()`` calls whose reported
intervals are the textbook ones — the stereotyping coefficients of
``compare_demographic_predictability()``.  Reproducing their numbers means
reproducing their standard errors, robustness be damned.

**Reference levels are the other half of this module's job.** A dummy-coded
contrast is meaningless without knowing what it is a contrast *against*, and the
default in R — and here — is the factor's first level.  Coercing a character
column with :func:`sorted` picks the alphabetically first level instead, which
for five of the benchmark's six moderators is the wrong one: it would silently
report the gap to "Bachelor's degree" where the codebook asks for the gap to
"Less than high school", and every interaction estimate and stereotyping
coefficient built on it would be a different quantity that still looks
plausible.  :func:`design_matrix` therefore takes the intended level order
(``levels=``) or the reference outright (``reference=``), and honours a pandas
categorical's own category order when it is given one.  A ``reference=`` level
that does not occur in the data is an error rather than a fallback: omitting no
dummy leaves the design rank deficient, :func:`ols` solves it anyway through
``pinv``, and the coefficients that come back are neither identified nor
obviously wrong — the exact silent failure a mislabeled control arm produces.

**Rank deficiency is the other silent failure**, and it survives even a correct
reference: an empty cell of a saturated interaction design aliases two columns,
and the minimum-norm solution splits the effect between them and reports both
with small standard errors.  R answers ``NA`` for the aliased coefficient and
``broom::tidy`` drops the row, so :func:`aliased_columns` finds the same columns
in the same left-to-right order and lets the caller report them missing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import pandas as pd


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
        from scipy import stats

        return 2 * stats.t.sf(np.abs(self.t), df=max(self.n - self.k, 1))

    def ci(self, level: float = 0.95) -> tuple[np.ndarray, np.ndarray]:
        from scipy import stats

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
    of each treatment arm, as here.  ``classical`` (equivalently ``none``) is the
    textbook ``sigma^2 (X'X)^-1``, for the two preregistered analyses that report
    plain ``lm()`` intervals.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n, k = X.shape
    xtx_inv = np.linalg.pinv(X.T @ X)
    beta = xtx_inv @ X.T @ y
    resid = y - X @ beta
    kind = robust.upper()
    if kind == "HC2":
        leverage = np.einsum("ij,jk,ik->i", X, xtx_inv, X)
        weights = resid**2 / np.clip(1 - leverage, 1e-12, None)
        cov = xtx_inv @ ((X * weights[:, None]).T @ X) @ xtx_inv
    elif kind in {"CLASSICAL", "NONE", "CONST", "IID"}:
        sigma2 = float(resid @ resid) / max(n - k, 1)
        cov = xtx_inv * sigma2
    else:
        meat = (X * (resid**2)[:, None]).T @ X
        cov = xtx_inv @ meat @ xtx_inv * (n / max(n - k, 1))
    centred = y - y.mean()
    ss_tot = float(centred @ centred)
    r2 = 1.0 - float(resid @ resid) / ss_tot if ss_tot > 0 else 0.0
    return Fit(names=list(names), beta=beta, cov=cov, n=n, k=k, resid=resid, r2=r2)


def level_order(values: Sequence, levels: Sequence[str] | None = None) -> list[str]:
    """The level order a factor built from ``values`` should have.

    ``levels`` is the intended (codebook) order: its levels come first, in that
    order, and anything observed but unlisted follows alphabetically — R's
    ``fct_relevel(factor(x), intersect(codebook, unique(x)))``, which is what
    ``align_submission_levels()`` does to every submission before fitting.
    Failing that, a pandas categorical's own category order is honoured, and a
    plain column falls back to alphabetical.
    """
    observed = {str(value) for value in values}

    def keep(candidates: Sequence) -> list[str]:
        # Duplicates in the intended order are normal (a caller naming the
        # control arm first, then listing every arm) and must not duplicate a
        # dummy column.
        seen: list[str] = []
        for candidate in candidates:
            text = str(candidate)
            if text in observed and text not in seen:
                seen.append(text)
        return seen

    if levels is not None:
        listed = keep(levels)
        return listed + sorted(observed - set(listed))
    dtype = getattr(values, "dtype", None)
    if isinstance(dtype, pd.CategoricalDtype):
        listed = keep(dtype.categories)
        return listed + sorted(observed - set(listed))
    return sorted(observed)


def design_matrix(
    columns: dict[str, Sequence],
    reference: Mapping[str, str] | None = None,
    levels: Mapping[str, Sequence[str]] | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Intercept plus dummy-coded categoricals.

    ``columns`` maps a variable name to its per-row values.  ``levels`` gives the
    intended level order per variable — the codebook order for the benchmark's
    moderators — and its first observed level becomes the omitted reference, the
    way R treats a factor.  ``reference`` overrides that choice outright, and
    raises if the level it names is not in the data.  With neither, an ordered
    categorical column supplies its own order and a plain column falls back to
    alphabetical, which is right only by accident.
    """
    reference = reference or {}
    levels = levels or {}
    n = len(next(iter(columns.values())))
    blocks = [np.ones((n, 1))]
    names = ["(Intercept)"]
    for variable, values in columns.items():
        order = level_order(values, levels.get(variable))
        base = reference.get(variable)
        if base is not None and str(base) not in order:
            raise ValueError(
                f"{variable}: the reference level {str(base)!r} does not occur in "
                f"the data, whose levels are {order} — every level would get a "
                "dummy and the fit would be rank deficient, so this halts instead "
                "(check for a mislabeled condition)"
            )
        if base is None:
            base = order[0] if order else None
        # Compared as a numpy array of strings rather than row by row: a saturated
        # interaction design has a hundred columns and ten thousand rows, and the
        # per-row Python comparison is what makes a moderator fit slow.
        as_text = np.asarray([str(value) for value in values], dtype=object)
        for level in order:
            if level == base:
                continue
            blocks.append((as_text == level).astype(float)[:, None])
            names.append(f"{variable}[{level}]")
    return np.hstack(blocks), names


def aliased_columns(X: np.ndarray, tol: float = 1e-10) -> list[int]:
    """Columns that add nothing to the span of the columns to their left.

    An empty cell in a saturated design makes two columns carry the same
    information, and least squares then has no unique answer: ``pinv`` returns the
    minimum-norm one, which splits the effect between the aliased pair and reports
    each half with a small standard error.  R's ``lm`` instead factors the design
    with a rank-revealing QR, answers ``NA`` for the column its pivoting leaves
    behind, and ``broom::tidy`` drops that row entirely.  Which of the pair goes
    missing is therefore part of the output, and it is decided by position: the
    sweep here runs left to right and keeps a column only if it is independent of
    the ones already kept, so in a ``condition * moderator`` design the main effect
    survives and the interaction is the term reported missing — R's answer.

    Detection runs on the Gram matrix (Gram-Schmidt in the ``X'X`` metric): O(k^3)
    in the hundred-odd columns of a saturated interaction design instead of a
    second factorisation of its ten thousand rows.  ``tol`` is relative to each
    column's own squared norm, and the two cases are nowhere near each other — an
    exactly aliased dummy leaves a residual at machine epsilon while the smallest
    genuine cell leaves one of order 1/n.
    """
    matrix = np.asarray(X, dtype=float)
    gram = matrix.T @ matrix
    k = gram.shape[0]
    # Row m holds the inner products of the m-th orthonormalised kept column with
    # every column of X, which is all the projection step needs.
    projections = np.zeros((k, k))
    kept = 0
    aliased: list[int] = []
    for column in range(k):
        loadings = projections[:kept, column]
        residual = gram[column, column] - float(loadings @ loadings)
        if residual <= tol * gram[column, column]:
            aliased.append(column)
            continue
        norm = float(np.sqrt(residual))
        projections[kept] = (gram[column] - loadings @ projections[:kept]) / norm
        kept += 1
    return aliased


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
    from scipy import stats

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
