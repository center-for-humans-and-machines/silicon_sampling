"""Analysis for silicon samples: OLS with robust errors, effects, distributions.

Deliberately dependency-light — numpy, scipy and pandas only.  ``statsmodels`` is
not installed in this container, and the estimators needed here (OLS with HC1
errors, a joint Wald test, Cronbach's alpha) are short enough to state explicitly,
which also makes what the reports claim auditable.
"""

from __future__ import annotations

from .ols import Fit, design_matrix, ols, wald

__all__ = ["Fit", "design_matrix", "ols", "wald"]
