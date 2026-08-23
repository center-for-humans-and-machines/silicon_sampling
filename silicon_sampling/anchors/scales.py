"""Categorical survey answers into slider points, and the size of the lie.

Every external anchor this package can offer arrives on a Likert scale with three
to five labelled options, and every Pfänder outcome is a 0-100 slider.  There is
no assumption-free way across that gap, so the honest move is to make the
assumption explicit, name a second defensible assumption, and report how far
apart the two land — because that distance is the anchor's *irreducible*
uncertainty, and it turns out to be the same order of magnitude as the error at
which anchoring stops paying (see :mod:`silicon_sampling.anchors.validate`).

**The default conversion is linear on the response options**: option 1 goes to 0,
option *k* goes to 100, everything in between at equal spacing, so a 1-5 Likert
becomes 0/25/50/75/100.  This assumes respondents treat the labelled categories
as equally spaced *and* as centred on the endpoints of the slider — that "Somewhat
expert" out of five options means the same thing as 75 on a 0-100 bar.  That is an
assumption about how people read scales, not a fact about the data, and nothing
in either source can test it.

**The competing conversion is the bin midpoint**: each option covers an
equal-width band of the latent scale and is scored at that band's centre, giving
10/30/50/70/90 for five options.  This is what one would use if the categories
were a coarsening of a continuous response rather than a scale in their own
right, and it is equally defensible a priori.

The two disagree by exactly ``(50 - mean) / k``: a converted mean sitting on the
scale midpoint is conversion-proof, and one sitting near an end is not.  For the
TISP trust battery (12 items, k=5, linear mean 71.5) that is 4.3 points, which is
why :mod:`silicon_sampling.anchors.levels` reports both and why the report treats
"about five points" as the floor on how well any Likert-sourced anchor can be
known.

**Standard errors** use Kish's effective sample size, ``(sum w)^2 / sum w^2``,
which charges for the weights' unequal spread but not for clustering or
stratification.  Neither source publishes the design variables needed for a
proper design-based interval, so these SEs are optimistic by whatever the design
effect beyond weighting is — a caveat that matters little here, since the SEs come
out near 0.5-1.2 points and the conversion ambiguity above is four to eight times
larger.

**Dispersion** is reported twice for the same reason.  A five-option answer
converted to sliders has all its mass on five spikes 25 points apart, and
grouping a continuous variable that coarsely inflates its variance by about
``h^2/12`` (Sheppard's correction, ``h`` the option spacing).  So the raw
converted SD overstates the slider SD it stands in for, and
:func:`sheppard_sd` gives the corrected figure a dispersion calibration would
actually want.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Weighted:
    """A weighted first and second moment, with the honesty of its own SE."""

    mean: float
    sd: float
    se: float
    n: int
    n_effective: float

    @property
    def as_dict(self) -> dict[str, float]:
        return {
            "mean": self.mean,
            "sd": self.sd,
            "se": self.se,
            "n": self.n,
            "n_effective": self.n_effective,
        }


NOT_MEASURED = Weighted(
    mean=float("nan"),
    sd=float("nan"),
    se=float("nan"),
    n=0,
    n_effective=float("nan"),
)


def to_slider(
    values: pd.Series,
    options: int,
    scale: float = 100.0,
    missing_codes: tuple[float, ...] = (),
) -> pd.Series:
    """Linear on the response options: option 1 to 0, option ``options`` to ``scale``."""
    numeric = pd.to_numeric(values, errors="coerce")
    if missing_codes:
        numeric = numeric.where(~numeric.isin(list(missing_codes)))
    numeric = numeric.where((numeric >= 1) & (numeric <= options))
    return (numeric - 1.0) / (options - 1.0) * scale


def to_slider_bin_midpoint(
    values: pd.Series,
    options: int,
    scale: float = 100.0,
    missing_codes: tuple[float, ...] = (),
) -> pd.Series:
    """The competing conversion: each option at the centre of an equal-width band."""
    numeric = pd.to_numeric(values, errors="coerce")
    if missing_codes:
        numeric = numeric.where(~numeric.isin(list(missing_codes)))
    numeric = numeric.where((numeric >= 1) & (numeric <= options))
    return (numeric - 0.5) / options * scale


def conversion_gap(mean_linear: float, options: int, scale: float = 100.0) -> float:
    """How far the bin-midpoint conversion sits from the linear one, at this mean.

    Closed form rather than recomputed, because it says something worth saying out
    loud: the disagreement is ``(midpoint - mean) / k`` and nothing else.  It does
    not shrink with sample size, so it is a floor on the anchor's error that no
    amount of survey data removes.
    """
    return (scale / 2.0 - mean_linear) / options


def sheppard_sd(sd: float, options: int, scale: float = 100.0) -> float:
    """The converted SD with the grouping inflation removed.

    ``h`` is the spacing between converted options; grouping a continuous variable
    into bins of width ``h`` adds ``h^2/12`` to its variance, so subtracting that
    estimates the SD of the slider distribution the anchor stands in for.  Returns
    0 rather than a complex number when the correction would overshoot, which
    happens only for an item so concentrated that its spread *is* the granularity.
    """
    spacing = scale / (options - 1.0)
    corrected = sd**2 - spacing**2 / 12.0
    return float(np.sqrt(corrected)) if corrected > 0 else 0.0


def weighted_moments(values: pd.Series, weights: pd.Series) -> Weighted:
    """Weighted mean, SD and SE over the rows where both are present."""
    numeric = pd.to_numeric(values, errors="coerce")
    weight = pd.to_numeric(weights, errors="coerce")
    keep = numeric.notna() & weight.notna() & (weight > 0)
    if keep.sum() < 2:
        return NOT_MEASURED
    x = numeric[keep].to_numpy(float)
    w = weight[keep].to_numpy(float)
    mean = float(np.average(x, weights=w))
    # The n/(n-1) factor keeps this comparable to an unweighted ddof=1 SD, which
    # is what every other dispersion figure in this repository is.
    variance = float(np.average((x - mean) ** 2, weights=w)) * len(x) / (len(x) - 1)
    n_effective = float(w.sum() ** 2 / (w**2).sum())
    return Weighted(
        mean=mean,
        sd=float(np.sqrt(variance)),
        se=float(np.sqrt(variance / n_effective)),
        n=int(keep.sum()),
        n_effective=n_effective,
    )


def composite(
    frame: pd.DataFrame,
    items: tuple[str, ...],
    options: int,
    missing_codes: tuple[float, ...] = (),
    scale: float = 100.0,
    min_answered: int | None = None,
    midpoint: bool = False,
) -> pd.Series:
    """One respondent-level composite, built the way Pfänder builds its own.

    Pfänder's composites are the respondent's mean over the items, and the arm
    mean is taken afterwards.  For the *mean* the order does not matter; for the
    *SD* it matters a great deal, and the SD is the whole reason a dispersion
    calibration would come here later, so the respondent-level route is the only
    correct one.

    ``min_answered`` defaults to every item, matching Pfänder's own requirement
    that a respondent answer all of a composite's items.  It is loosened only for
    source batteries carrying a "not applicable" option, where listwise deletion
    would discard a quarter of the sample to protect a number that is graded
    unusable anyway.
    """
    convert = to_slider_bin_midpoint if midpoint else to_slider
    converted = pd.DataFrame(
        {
            item: convert(frame[item], options, scale, missing_codes)
            for item in items
            if item in frame.columns
        }
    )
    if converted.empty:
        return pd.Series(np.nan, index=frame.index)
    required = len(items) if min_answered is None else min_answered
    answered = converted.notna().sum(axis=1)
    return converted.mean(axis=1).where(answered >= required)
