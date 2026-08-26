"""Human reference and effect estimates for the Climate Change Challenge.

Two estimands live here and the difference matters.

``effects`` fits ``outcome ~ condition`` with HC2 standard errors — the **simple
ATE**, which is what the Pfänder benchmark refits from a submission.  This is the
one the cross-validation uses, because a fold is only informative about Pfänder if
it is scored the way Pfänder is scored.

``effects_ancova`` fits ``post ~ condition + pre``, which is what the published
paper does (with HC3).  It is kept because the gap between the two is itself a
useful number: it measures how much precision the pre-measure buys, and therefore
how much of the achievable correlation on a study like this is capped by noise in
the human effect estimates rather than by the model being wrong.

The three placebo arms — neckties, baseball, dances — are pooled into one
``Control``, exactly as the published ``ConditionR`` does, giving n = 3,183 against
about 1,060 per treatment.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..benchmark.reference import treatment_effects
from . import instrument as inst
from . import outcomes as oc
from .paths import RECODED_CSV

CONTROL = "Control"


def load_humans() -> pd.DataFrame:
    """Real respondents, with the placebo arms pooled and moderators normalised.

    Restricted to the arms our sample covers, so the dropped
    ``System Preservation Framing`` is excluded from the human side too — scoring
    against an arm we never sampled would silently compare nothing.
    """
    frame = pd.read_csv(RECODED_CSV, encoding="utf-8-sig", low_memory=False)
    keep = set(inst.ARM_BLOCKS)
    frame = frame[frame["Condition"].isin(keep)].copy()
    frame["condition"] = np.where(
        frame["Condition"].isin(inst.CONTROL_ARMS), CONTROL, frame["Condition"]
    )
    frame["gender"] = frame["Gender"].astype(str)
    # Education is already collapsed to three strings in the released file; the
    # survey asked five levels.  Use what was released rather than pretending.
    frame["education"] = frame["Education"].astype(str)
    frame["party"] = frame["PartyC3"].astype(str)
    frame["age_band"] = pd.cut(
        pd.to_numeric(frame["Age"], errors="coerce"),
        [17, 29, 44, 59, 200],
        labels=["18-29", "30-44", "45-59", "60+"],
    ).astype(str)
    race_labels = {
        1: "White / Caucasian",
        2: "Black / African-American",
        3: "Latino / Hispanic",
        4: "Asian / Asian-American",
        5: "Other",
    }
    frame["race"] = pd.to_numeric(frame["Race"], errors="coerce").map(race_labels)
    return frame.reset_index(drop=True)


def effects(frame: pd.DataFrame) -> pd.DataFrame:
    """Simple ATEs per outcome, in pp of scale range — the benchmark's estimand."""
    present = {k: v for k, v in oc.SCORED.items() if k in frame.columns}
    return treatment_effects(frame, present, control=CONTROL)


def effects_ancova(frame: pd.DataFrame) -> pd.DataFrame:
    """``post ~ condition + pre``, the published estimand, for comparison only.

    Reported beside :func:`effects` so the precision the pre-measure buys is
    visible rather than assumed.  Uses the same HC2 the rest of this project uses
    rather than the paper's HC3, so the only difference from :func:`effects` is the
    covariate.
    """
    from ..analysis.ols import design_matrix, ols

    rows = []
    for outcome, scale in oc.SCORED.items():
        pre = outcome.replace("_Post", "_Pre")
        if outcome not in frame.columns or pre not in frame.columns:
            continue
        data = frame[["condition", outcome, pre]].dropna()
        if data.empty or data["condition"].nunique() < 2:
            continue
        X, names = design_matrix(
            {"condition": data["condition"].tolist()},
            reference={"condition": CONTROL},
            numeric={pre: data[pre].to_numpy(dtype=float)},
        )
        fit = ols(X, data[outcome].to_numpy(dtype=float), names, robust="HC2")
        for name in names:
            if name == "(Intercept)" or not name.startswith("condition["):
                continue
            term = fit.term(name)
            rows.append(
                {
                    "outcome": outcome,
                    "condition": name[len("condition") + 1 : -1],
                    "estimate": term.estimate / scale * 100,
                    "se": term.se / scale * 100,
                }
            )
    return pd.DataFrame(rows)
