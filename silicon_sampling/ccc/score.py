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


def pool_controls(frame: pd.DataFrame) -> pd.DataFrame:
    """Collapse the three placebo arms into one ``Control``, wherever they appear.

    ``load_humans`` does this for the human side, and a silicon sample arrives with
    the raw labels — so ``effects`` has to do it for both or the two sides carry
    different condition vocabularies. Left undone, the OLS design matrix halts
    because its reference level is missing, which is how this was caught; a
    scorer that quietly dropped the reference instead would have compared a
    treatment against nothing.
    """
    if "condition" not in frame.columns:
        return frame
    out = frame.copy()
    out["condition"] = np.where(
        out["condition"].isin(inst.CONTROL_ARMS), CONTROL, out["condition"]
    )
    return out


#: The six boxes of the constant-sum donation item: five organisations plus
#: "keep for myself".
DONATION_BOXES = tuple(f"Donation_{index}" for index in range(1, 7))


def normalise_donation(frame: pd.DataFrame) -> pd.DataFrame:
    """Put the donation allocation back on the budget the survey enforced.

    CCC's donation question is a **constant-sum** item: Qualtrics holds the six
    boxes to a total of exactly 100, and every one of the 13,173 real respondents
    obeys it because the instrument would not let them do otherwise.  The
    transcript the silicon samples were drawn against rendered it as six
    independent sliders with no such constraint, so the models allocate whatever
    they like -- 42% of Qwen2.5-7B's respondents, 13% of Qwen2.5-72B's and 1% of
    Muse's write a total other than 100, and totals reach 600.  ``Donation`` is
    then the raw sum of five of those boxes, which runs to 500 on a scale the
    scoring code declares to be 0-100.

    This cannot be re-elicited, so it is repaired: the six boxes are rescaled to
    total 100, which preserves the respondent's *relative* allocation -- the only
    thing a constant-sum item actually elicits -- and discards an absolute
    magnitude the instrument never asked for.  Rows whose six boxes total zero
    have no allocation to preserve and are left missing rather than invented.

    On the human frame this is the identity, since the totals are already 100.
    It moves the synthetic control means *toward* the humans (65.5 -> 63.2 and
    66.9 -> 61.6 against a human 60.6), so it is a correction rather than a
    rescaling that happens to flatter.
    """
    if not all(box in frame.columns for box in DONATION_BOXES):
        return frame
    out = frame.copy()
    boxes = out[list(DONATION_BOXES)].apply(pd.to_numeric, errors="coerce")
    total = boxes.sum(axis=1, min_count=len(DONATION_BOXES))
    factor = np.where(total > 0, 100.0 / total.where(total > 0), np.nan)
    for box in DONATION_BOXES:
        out[box] = boxes[box] * factor
    if "Donation" in out.columns:
        out["Donation"] = out[list(DONATION_BOXES[:5])].sum(
            axis=1, min_count=len(DONATION_BOXES) - 1
        )
    return out


#: The survey's on-screen education wording, and the three levels the released
#: human file collapses it to.  The clone carries the real respondent's education
#: -- the two sides' counts agree exactly, 4765 / 4262 / 3730 -- so this is
#: relabelling, not recoding.
EDUCATION_ONSCREEN_TO_RELEASED = {
    "Bachelor's degree": "Bachelor or Postgraduate",
    "Master's degree / Professional degree": "Bachelor or Postgraduate",
    "Doctorate degree / Ph.D.": "Bachelor or Postgraduate",
    "Some college or Associate's degree": "Some college",
    "High school diploma / GED": "HS or less",
    "Less than high school": "HS or less",
}

#: The elicited party answer, folded onto the released file's three-way coding.
#:
#: The human variable is ``PartyC3``, which folds leaners into the party they
#: lean toward; the synthetic variable is the model's own self-identification,
#: which has no leaner follow-up to fold.  Mapping the residual categories onto
#: ``Neither`` puts both sides on the same three groups, which is the most that
#: can be done after the fact -- the leaner question was never asked of the
#: models, so their ``Neither`` stays larger than the humans' (31% against 13%).
#: Any party-gap comparison on CCC carries that caveat.
PARTY_ELICITED_TO_RELEASED = {
    "Democrat": "Democrat",
    "Republican": "Republican",
    "Independent": "Neither",
    "Other": "Neither",
    "Other (please specify)": "Neither",
    "Neither": "Neither",
}


def harmonise_moderators(frame: pd.DataFrame) -> pd.DataFrame:
    """Put the synthetic moderator labels in the released file's vocabulary.

    Left undone, ``education`` and ``party`` carry different level names on the
    two sides, so every subgroup interaction, demographic baseline and parity gap
    is computed against a reference level that does not exist on the other side --
    which the benchmark's grid assertion catches as a join of 748 pairs where 836
    were expected, and which a looser scorer would have reported as a number.
    """
    out = frame.copy()
    for column, mapping in (
        ("education", EDUCATION_ONSCREEN_TO_RELEASED),
        ("party", PARTY_ELICITED_TO_RELEASED),
    ):
        if column not in out.columns:
            continue
        labels = out[column].astype(str)
        out[column] = labels.map(mapping).fillna(labels)
    return out


def prepare(frame: pd.DataFrame) -> pd.DataFrame:
    """Everything a CCC frame needs before it is scored, human or synthetic.

    Pooling the placebo arms, repairing the constant-sum donation and putting the
    moderator labels in one vocabulary all have to happen to *both* sides, or the
    two carry different condition names, different budgets and different
    demographic groups.
    """
    return harmonise_moderators(normalise_donation(pool_controls(frame)))


def effects(frame: pd.DataFrame) -> pd.DataFrame:
    """Simple ATEs per outcome, in pp of scale range — the benchmark's estimand."""
    frame = prepare(frame)
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

    frame = pool_controls(frame)
    rows = []
    for outcome, scale in oc.SCORED.items():
        pre = outcome.replace("_Post", "_Pre")
        # ``Donation`` has no pre-measure and no ``_Post`` in its name, so the
        # substitution returns the outcome itself.  Left unguarded, the frame gets
        # a duplicated column, ``data[outcome]`` comes back two-dimensional, and
        # the OLS fails on a broadcast deep inside the HC2 weights.
        if pre == outcome or outcome not in frame.columns or pre not in frame.columns:
            continue
        data = frame[["condition", outcome, pre]].dropna()
        if data.empty or data["condition"].nunique() < 2:
            continue
        X, names = design_matrix(
            {"condition": data["condition"].tolist()},
            reference={"condition": CONTROL},
        )
        # ``design_matrix`` builds an intercept plus dummies and knows nothing
        # about continuous regressors, so the covariate is appended here rather
        # than passed in.  An earlier version passed ``numeric=`` and had never
        # been run; see ``test_ccc_ancova_actually_runs_and_buys_precision``.
        X = np.column_stack([X, data[pre].to_numpy(dtype=float)])
        names = list(names) + [pre]
        fit = ols(X, data[outcome].to_numpy(dtype=float), names, robust="HC2")
        for name in names:
            if name == "(Intercept)" or not name.startswith("condition["):
                continue
            term = fit.term(name)
            arm = name[len("condition") + 1 : -1]
            rows.append(
                {
                    "outcome": outcome,
                    "condition": arm,
                    # ``n`` is what ``ate_pairs`` merges on alongside the estimate,
                    # so omitting it made every downstream comparison a KeyError.
                    "n": int((data["condition"] == arm).sum()),
                    "estimate": term["estimate"] / scale * 100,
                    "se": term["se"] / scale * 100,
                }
            )
    return pd.DataFrame(rows)
