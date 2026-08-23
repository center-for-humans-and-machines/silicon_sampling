"""Score a silicon sample of this study against the real responses.

The shape of the comparison, in one paragraph: split the humans in half on a
fixed seed; **Human 1** is the reference every prediction is scored against;
**Human 2** predicts Human 1 exactly as our sample does, and its score says what
a fresh human sample of that size achieves.  Two further rows — "no effect" and
"all positive" — anchor the metrics that have no natural null.  Everything is
read *relative to those rows*, because an absolute correlation over ten
intervention clusters means very little on its own.

Two things are specific to this study and both are handled here rather than left
for whoever reads the table.

**Only the arms a text transcript can carry are scored.**  Seven of eighteen arms
are built around a video or a screenshot of a newspaper article, so a silicon
sample cannot be given them and their human effects are not comparable to
anything we could produce.  :func:`load_humans` drops them, leaving the control
and ten interventions.

**Display position is a covariate, not noise.**  The nine outcome blocks were
presented in a random order per respondent, and the study's own analysis finds
five-percentage-point swings between first and last position — ``march`` falls
from 47 to 39, ``belief_1`` *rises* from 48 to 58.  Randomisation means position
cannot bias a treatment contrast, but it inflates the variance of every one, and
a silicon sample that always shows one order has no such variance at all.
:func:`position_effects` puts the size of it on the table so that a distributional
comparison is read knowing it is there.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..benchmark import distributions as D
from ..benchmark import metrics as M
from ..benchmark.reference import (
    all_positive_prediction,
    ate_pairs,
    half_split,
    null_prediction,
    treatment_effects,
)
from . import outcomes as oc
from .instrument import CONDITIONS, CONTROL, DV_BLOCK_ORDER
from .paths import RESPONSES_CSV

#: Moderators a synthetic respondent could condition on, and those it could not.
#: Everything here is asked on screen *after* the outcomes, so in the human data
#: it is missing for anyone who dropped out early — which is 25% of the sample.
VISIBLE_MODERATORS = ("party", "gender", "age_band", "education")
INVISIBLE_MODERATORS = ()

AGE_BANDS = ([17, 29, 44, 59, 200], ["18-29", "30-44", "45-59", "60+"])
EDUCATION = {1: "Grade school", 2: "High school", 3: "College", 4: "Postgraduate"}


def load_humans(conditions=CONDITIONS, path=RESPONSES_CSV) -> pd.DataFrame:
    """Real respondents with every derived outcome, restricted to the usable arms."""
    frame = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    frame = frame[frame["condName"].isin(conditions)].copy()
    frame = oc.compute(frame)
    frame["condition"] = frame["condName"]
    frame["party"] = frame["Party"]
    frame["gender"] = frame["Gender"]
    frame["education"] = pd.to_numeric(frame["Edu"], errors="coerce").map(EDUCATION)
    frame["age_band"] = pd.cut(
        pd.to_numeric(frame["Age"], errors="coerce"), AGE_BANDS[0], labels=AGE_BANDS[1]
    ).astype("object")
    return frame.reset_index(drop=True)


def effects(frame: pd.DataFrame) -> pd.DataFrame:
    """ATEs of every arm against the control, for every scored outcome."""
    return treatment_effects(frame, dict(oc.SCORED), control=CONTROL)


def human_reference(
    frame: pd.DataFrame | None = None, seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """The reference effects, and the two halves they came from.

    Returns ``(reference, human1, human2)``.  ``reference`` is what a submission
    is scored against; ``human2`` is the replication that says how good a score is
    achievable at this sample size.
    """
    humans = load_humans() if frame is None else frame
    human1, human2 = half_split(humans, seed=seed)
    return effects(human1), human1, human2


def score_one(
    reference: pd.DataFrame,
    prediction: pd.DataFrame,
    label: str,
    bootstrap: bool = True,
) -> dict:
    """All pooled metrics for one prediction against the reference."""
    pairs = ate_pairs(reference, prediction)
    row = {
        "submission": label,
        **M.pooled_metrics(pairs),
        **M.run_calibration_pooled(pairs),
    }
    if bootstrap and pairs["condition"].nunique() >= 3:
        interval = M.cluster_bootstrap(
            pairs,
            lambda p: {
                **M.pooled_metrics(p, include_rmse=True),
                **M.run_calibration_pooled(p),
            },
            draws=2000,
        )
        row.update(
            {
                key: value
                for key, value in interval.items()
                if key.endswith(("_lo", "_hi")) or key == "n_clusters"
            }
        )
    return row


def leaderboard(
    synthetic: pd.DataFrame | dict[str, pd.DataFrame] | None = None,
    frame: pd.DataFrame | None = None,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The comparison table, and the reference effects it is built on.

    Called with no sample at all it still produces the human-replication and
    baseline rows, which is the useful thing to have before any sampling has run:
    it says what score this study's own noise floor allows.
    """
    reference, _, human2 = human_reference(frame, seed=seed)
    samples: dict[str, pd.DataFrame] = {}
    if isinstance(synthetic, pd.DataFrame):
        samples = {"Silicon sample": synthetic}
    elif synthetic:
        samples = dict(synthetic)
    rows = [
        score_one(reference, effects(sample), label)
        for label, sample in samples.items()
    ]
    rows += [
        score_one(reference, effects(human2), "Human replication (Human 2)"),
        score_one(
            reference,
            null_prediction(reference),
            "Baseline: no effect",
            bootstrap=False,
        ),
        score_one(
            reference,
            all_positive_prediction(reference),
            "Baseline: all positive",
            bootstrap=False,
        ),
    ]
    return pd.DataFrame(rows), reference


def position_effects(frame: pd.DataFrame | None = None) -> pd.DataFrame:
    """Each outcome's mean by the display position of its own block.

    ``DV_order`` names the nine blocks in the order that respondent saw them, so
    the position of the block an outcome sits in is recoverable per row.  The
    slope of the mean on position is the quantity a silicon sample cannot
    reproduce, because a template shows one order.
    """
    humans = load_humans() if frame is None else frame
    block_of = {
        "belief_1": "BeliefandPolicySupport",
        "policy_1": "BeliefandPolicySupport",
        "petition": "Petition",
        "letter": "OpenEndedLetter",
        "pol_candidate": "supportclimaterepelection",
        "pol_campaign": "supportclimaterepelection",
        "bank": "Bank",
        "donation": "Donation",
        "march": "Attendmarch",
        "newsletter": "Newsletter",
        "conversation": "Commitment",
        "flyless": "Commitment",
        "lessbeef": "Commitment",
    }
    orders = humans["DV_order"].fillna("").str.split("|")
    rows = []
    for column, block in block_of.items():
        if column not in humans.columns:
            continue
        position = orders.map(
            lambda names, want=block: (
                [n.strip() for n in names].index(want)
                if want in [n.strip() for n in names]
                else np.nan
            )
        )
        values = pd.to_numeric(humans[column], errors="coerce")
        usable = position.notna() & values.notna()
        if usable.sum() < 100:
            continue
        first = values[usable & (position == 0)].mean()
        last = values[usable & (position == len(DV_BLOCK_ORDER) - 1)].mean()
        slope = np.polyfit(position[usable].astype(float), values[usable], 1)[0]
        rows.append(
            {
                "outcome": column,
                "block": block,
                "n": int(usable.sum()),
                "mean_first_position": round(float(first), 4),
                "mean_last_position": round(float(last), 4),
                "slope_per_position": round(float(slope), 4),
            }
        )
    return pd.DataFrame(rows)


def distribution_table(human1: pd.DataFrame, synthetic: pd.DataFrame) -> pd.DataFrame:
    """Shape metrics for every condition x outcome cell."""
    rows = []
    for condition in sorted(set(human1["condition"]) & set(synthetic["condition"])):
        for outcome in oc.SCORED:
            human_values = pd.to_numeric(
                human1.loc[human1["condition"] == condition, outcome], errors="coerce"
            ).dropna()
            synth_values = pd.to_numeric(
                synthetic.loc[synthetic["condition"] == condition, outcome],
                errors="coerce",
            ).dropna()
            if len(human_values) < 10 or len(synth_values) < 10:
                continue
            rows.append(
                {
                    "condition": condition,
                    "outcome": outcome,
                    **D.compare_distributions(human_values, synth_values),
                }
            )
    return pd.DataFrame(rows)


def control_levels(frame: pd.DataFrame | None = None) -> pd.DataFrame:
    """Control-arm level for every scored outcome: what a sample has to hit first.

    Before any effect can be calibrated the *baseline* has to be right, and for
    the two outcomes this study exists to anchor — the donation and the newsletter
    signup — this table is the whole deliverable.
    """
    humans = load_humans() if frame is None else frame
    control = humans[humans["condition"] == CONTROL]
    rows = []
    for outcome, scale in oc.SCORED.items():
        values = pd.to_numeric(control[outcome], errors="coerce").dropna()
        if values.empty:
            continue
        rows.append(
            {
                "outcome": outcome,
                "label": oc.LABELS.get(outcome, outcome),
                "scale_max": scale,
                "n_control": len(values),
                "mean": round(float(values.mean()), 4),
                "sd": round(float(values.std()), 4),
                "median": round(float(values.median()), 4),
                "share_at_zero": round(float((values == 0).mean()), 4),
            }
        )
    return pd.DataFrame(rows)


#: There is no US filter to apply.  The whole recruited sample is US-resident:
#: 31,324 adults recruited through CloudResearch Connect in June 2024, quota-matched
#: to the US Census on age, race, gender and ethnicity, with residence verified by
#: the panel provider.  Two columns look like they could filter on it and neither
#: can — ``Country Of Residence == "United States"`` is populated for only the
#: 10,568 respondents who carry a Connect profile, and ``Region`` for 20,668 — so
#: applying either would throw away two thirds of the sample to no purpose.  The
#: known deviation from the Census is sex: 60% female against 40% male.
US_FILTER = None
US_FILTER_NOTE = (
    "none required: every respondent is a US resident by recruitment. "
    '`Country Of Residence == "United States"` covers 10,568 of 31,324 rows and '
    "`Region` covers 20,668; both are panel-provider metadata, not eligibility."
)


def us_coverage(path=RESPONSES_CSV) -> pd.DataFrame:
    """Per-arm counts, and how far each US-looking column actually reaches.

    Written to be read alongside :data:`US_FILTER_NOTE`: the point is that the
    per-arm n is 1,733-1,745 whichever way you look at it, and every column that
    mentions the United States is sparser than the sample it describes.
    """
    frame = pd.read_csv(path, low_memory=False)
    rows = []
    for arm, group in frame.groupby("condName"):
        rows.append(
            {
                "condName": arm,
                "cond": int(pd.to_numeric(group["cond"], errors="coerce").iloc[0]),
                "n_assigned": len(group),
                "n_country_is_us": int(
                    (group["Country Of Residence"] == "United States").sum()
                ),
                "n_region_known": int(group["Region"].notna().sum()),
                "n_state_known": int(group["State"].notna().sum()),
                "n_demographics_complete": int(
                    group[["Gender", "Age", "Party"]].notna().all(axis=1).sum()
                ),
                "n_reached_donation": int(group["donation"].notna().sum()),
            }
        )
    return pd.DataFrame(rows).sort_values("cond").reset_index(drop=True)
