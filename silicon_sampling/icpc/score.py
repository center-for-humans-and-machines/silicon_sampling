"""Score a silicon sample against the ICPC's real US respondents.

The comparison is the benchmark's: split the humans in half on a fixed seed;
**Human 1** is the reference every prediction is scored against, **Human 2**
predicts Human 1 exactly as our sample does and its score says what a fresh human
sample of that size achieves, and two baselines — "no effect" and "all positive" —
anchor the metrics that have no natural null.

Everything here rests on one filter, and it is worth being precise about it.  The
published file holds 59,508 rows from 63 countries; ``country == "usa"`` selects
8,253 of them.  That is the whole US collection, pooled over the three US teams
(``teams`` is ``usa_1``/``usa_2``/``usa_3``, n = 838 / 2,360 / 5,055), all three of
which fielded the same instrument — same block ids, same flow, same stimuli, with
three spelling variants between them ("gray"/"grey", "labor"/"labour",
"droughts"/"draughts") and nothing else.  No further filtering is applied, because the file is already the
*cleaned* export: non-consenters, out-of-range ages, failed translations, both
attention checks and the WEPT demonstration have all been removed upstream, which
is why ``AttentionCheck60`` is 1 for every one of the 8,253 rows.  Two further
cuts were considered and rejected: dropping the 111 unfinished sessions would
discard respondents who answered the outcome battery before quitting, and
restricting to the largest team would throw away 39% of the sample to buy nothing.

Note the polarity of the fourth outcome.  Belief, policy support and sharing rise
under all eleven arms; the effortful task *falls* under nine of them, by as much
as 8.6 points of its 0-8 scale, and rises under the other two (BindingMoral
+0.88, DynamicNorm +1.88).  So it is not enough to flip one outcome's assumed
sign: the WEPT column is the one place in this study where the direction has to
be predicted rather than assumed, which is why the "all positive" baseline still
manages 81.8% directional agreement here and is nonetheless wrong about a quarter
of the cells.
"""

from __future__ import annotations

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
from .instrument import CONDITIONS, CONTROL
from .paths import DOELL_CSV, VLASCEANU_XLSX

#: The one column that identifies the US quota subsample.
US_FILTER_COLUMN = "country"
US_FILTER_VALUE = "usa"

#: Moderators a synthetic respondent is given and can therefore condition on.
#: Every one of them is asked on screen in this instrument — unlike the Voelkel
#: run, where age, education and ideology came from the panel supplier — so there
#: is no invisible-moderator caveat to carry here.
VISIBLE_MODERATORS = ("gender", "age_band", "education", "income_band", "ideology_band")

GENDER_LABELS = {
    1: "Male",
    2: "Female",
    3: "Prefer not to say",
    4: "Non-binary/third gender/other",
}
EDUCATION_LABELS = {
    1: "Up to grade school",
    2: "Up to high school",
    3: "College / undergraduate",
    4: "More than 17 years",
    5: "Prefer not to answer",
}
INCOME_BANDS = {
    1: "Under $25k",
    2: "Under $25k",
    3: "Under $25k",
    4: "$25k-$50k",
    5: "$50k-$100k",
    6: "$100k-$150k",
    7: "$150k+",
    8: "$150k+",
    9: "Prefer not to respond",
}
AGE_BREAKS = ([17, 29, 44, 59, 200], ["18-29", "30-44", "45-59", "60+"])
#: Political orientation is two 0-100 sliders, not a 7-point scale; the bands are
#: thirds of the range so a subgroup table has cells with respondents in them.
IDEOLOGY_BREAKS = ([-1, 33, 66, 100], ["left", "centre", "right"])

#: Columns the human loader needs out of the 668-column export.
RAW_COLUMNS = (
    "ResponseId",
    "country",
    "teams",
    "cond",
    "condName",
    "Finished",
    "Gender",
    "Age",
    "Education.2",
    "Income",
    "MacArthur_SES",
    "Politics2_1",
    "Politics2_9",
) + oc.REQUIRED_ITEMS


def load_raw(columns=RAW_COLUMNS, path=DOELL_CSV) -> pd.DataFrame:
    """The published export, read with the encoding it was actually written in.

    The file is not valid UTF-8 — several collaborators' free-text answers are
    latin-1 — so reading it any other way fails on row 1.
    """
    return pd.read_csv(
        path, encoding="latin-1", low_memory=False, usecols=list(columns)
    )


def us_subsample(frame: pd.DataFrame) -> pd.DataFrame:
    """``country == "usa"``: the 8,253-respondent US quota sample."""
    return frame[frame[US_FILTER_COLUMN] == US_FILTER_VALUE].copy()


def load_humans(conditions=CONDITIONS, path=DOELL_CSV) -> pd.DataFrame:
    """Real US respondents, with the four outcomes and the moderators attached."""
    frame = us_subsample(load_raw(path=path))
    frame = frame[frame["condName"].isin(conditions)].copy()
    frame = oc.compute(frame)
    frame["condition"] = frame["condName"]
    frame["gender"] = pd.to_numeric(frame["Gender"], errors="coerce").map(GENDER_LABELS)
    frame["education"] = pd.to_numeric(frame["Education.2"], errors="coerce").map(
        EDUCATION_LABELS
    )
    frame["income_band"] = pd.to_numeric(frame["Income"], errors="coerce").map(
        INCOME_BANDS
    )
    breaks, labels = AGE_BREAKS
    frame["age_band"] = pd.cut(
        pd.to_numeric(frame["Age"], errors="coerce"), breaks, labels=labels
    ).astype(str)
    mean_ideology = (
        pd.to_numeric(frame["Politics2_1"], errors="coerce")
        + pd.to_numeric(frame["Politics2_9"], errors="coerce")
    ) / 2
    breaks, labels = IDEOLOGY_BREAKS
    frame["ideology_band"] = pd.cut(mean_ideology, breaks, labels=labels).astype(str)
    return frame.reset_index(drop=True)


def arm_counts(frame: pd.DataFrame) -> pd.DataFrame:
    """Respondents per arm, and how many contributed each outcome."""
    rows = []
    for condition in CONDITIONS:
        cell = frame[frame["condition"] == condition]
        row = {"condition": condition, "n": len(cell)}
        for outcome in oc.OUTCOMES:
            values = pd.to_numeric(cell[outcome], errors="coerce")
            row[f"n_{outcome}"] = int(values.notna().sum())
            row[f"mean_{outcome}"] = float(values.mean())
        rows.append(row)
    return pd.DataFrame(rows)


def effects(frame: pd.DataFrame) -> pd.DataFrame:
    """ATEs of every arm against the control, per outcome, in points of scale."""
    return treatment_effects(frame, dict(oc.OUTCOMES), control=CONTROL)


def reference_halves(frame: pd.DataFrame, seed: int = 42):
    """Human 1 (the reference) and Human 2 (the replication), on a fixed seed."""
    return half_split(frame, seed=seed)


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
    human1: pd.DataFrame,
    human2: pd.DataFrame,
    synthetic: pd.DataFrame | dict[str, pd.DataFrame] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The comparison table, and the reference effects it is built on."""
    if synthetic is None:
        synthetic = {}
    elif isinstance(synthetic, pd.DataFrame):
        synthetic = {"Silicon sample": synthetic}
    reference = effects(human1)
    rows = [
        score_one(reference, effects(sample), label)
        for label, sample in synthetic.items()
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


def human_reference(seed: int = 42, path=DOELL_CSV) -> dict:
    """Everything a report needs about the human side, computed once."""
    humans = load_humans(path=path)
    human1, human2 = reference_halves(humans, seed=seed)
    board, reference = leaderboard(human1, human2)
    return {
        "humans": humans,
        "human1": human1,
        "human2": human2,
        "counts": arm_counts(humans),
        "effects": reference,
        "effects_all": effects(humans),
        "board": board,
    }


def verify_outcomes(doell=DOELL_CSV, vlasceanu=VLASCEANU_XLSX) -> pd.DataFrame:
    """Check the outcome construction against the cleaned publication."""
    raw = load_raw(path=doell)
    published = pd.read_excel(vlasceanu)
    return oc.verify_against_published(raw, published)


def verify_items(doell=DOELL_CSV, vlasceanu=VLASCEANU_XLSX) -> pd.DataFrame:
    """The same check per item, which is the one a permuted battery fails."""
    raw = load_raw(path=doell)
    published = pd.read_excel(vlasceanu)
    return oc.verify_items_against_published(raw, published)


def distribution_table(human1: pd.DataFrame, synthetic: pd.DataFrame) -> pd.DataFrame:
    """Shape metrics for every arm x outcome cell."""
    rows = []
    for condition in sorted(set(human1["condition"]) & set(synthetic["condition"])):
        for outcome in oc.OUTCOMES:
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


def subgroup_table(
    human1: pd.DataFrame, synthetic: pd.DataFrame, moderators=VISIBLE_MODERATORS
) -> pd.DataFrame:
    """Arm x moderator-level effects, scored the same way as the ATEs."""
    rows = []
    for moderator in moderators:
        pairs_all = []
        for level in sorted(
            set(human1[moderator].dropna()) & set(synthetic[moderator].dropna())
        ):
            human_cell = human1[human1[moderator] == level]
            synth_cell = synthetic[synthetic[moderator] == level]
            if (human_cell["condition"] == CONTROL).sum() < 30 or (
                synth_cell["condition"] == CONTROL
            ).sum() < 30:
                continue
            pairs = ate_pairs(effects(human_cell), effects(synth_cell))
            if len(pairs) >= 3:
                pairs_all.append(pairs.assign(level=level))
        if pairs_all:
            joined = pd.concat(pairs_all, ignore_index=True)
            rows.append(
                {
                    "moderator": moderator,
                    "n_levels": joined["level"].nunique(),
                    **M.signed_metrics(joined),
                }
            )
    return pd.DataFrame(rows)


def baseline_means(human1: pd.DataFrame, synthetic: pd.DataFrame) -> pd.DataFrame:
    """Control-arm group means, ours against theirs, per moderator level."""
    rows = []
    for moderator in VISIBLE_MODERATORS:
        for level in sorted(
            set(human1[moderator].dropna()) & set(synthetic[moderator].dropna())
        ):
            for outcome in oc.OUTCOMES:
                human_values = pd.to_numeric(
                    human1.loc[
                        (human1["condition"] == CONTROL) & (human1[moderator] == level),
                        outcome,
                    ],
                    errors="coerce",
                ).dropna()
                synth_values = pd.to_numeric(
                    synthetic.loc[
                        (synthetic["condition"] == CONTROL)
                        & (synthetic[moderator] == level),
                        outcome,
                    ],
                    errors="coerce",
                ).dropna()
                if len(human_values) < 30 or len(synth_values) < 30:
                    continue
                rows.append(
                    {
                        "moderator": moderator,
                        "level": level,
                        "outcome": outcome,
                        "human_mean": float(human_values.mean()),
                        "synthetic_mean": float(synth_values.mean()),
                        "abs_error": float(
                            abs(synth_values.mean() - human_values.mean())
                        ),
                    }
                )
    return pd.DataFrame(rows)


def parity_gap(baselines: pd.DataFrame) -> pd.DataFrame:
    """Worst- minus best-served group, per moderator: the benchmark's DPD."""
    rows = []
    for moderator, group in baselines.groupby("moderator"):
        per_level = group.groupby("level")["abs_error"].mean()
        if len(per_level) < 2:
            continue
        rows.append(
            {
                "moderator": moderator,
                "dpd": float(per_level.max() - per_level.min()),
                "worst_abs_err": float(per_level.max()),
                "worst_group": str(per_level.idxmax()),
                "best_group": str(per_level.idxmin()),
            }
        )
    return pd.DataFrame(rows)
