"""Score the silicon sample against the real responses, the benchmark's way.

The shape of the comparison, in one paragraph: split the humans in half on a
fixed seed; **Human 1** is the reference every prediction is scored against;
**Human 2** predicts Human 1 exactly as our sample does, and its score says what
a fresh human sample of that size achieves.  Two further rows — "no effect" and
"all positive" — anchor the metrics that have no natural null.  Everything is
then read *relative to those rows*, because an absolute correlation over six
intervention clusters means very little on its own.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..benchmark import distributions as D
from ..benchmark import metrics as M
from ..benchmark.reference import (
    all_positive_prediction,
    ate_pairs,
    null_prediction,
    treatment_effects,
)
from . import outcomes as oc
from .instrument import CONDITIONS
from .paths import RECODED_CSV

CONTROL = "Null_Control"
#: Moderators the model could actually see, and those it could not.
VISIBLE_MODERATORS = ("gender", "race", "party_gen")
INVISIBLE_MODERATORS = ("age_band", "education")


def load_humans(conditions=CONDITIONS) -> pd.DataFrame:
    """Real respondents, restricted to the arms our sample covers."""
    frame = pd.read_csv(RECODED_CSV, encoding="utf-8-sig", low_memory=False)
    frame = frame[frame["Condition"].isin(conditions)].copy()
    frame = frame.rename(
        columns={"Condition": "condition", "Inparty_Person": "inparty"}
    )
    frame["party_gen"] = frame["Party_Gen"].map(
        {1: "Republican", 2: "Democrat", 3: "Independent"}
    )
    frame["gender"] = frame["Gender"]
    frame["race"] = frame["Race"]
    frame["education"] = frame["Education"]
    frame["age_band"] = pd.cut(
        frame["Age"], [17, 29, 44, 59, 200], labels=["18-29", "30-44", "45-59", "60+"]
    ).astype(str)
    return frame.reset_index(drop=True)


def effects(frame: pd.DataFrame, weights: str | None = None) -> pd.DataFrame:
    """ATEs for every outcome, unweighted or with the study's survey weights."""
    if weights is None:
        return treatment_effects(frame, dict(oc.OUTCOMES), control=CONTROL)
    rows = []
    for outcome in oc.OUTCOMES:
        column = _weight_column(outcome)
        if column not in frame.columns:
            continue
        data = frame[["condition", outcome, column]].dropna()
        control = data[data["condition"] == CONTROL]
        base = (
            np.average(control[outcome], weights=control[column])
            if len(control)
            else np.nan
        )
        for condition, group in data.groupby("condition"):
            if condition == CONTROL:
                continue
            mean = np.average(group[outcome], weights=group[column])
            rows.append(
                {
                    "outcome": outcome,
                    "condition": condition,
                    "n": len(group),
                    "estimate": mean - base,
                    "se": np.nan,
                }
            )
    return pd.DataFrame(rows)


def _weight_column(outcome: str) -> str:
    return {
        "PA": "weights_pa",
        "ADA": "weights_ada",
        "SPV": "weights_spv",
        "SUC": "weights_suc",
        "OppBip": "weights_oppbip",
        "SocDistrust": "weights_socdistrust",
        "SocDis": "weights_socdis",
        "BEPF": "weights_bepf",
        "Composite": "weights_composite",
    }.get(outcome, "")


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
    human1: pd.DataFrame, human2: pd.DataFrame, synthetic: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The comparison table, and the reference effects it is built on."""
    reference = effects(human1)
    rows = [
        score_one(reference, effects(synthetic), "Silicon sample (Qwen2.5-7B)"),
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


def distribution_table(human1: pd.DataFrame, synthetic: pd.DataFrame) -> pd.DataFrame:
    """Shape metrics for every condition x outcome cell."""
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
    human1: pd.DataFrame,
    synthetic: pd.DataFrame,
    moderators=VISIBLE_MODERATORS + INVISIBLE_MODERATORS,
) -> pd.DataFrame:
    """Condition x moderator-level effects, scored the same way as the ATEs."""
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
            reference = effects(human_cell)
            prediction = effects(synth_cell)
            pairs = ate_pairs(reference, prediction)
            if len(pairs) >= 3:
                pairs_all.append(pairs.assign(level=level))
        if pairs_all:
            joined = pd.concat(pairs_all, ignore_index=True)
            rows.append(
                {
                    "moderator": moderator,
                    "visible_to_model": moderator in VISIBLE_MODERATORS,
                    "n_levels": joined["level"].nunique(),
                    **M.signed_metrics(joined),
                }
            )
    return pd.DataFrame(rows)


def baseline_means(human1: pd.DataFrame, synthetic: pd.DataFrame) -> pd.DataFrame:
    """Control-condition group means, ours against theirs, per moderator level."""
    rows = []
    for moderator in VISIBLE_MODERATORS + INVISIBLE_MODERATORS:
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
                        "visible_to_model": moderator in VISIBLE_MODERATORS,
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
                "visible_to_model": bool(group["visible_to_model"].iloc[0]),
                "dpd": float(per_level.max() - per_level.min()),
                "worst_abs_err": float(per_level.max()),
                "worst_group": str(per_level.idxmax()),
                "best_group": str(per_level.idxmin()),
            }
        )
    return pd.DataFrame(rows)
