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


def letter_contribution(frame: pd.DataFrame | None = None) -> pd.DataFrame:
    """How much of ``political_advocacy`` rests on the one item we cannot code.

    ``letter`` is one of that composite's four members and the only outcome in this
    study whose value in the published file was produced by a classifier that no
    longer exists: GPT-3.5 read each respondent's free-text letter, a human checked
    the labels, and then the text was de-identified out of the file.  Our stand-in
    rule is a keyword-and-length test, and its base rate cannot be made to match,
    for a reason worth being precise about.

    The human column is 0.4155 over all 31,324 rows.  It is *zero-filled*: only
    23,575 respondents reached the letter screen at all, essentially every ``1`` is
    among those, and so 7,712 of the 18,287 zeros — 42% of them across the whole file,
    38% across the eleven kept arms — are attrition rather than a blank letter.  A sampled respondent cannot drop out, so there is
    no rule over their text that reproduces 0.4155 without fabricating dropout.
    Conditioning the comparison on reaching the page (0.545 over the kept arms)
    trades that bias for a collider: the control arm has the *lowest* reach, 69%
    against ~81% for the treatments, so its reachers are the most selected group in
    the study and every intervention's conditional letter rate comes out negative.

    What this table reports is the size of the resulting distortion, three ways.
    ``r_ate`` is the correlation between the real arm-level effects on the four-item
    composite and the effects on the same composite with ``letter`` held constant —
    which is *identical* for every constant, and identical again to dropping the
    item, because the three are affine transforms of one another.  Whatever it is
    that ``letter`` contributes to the between-arm signal, no rule available to us
    recovers any of it.  ``ate_inflation`` is what a constant does to the average
    absolute effect; ``level_bias`` is what our own 0.85 does to the composite's
    level.  All three belong on the table next to any ``political_advocacy`` result.
    """
    humans = load_humans() if frame is None else frame
    parts = list(oc.COMPOSITES["political_advocacy"])
    arms = humans.groupby("condition")

    real = arms["political_advocacy"].mean()
    ours = 0.85
    held = humans.assign(letter=ours)
    held["_pa"] = sum(
        pd.to_numeric(held[part], errors="coerce") for part in parts
    ) / len(parts)
    constant = held.groupby("condition")["_pa"].mean()

    real_ate = (real - real[CONTROL]).drop(CONTROL)
    constant_ate = (constant - constant[CONTROL]).drop(CONTROL)
    reached = humans["letter_timing"].notna()
    return pd.DataFrame(
        [
            {
                "human_letter_rate_all_rows": round(float(humans["letter"].mean()), 4),
                "human_letter_rate_reached_page": round(
                    float(humans.loc[reached, "letter"].mean()), 4
                ),
                "share_of_zeros_that_are_attrition": round(
                    float(
                        ((humans["letter"] == 0) & ~reached).sum()
                        / max(1, (humans["letter"] == 0).sum())
                    ),
                    4,
                ),
                "our_letter_rate": ours,
                "r_ate": round(float(real_ate.corr(constant_ate)), 4),
                "rho_ate": round(
                    float(real_ate.corr(constant_ate, method="spearman")), 4
                ),
                "mean_abs_ate_real": round(float(real_ate.abs().mean()), 4),
                "mean_abs_ate_constant": round(float(constant_ate.abs().mean()), 4),
                "ate_inflation": round(
                    float(constant_ate.abs().mean() / real_ate.abs().mean()), 3
                ),
                "level_real": round(float(humans["political_advocacy"].mean()), 4),
                "level_constant": round(float(held["_pa"].mean()), 4),
                "level_bias": round(
                    float(held["_pa"].mean() - humans["political_advocacy"].mean()), 4
                ),
            }
        ]
    )


def newsletter_contribution(frame: pd.DataFrame | None = None) -> pd.DataFrame:
    """How much of ``newsletter`` is differential attrition rather than signup.

    The counterpart to :func:`letter_contribution`, and it needed writing for the
    same reason: ``newsletter`` is a *zero-filled* column.  The published file
    marks a signup 1, a refusal 0, and the 4,140 of 19,141 kept-arm respondents
    who never reached the signup form 0 as well, because ``newsletter`` is
    constructed as the OR of two items that are themselves missing for anyone who
    dropped out.  Unlike ``letter`` there is nothing wrong with our *measurement*
    — the two signup questions are ordinary yes/no items and a sampled respondent
    answers them as well as a real one — so this is not a coding problem.  It is
    an estimand problem, and it is worse.

    ``letter`` at least sits inside a four-member composite.  ``newsletter`` is a
    standalone member of :data:`~silicon_sampling.goldwert.outcomes.SCORED` at
    weight 1.0 *and* one of the four in ``public_awareness``, so the attrition is
    scored twice, and :func:`effects` takes the all-rows mean both times.

    What the table says, and why the direction is knowable.  Reach is *lowest* in
    the control arm (69.2%) and higher in nine of the ten treatments (up to 83.3%
    for ``HopeAngerNarratives``), so an intervention that kept people in the
    survey raised its own all-rows mean without persuading anyone to sign up.  The
    zero-fill therefore inflates the mean absolute arm effect by 1.95x, from
    0.0191 to 0.0372, and it flips the sign on four of the ten arms —
    ``MispCorrectionRisks`` +0.0405 -> -0.0006, ``HopeAngerNarratives`` +0.0397 ->
    -0.0048, ``IndStructuralChange`` +0.0160 -> -0.0230, ``EcologicalDisruptions``
    +0.0151 -> -0.0252.  The two sets of arm effects correlate r = 0.4526
    (Spearman 0.6121), below ``letter``'s 0.8212.

    **The remaining bias, plainly.**  A silicon sample has reach 1.0 by
    construction; it cannot drop out and it cannot be made to.  So it estimates
    the reach-conditional effect, and it is scored against the all-rows one.  The
    direction is *attenuation*, by about a factor of two on average, plus a
    wrong-signed prediction on those four arms — and a level that sits about +0.07
    high, 0.3183 against 0.2495, because our respondents are all reachers and
    reachers sign up more.  None of that is fixable inside a transcript: making it
    right would mean fabricating dropout, and conditioning the human side on reach
    instead trades the bias for a collider, exactly as it does for ``letter``,
    since the control arm's reachers are the most selected group in the study.
    The honest move is to report both columns and read every ``newsletter`` and
    ``public_awareness`` result knowing which one it was scored on.
    """
    humans = load_humans() if frame is None else frame
    reached = humans["newsletter1_timing"].notna()
    all_rows = humans.groupby("condition")["newsletter"].mean()
    conditional = humans[reached].groupby("condition")["newsletter"].mean()
    all_ate = (all_rows - all_rows[CONTROL]).drop(CONTROL)
    conditional_ate = (conditional - conditional[CONTROL]).drop(CONTROL)
    reach = humans.assign(_reached=reached).groupby("condition")["_reached"].mean()
    flipped = [
        arm
        for arm in all_ate.index
        if np.sign(all_ate[arm]) != np.sign(conditional_ate[arm])
    ]
    return pd.DataFrame(
        [
            {
                "n_rows": len(humans),
                "n_reached_page": int(reached.sum()),
                "human_rate_all_rows": round(float(humans["newsletter"].mean()), 4),
                "human_rate_reached_page": round(
                    float(humans.loc[reached, "newsletter"].mean()), 4
                ),
                "level_bias_of_reach_one": round(
                    float(
                        humans.loc[reached, "newsletter"].mean()
                        - humans["newsletter"].mean()
                    ),
                    4,
                ),
                "reach_control": round(float(reach[CONTROL]), 4),
                "reach_min_treatment": round(float(reach.drop(CONTROL).min()), 4),
                "reach_max_treatment": round(float(reach.drop(CONTROL).max()), 4),
                "r_ate": round(float(all_ate.corr(conditional_ate)), 4),
                "rho_ate": round(
                    float(all_ate.corr(conditional_ate, method="spearman")), 4
                ),
                "mean_abs_ate_all_rows": round(float(all_ate.abs().mean()), 4),
                "mean_abs_ate_reach_conditional": round(
                    float(conditional_ate.abs().mean()), 4
                ),
                "ate_inflation": round(
                    float(all_ate.abs().mean() / conditional_ate.abs().mean()), 3
                ),
                "n_arms_sign_flipped": len(flipped),
                "arms_sign_flipped": "|".join(sorted(flipped)),
            }
        ]
    )


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
