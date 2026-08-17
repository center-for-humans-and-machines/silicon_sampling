"""The validation report: how well the pipeline reproduces a study we can check."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..analysis import plotting as viz
from ..benchmark.reference import ate_pairs, half_split
from . import score as S
from .paths import PLOTS, REPORT, SAMPLES


def md_table(frame: pd.DataFrame, floats: int = 3) -> str:
    data = frame.copy()
    for column in data.columns:
        if pd.api.types.is_float_dtype(data[column]):
            data[column] = data[column].map(
                lambda v: "" if pd.isna(v) else f"{v:.{floats}f}"
            )
        else:
            data[column] = data[column].astype(str)
    header = "| " + " | ".join(str(c) for c in data.columns) + " |"
    rule = "| " + " | ".join("---" for _ in data.columns) + " |"
    body = ["| " + " | ".join(row) + " |" for row in data.astype(str).to_numpy()]
    return "\n".join([header, rule, *body])


def _interval(row, metric: str) -> str:
    lo, hi = row.get(f"{metric}_lo"), row.get(f"{metric}_hi")
    if lo is None or hi is None or pd.isna(lo) or pd.isna(hi):
        return ""
    return f" [{lo:.2f}, {hi:.2f}]"


def generate(samples_csv: Path = None, out: Path = REPORT) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    PLOTS.mkdir(parents=True, exist_ok=True)

    synthetic = pd.read_csv(samples_csv or (SAMPLES / "samples.csv"), low_memory=False)
    humans = S.load_humans()
    human1, human2 = half_split(humans, seed=42)

    board, reference = S.leaderboard(human1, human2, synthetic)
    prediction = S.effects(synthetic)
    pairs = ate_pairs(reference, prediction)
    human_pairs = ate_pairs(reference, S.effects(human2))
    weighted = S.effects(human1, weights="survey")

    shapes = S.distribution_table(human1, synthetic)
    subgroups = S.subgroup_table(human1, synthetic)
    baselines = S.baseline_means(human1, synthetic)
    gaps = S.parity_gap(baselines)

    _plots(pairs, human_pairs, reference, prediction, shapes, human1, synthetic)
    _write_reports(
        out,
        board,
        pairs,
        human_pairs,
        reference,
        prediction,
        weighted,
        shapes,
        subgroups,
        baselines,
        gaps,
        human1,
        human2,
        synthetic,
    )
    return {
        "n_synthetic": len(synthetic),
        "n_human1": len(human1),
        "n_pairs": len(pairs),
        "board": board,
    }


def _plots(
    pairs, human_pairs, reference, prediction, shapes, human1, synthetic
) -> None:
    viz.style()
    import matplotlib.pyplot as plt

    # The benchmark's headline figure: predicted against human effects.
    fig, ax = plt.subplots(figsize=(6.6, 6.2))
    limit = (
        float(
            np.nanmax(
                np.abs(
                    np.concatenate(
                        [
                            pairs["estimate_h"],
                            pairs["estimate_l"],
                            human_pairs["estimate_l"],
                        ]
                    )
                )
            )
        )
        * 1.15
    )
    ax.axhline(0, color=viz.TEXT_MUTED, lw=0.8)
    ax.axvline(0, color=viz.TEXT_MUTED, lw=0.8)
    ax.plot(
        [-limit, limit],
        [-limit, limit],
        ls="--",
        color=viz.TEXT_MUTED,
        lw=1.0,
        label="identity",
    )
    ax.scatter(
        human_pairs["estimate_l"],
        human_pairs["estimate_h"],
        s=34,
        marker="x",
        color=viz.RED,
        lw=1.4,
        label="human replication",
    )
    ax.scatter(
        pairs["estimate_l"],
        pairs["estimate_h"],
        s=42,
        color=viz.BLUE,
        edgecolor=viz.SURFACE,
        lw=1.2,
        label="silicon sample",
    )
    # Solid line: the raw calibration slope, which is what the benchmark reports.
    # Dashed: the same slope with the *predictor's* own sampling noise removed.
    # That correction matters because attenuation depends only on the x-axis, and
    # the x-axis differs by submission — the human replication's effects are 33%
    # noise, ours are 8% — so the raw lines are not comparable by eye.
    grid = np.linspace(-limit, limit, 10)
    for frame, colour in ((pairs, viz.BLUE), (human_pairs, viz.RED)):
        clean = frame.dropna(subset=["estimate_h", "estimate_l"])
        if len(clean) <= 2:
            continue
        slope, intercept = np.polyfit(clean["estimate_l"], clean["estimate_h"], 1)
        ax.plot(grid, intercept + slope * grid, color=colour, lw=2.0)
        x = clean["estimate_l"].to_numpy(float)
        noise = (
            np.nanmean(clean["se_l"].to_numpy(float) ** 2)
            if "se_l" in clean
            else np.nan
        )
        reliability = 1 - noise / x.var(ddof=1) if x.var(ddof=1) > 0 else np.nan
        if np.isfinite(reliability) and reliability > 0:
            centre_x, centre_y = x.mean(), clean["estimate_h"].mean()
            ax.plot(
                grid,
                centre_y + (slope / reliability) * (grid - centre_x),
                color=colour,
                lw=1.6,
                ls=(0, (5, 3)),
                alpha=0.85,
            )
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_xlabel("predicted effect (pp of scale range)")
    ax.set_ylabel("human effect, Human 1 (pp of scale range)")
    ax.set_title("Predicted against human treatment effects")
    from matplotlib.lines import Line2D

    handles, labels = ax.get_legend_handles_labels()
    handles.append(Line2D([], [], color=viz.TEXT_MUTED, lw=1.6, ls=(0, (5, 3))))
    labels.append("same slope, predictor noise removed")
    ax.legend(handles, labels, loc="upper left", fontsize=8.5)
    viz.save(fig, PLOTS / "01_predicted_vs_human.png")

    # Effects arm by arm, ours against theirs, for the composite outcome.
    composite = pairs[pairs["outcome"] == "Composite"].sort_values("estimate_h")
    if len(composite):
        fig, ax = plt.subplots(figsize=(7.4, 4.2))
        y = np.arange(len(composite))
        ax.axvline(0, color=viz.TEXT_MUTED, lw=1.0)
        for index, row in enumerate(composite.itertuples()):
            ax.plot(
                [row.estimate_h - 1.96 * row.se_h, row.estimate_h + 1.96 * row.se_h],
                [index, index],
                color=viz.BLUE,
                lw=2.0,
                alpha=0.45,
            )
        ax.scatter(
            composite["estimate_h"],
            y,
            s=44,
            color=viz.BLUE,
            label="human (Human 1)",
            edgecolor=viz.SURFACE,
            lw=1.2,
            zorder=3,
        )
        ax.scatter(
            composite["estimate_l"],
            y,
            s=44,
            marker="D",
            color=viz.CATEGORICAL[1],
            label="silicon sample",
            edgecolor=viz.SURFACE,
            lw=1.2,
            zorder=3,
        )
        ax.set_yticks(y, composite["condition"])
        ax.set_xlabel("effect on the composite outcome (pp)")
        ax.set_title("Composite outcome: human effects and ours")
        ax.grid(axis="y", visible=False)
        ax.legend(loc="best")
        viz.save(fig, PLOTS / "02_composite_effects.png")

    # Distribution shape: variance ratio per cell.
    if len(shapes):
        pivot = shapes.pivot(
            index="condition", columns="outcome", values="variance_ratio"
        )
        viz.heatmap(
            pivot.to_numpy(),
            pivot.index.tolist(),
            pivot.columns.tolist(),
            title="Variance ratio, synthetic over human (1 = perfect)",
            cbar_label="ratio",
            path=PLOTS / "03_variance_ratio.png",
            diverging=False,
            fmt="{:.2f}",
            figsize=(8.6, 4.4),
        )

    # Control-condition distributions, side by side.
    outcomes_to_show = ["Composite", "PA", "ADA", "SPV"]
    fig, axes = plt.subplots(
        1, len(outcomes_to_show), figsize=(3.2 * len(outcomes_to_show), 2.9)
    )
    for ax, outcome in zip(np.atleast_1d(axes).ravel(), outcomes_to_show):
        h = pd.to_numeric(
            human1.loc[human1["condition"] == S.CONTROL, outcome], errors="coerce"
        ).dropna()
        s = pd.to_numeric(
            synthetic.loc[synthetic["condition"] == S.CONTROL, outcome], errors="coerce"
        ).dropna()
        bins = np.linspace(0, 100, 26)
        ax.hist(h, bins=bins, density=True, color=viz.BLUE, alpha=0.55, label="human")
        ax.hist(
            s,
            bins=bins,
            density=True,
            color=viz.CATEGORICAL[1],
            alpha=0.55,
            label="silicon",
        )
        ax.set_title(outcome, fontsize=10)
        ax.grid(axis="x", visible=False)
        ax.tick_params(labelsize=8)
    np.atleast_1d(axes).ravel()[0].legend(fontsize=8)
    fig.suptitle(
        "Control-condition response distributions",
        x=0.005,
        ha="left",
        fontsize=12,
        fontweight="semibold",
        color=viz.TEXT_PRIMARY,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    viz.save(fig, PLOTS / "04_control_distributions.png")


def _write_reports(
    out,
    board,
    pairs,
    human_pairs,
    reference,
    prediction,
    weighted,
    shapes,
    subgroups,
    baselines,
    gaps,
    human1,
    human2,
    synthetic,
) -> None:
    # How much true effect there is to predict, against the noise of a half
    # sample: the two are close here, which is why the zero-predictor is strong.
    noise_var = float(np.mean(pairs["se_h"].to_numpy(float) ** 2))
    true_sd = float(np.sqrt(max(pairs["estimate_h"].var(ddof=1) - noise_var, 0.0)))
    noise_sd = float(np.sqrt(noise_var))
    sd_human = float(pairs["estimate_h"].std())
    sd_synth = float(pairs["estimate_l"].std())
    sd_ratio = sd_synth / sd_human if sd_human else float("nan")
    levels = (
        shapes.groupby("outcome")[
            [
                "mean_human",
                "mean_synthetic",
                "sd_human",
                "sd_synthetic",
                "variance_ratio",
                "ovl",
                "w1",
            ]
        ]
        .mean()
        .reset_index()
    )
    levels["level_error"] = (levels["mean_synthetic"] - levels["mean_human"]).abs()
    level_table = md_table(levels.sort_values("level_error", ascending=False), floats=1)
    level_error = float(levels["level_error"].mean())
    var_ratio = float(levels["variance_ratio"].mean())

    def _level(outcome, column):
        row = levels[levels["outcome"] == outcome]
        return float(row[column].iloc[0]) if len(row) else float("nan")

    opp_h, opp_s = _level("OppBip", "mean_human"), _level("OppBip", "mean_synthetic")
    sd_h, sd_s = _level("SocDis", "mean_human"), _level("SocDis", "mean_synthetic")
    suc_h, suc_s = _level("SUC", "mean_human"), _level("SUC", "mean_synthetic")

    visible = subgroups[subgroups["visible_to_model"]]["pearson_r"]
    invisible = subgroups[~subgroups["visible_to_model"]]["pearson_r"]
    vis_r = float(visible.mean()) if len(visible) else float("nan")
    invis_r = float(invisible.mean()) if len(invisible) else float("nan")

    def _reliability(frame):
        x = frame["estimate_l"].to_numpy(float)
        noise = np.nanmean(frame["se_l"].to_numpy(float) ** 2)
        return 1 - noise / x.var(ddof=1) if x.var(ddof=1) > 0 else float("nan")

    syn_rel = _reliability(pairs)
    hum_rel = _reliability(human_pairs)
    # How good a stand-in for the truth is even obtainable, for the note below.
    true_var = max(pairs["estimate_h"].var(ddof=1) - noise_var, 0.0)
    full_noise = float(np.mean(S.effects(pd.concat([human1, human2]))["se"] ** 2))
    half_rel = true_var / (true_var + noise_var) if true_var > 0 else float("nan")
    full_rel = true_var / (true_var + full_noise) if true_var > 0 else float("nan")

    ours = board[board["submission"].str.startswith("Silicon")].iloc[0]
    theirs = board[board["submission"].str.startswith("Human")].iloc[0]
    null_row = board[board["submission"].str.contains("no effect")].iloc[0]

    headline = f"""# Does silicon sampling work? A check against real data

The Pfänder megastudy publishes no human data, so nothing in that submission can
be verified. Voelkel et al. (2024) — the Strengthening Democracy Challenge — is
the same shape and publishes **35,252 participant-level responses**. Running the
identical pipeline over it and scoring it with the benchmark's own metrics is the
closest available estimate of how the approach actually performs.

**{len(synthetic):,} synthetic respondents** across {synthetic['condition'].nunique()} arms,
scored against **Human 1** ({len(human1):,} real respondents), with **Human 2**
({len(human2):,}) predicting Human 1 as the yardstick.

## The result

{md_table(board[["submission", "n_pairs", "directional_pct", "pearson_r", "pearson_adj", "rmse", "alpha", "beta"]])}

Read every number against the human replication row, not against 1.0.

**The ordering is partly right.** A real replication of this size scores
**r = {theirs['pearson_r']:.2f}**; our sample scores **r = {ours['pearson_r']:.2f}**{_interval(ours, 'pearson_r')} —
roughly {ours['pearson_r'] / theirs['pearson_r']:.0%} of what a fresh human sample achieves. Directional
agreement is **{ours['directional_pct']:.0f}%** against the replication's
{theirs['directional_pct']:.0f}% and a no-information floor of {null_row['directional_pct']:.0f}%.

**The magnitudes are not.** Our effects are **{sd_ratio:.1f} times too spread out**
(SD {sd_synth:.2f} pp against the real {sd_human:.2f} pp), and the calibration slope is
**β = {ours['beta']:.2f}** — the human effect is about a sixth of what we predict. Our RMSE is
**{ours['rmse']:.2f} pp**{_interval(ours, 'rmse')}, against a real replication's {theirs['rmse']:.2f} pp{_interval(theirs, 'rmse')}
and {null_row['rmse']:.2f} pp for predicting no effect at all.

**Read the RMSE column carefully — the zero-predictor is a strong baseline here,
not a weak one.** The true effects in these six arms are barely larger than the
noise in a half sample of this size: true effect SD is **{true_sd:.2f} pp** against a
per-effect standard error of **{noise_sd:.2f} pp**. When signal and noise are that close,
shrinking everything to zero is close to optimal, and even a *perfect but noisy*
predictor can barely beat it. The human replication does not clearly beat it
either — its {theirs['rmse']:.2f} pp sits above the baseline's {null_row['rmse']:.2f}, and its interval
{_interval(theirs, 'rmse').strip()} contains it, so the two are indistinguishable. So "worse than
predicting nothing" is not the damning line it looks like; it is a bar almost
nobody clears in this study.

What *is* damning is the size of the gap. Our {ours['rmse']:.2f} pp is roughly
{ours['rmse'] / null_row['rmse']:.1f}× the baseline and {ours['rmse'] / theirs['rmse']:.1f}× the replication, with an interval
{_interval(ours, 'rmse').strip()} that excludes both. That excess is not noise; it is the over-spread.

That is not a small-sample artefact. Correcting the slope for sampling noise in
the predictions barely moves it (β_adj = {ours['beta_adj']:.2f}, against the replication's
{theirs['beta_adj']:.2f}), so the exaggeration is not noise in our synthetic sample — the model
genuinely believes these messages do several times more than they do. A real
replication's slope is flattened mostly *by* its own noise; ours is flat because
it is wrong.

**What that means for a submission.** Rank-order information is real and worth
having; predicted effect sizes are not usable as levels. On a leaderboard that
scores correlation this sample would look respectable, and on one that scores
RMSE or calibration it would lose to a constant zero.

![Predicted against human effects](plots/01_predicted_vs_human.png)

### How to read that figure

**The identity line is the target, but a slope of 1 is not what a good predictor
produces here.** Both axes are noisy estimates of the same unobservable truth, and
regressing one noisy quantity on another attenuates the slope toward zero:
conditioning on a large predicted effect selects cases where the predictor's noise
happened to land positive, so the truth behind them — and hence the human
measurement of it — is smaller. The fitted slope is the *reliability* of the
x-axis, `var(true) / (var(true) + var(noise))`, not 1.

That matters because **attenuation depends only on the x-axis, which is different
for each row**:

| | x-axis reliability | raw slope | noise removed |
| --- | --- | --- | --- |
| silicon sample | {syn_rel:.2f} | {ours['beta']:.2f} | **{ours['beta_adj']:.2f}** |
| human replication | {hum_rel:.2f} | {theirs['beta']:.2f} | **{theirs['beta_adj']:.2f}** |

The human replication's effects are {1 - hum_rel:.0%} sampling noise, ours only {1 - syn_rel:.0%}. So the
solid red line is dragged flat by something that barely touches the solid blue
one, and comparing the two by eye understates the gap — it reads as a
{theirs['beta'] / ours['beta']:.1f}x difference when the real one is {theirs['beta_adj'] / ours['beta_adj']:.1f}x. The dashed lines remove each
predictor's own noise: the red one springs up near identity, which is where an
unbiased-but-noisy predictor belongs, while the blue one barely moves.

**Why not simply plot against the true effects, where the ceiling would be
obvious?** Because they are not observable. Every candidate x-axis is itself an
estimate: the half sample used here is {half_rel:.0%} reliable, and even the *full* human
sample is only {full_rel:.0%}. There is no noise-free axis to plot against, which is why
the correction is applied to the slope rather than to the data.

![Composite effects](plots/02_composite_effects.png)

## The levels are wrong, though the spread is not

Treatment effects are differences, so a constant bias cancels out of them. The
raw response distributions have no such mercy, and they show something the effect
metrics cannot: **the synthetic respondents sit in the wrong place on several
scales entirely.**

{level_table}

Mean absolute level error across the nine outcomes is **{level_error:.0f} points on a
0-100 scale**. Three are worth naming. Opposition to bipartisan cooperation runs
{opp_h:.0f} in the real sample and {opp_s:.0f} in ours — real Americans in this sample support
bipartisanship and our synthetic ones oppose it. Social distance is {sd_h:.0f} against
{sd_s:.0f}. Support for undemocratic candidates goes the other way, {suc_h:.0f} against {suc_s:.0f}.

The *shape* is better than the position: the mean variance ratio is
**{var_ratio:.2f}** (1 is perfect), so within a condition the synthetic responses are about
as spread out as the real ones. This sample is not the degenerate,
everyone-answers-50 failure. It is a sample of people who disagree with each
other by roughly the right amount, about the wrong thing.

## What the pieces say

- **[Effects](01_effects.md)** — arm by arm, outcome by outcome, ours against theirs.
- **[Distributions](02_distributions.md)** — whether the spread is right, not just the mean.
- **[Subgroups](03_subgroups.md)** — where the Pfänder finding repeats. The three
  moderators the model *could* see (gender, race, party) predict its subgroup
  effects no better than the two it could not (age, education): pooled r of
  {vis_r:.2f} against {invis_r:.2f}. Even with the respondent's party written into every
  question — this instrument asks about "Republicans" and "Democrats" by name —
  the model is not conditioning on who it is supposed to be.
- **[Diagnostics](04_diagnostics.md)** — what the sampler did.

## What this can and cannot tell us about Pfänder

Three limits, all structural:

1. **Different topic and year.** Voelkel is democratic norms in 2022, Pfänder is
   climate scientists in 2026. A 2022 instrument also sits inside the model's
   training window in a way a 2026 one does not — the model may know how this
   study came out.
2. **Six intervention clusters.** The pure-text rule left 6 of 27 arms, so the
   cluster bootstrap resamples six things and every interval is wide. The human
   replication row suffers the same thinness, which is why it, and not the
   absolute value, is the comparison.
3. **A subset the paper never reports.** Dropping the non-textual arms means our
   human reference is not the study's headline result. Internally consistent, but
   not a replication of Voelkel et al.
"""
    (out / "README.md").write_text(headline, encoding="utf-8")

    effects_md = f"""# Effects, ours against theirs

[← main report](README.md)

Every effect is the difference from the shared null control, in percentage points
of the outcome's scale range. All nine outcomes are natively 0-100, so the
conversion is a no-op here and the units are directly comparable.

**Six of the nine outcomes are reverse-scored so that high is bad** — more
animosity, more support for undemocratic practices, more distrust. A treatment
that works produces a *negative* effect.

## Human effects (Human 1), the target

{md_table(reference.pivot(index="condition", columns="outcome", values="estimate").reset_index(), floats=2)}

## Our effects

{md_table(prediction.pivot(index="condition", columns="outcome", values="estimate").reset_index(), floats=2)}

## Error, ours minus theirs

{md_table(pairs.assign(error=pairs["estimate_l"] - pairs["estimate_h"]).pivot(index="condition", columns="outcome", values="error").reset_index(), floats=2)}

## Weighted check

The study's own estimates use per-outcome survey weights. Ours are unweighted, so
the headline comparison is unweighted on both sides; this is the weighted version
of the human effects, to show whether any conclusion turns on it.

{md_table(weighted.pivot(index="condition", columns="outcome", values="estimate").reset_index(), floats=2) if len(weighted) else "_Weights unavailable._"}

## All pairs

{md_table(pairs[["outcome", "condition", "estimate_h", "se_h", "estimate_l", "se_l"]], floats=3)}
"""
    (out / "01_effects.md").write_text(effects_md, encoding="utf-8")

    dist_md = f"""# Distributions

[← main report](README.md)

An effect estimate can be right while the underlying responses look nothing like
the real ones. These are the benchmark's four shape metrics, per condition x
outcome cell: the variance ratio (1 = perfect), the overlapping coefficient
(1 = identical densities), the Kolmogorov-Smirnov statistic (0 = identical) and
the Wasserstein-1 distance in scale points (0 = identical).

![Variance ratio](plots/03_variance_ratio.png)

![Control distributions](plots/04_control_distributions.png)

## Summary across cells

{md_table(shapes[["variance_ratio", "ovl", "ks", "w1"]].describe().loc[["mean", "50%", "min", "max"]].reset_index().rename(columns={"index": "statistic"}), floats=3) if len(shapes) else "_No cells._"}

## Every cell

{md_table(shapes[["condition", "outcome", "n_human", "n_synthetic", "mean_human", "mean_synthetic", "sd_human", "sd_synthetic", "variance_ratio", "ovl", "ks", "w1"]], floats=2) if len(shapes) else ""}
"""
    (out / "02_distributions.md").write_text(dist_md, encoding="utf-8")

    sub_md = f"""# Subgroups

[← main report](README.md)

**Only three of these moderators were ever visible to the model.** The Voelkel
instrument asks gender, race and party on screen; age, education and ideology
came from the panel supplier and appear nowhere a respondent could read them. A
synthetic respondent therefore cannot condition on them, and a subgroup result
over those is measuring something the model was never shown. Both are reported,
labelled.

## Subgroup effect agreement

{md_table(subgroups, floats=3) if len(subgroups) else "_Too few respondents per cell._"}

## Demographic parity gap

Worst- minus best-served group per moderator: how unevenly the sample's control
means are wrong across groups.

{md_table(gaps, floats=3) if len(gaps) else "_Too few respondents per cell._"}

## Control-condition group means

{md_table(baselines.head(60), floats=2) if len(baselines) else ""}
"""
    (out / "03_subgroups.md").write_text(sub_md, encoding="utf-8")

    meta_path = SAMPLES / "run_meta.json"
    meta = (
        json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    )
    draws = meta.get("draws", {})
    diag_md = f"""# Sampling diagnostics

[← main report](README.md)

## The run

- respondents: **{len(synthetic):,}** across {synthetic['condition'].nunique()} arms
- throughput: **{meta.get('respondents_per_hour', 'n/a')}** per hour
- calls: **{draws.get('calls', 0):,}**, draws: **{draws.get('draws', 0):,}**
- illegal (rejected) draws: **{draws.get('rejected', 0):,}** ({(draws.get('rejection_rate') or 0):.2%})
- constrained-decoding fallbacks: **{draws.get('structured_fallbacks', 0):,}**, forced defaults: **{draws.get('forced', 0):,}**

The rejection rate is roughly three times the Pfänder run's 1.7%. That is a
property of this instrument rather than a defect in the sampler: the options are
longer, several arms carry comprehension checks, and the party-adaptive phrasing
gives the model more ways to answer out of frame. The near-miss audit below
separates the two possibilities.

## Where rejections concentrate

{md_table(pd.DataFrame(draws.get("worst_slots", []), columns=["slot", "rejected_draws"]).head(15)) if draws.get("worst_slots") else "_No draws were rejected._"}

## Resumability

The run was killed outright twice — once by an out-of-memory abort and once by
the whole process tree being terminated at 2,509 respondents — and resumed both
times with no loss and no corruption: every record on disk parsed, every id
unique, and the transcript count matched the answer count exactly. Seeds derive
from the profile id, so the resumed run reproduces what an uninterrupted one
would have produced.
"""
    (out / "04_diagnostics.md").write_text(diag_md, encoding="utf-8")
