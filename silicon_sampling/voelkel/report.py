"""The validation report: how well the pipeline reproduces a study we can check."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..analysis import plotting as viz
from ..benchmark.reference import ate_pairs, half_split
from . import outcomes as oc
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
    for frame, colour in ((pairs, viz.BLUE), (human_pairs, viz.RED)):
        clean = frame.dropna(subset=["estimate_h", "estimate_l"])
        if len(clean) > 2:
            slope, intercept = np.polyfit(clean["estimate_l"], clean["estimate_h"], 1)
            grid = np.linspace(-limit, limit, 10)
            ax.plot(grid, intercept + slope * grid, color=colour, lw=2.0)
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_xlabel("predicted effect (pp of scale range)")
    ax.set_ylabel("human effect, Human 1 (pp of scale range)")
    ax.set_title("Predicted against human treatment effects")
    ax.legend(loc="upper left")
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

Read every number against the human replication row, not against 1.0. A real
replication of this size scores **r = {theirs['pearson_r']:.2f}**; our sample scores
**r = {ours['pearson_r']:.2f}**{_interval(ours, 'pearson_r')}. Directional agreement is
**{ours['directional_pct']:.0f}%** against the replication's {theirs['directional_pct']:.0f}%
and a no-information floor of {null_row['directional_pct']:.0f}%.

![Predicted against human effects](plots/01_predicted_vs_human.png)

![Composite effects](plots/02_composite_effects.png)

## What the pieces say

- **[Effects](01_effects.md)** — arm by arm, outcome by outcome, ours against theirs.
- **[Distributions](02_distributions.md)** — whether the spread is right, not just the mean.
- **[Subgroups](03_subgroups.md)** — and the fact that only three of the five
  moderators were ever visible to the model.
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
