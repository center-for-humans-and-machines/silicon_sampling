"""Did the bigger base model make the silicon sample more faithful?

Voelkel is the only one of the two studies with real participant-level responses,
so it is the only place the question can actually be answered rather than
described.  Every number here is one model against the other **against the same
humans**, with the human replication (Human 2 predicting Human 1) as the yardstick
a fresh sample of this size achieves.

The comparison is *paired*: both models answered the same instrument about the
same interventions and are scored against the same reference, so their errors are
correlated and the difference between them is pinned down far better than either
score is.  Bootstrapping them separately and asking whether the intervals overlap
throws that away — see :func:`..benchmark.metrics.paired_cluster_bootstrap`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..analysis import plotting as viz
from ..benchmark.reference import ate_pairs, half_split
from ..models import label as model_label
from . import outcomes as oc
from . import score as S
from .paths import REPORT, samples_dir

#: One colour per submission, held stable across every figure so a reader can
#: carry identity between them.  Validated as a categorical trio on the light
#: surface (worst normal-vision ΔE 24.1, worst deutan 19.8); amber sits below 3:1
#: against the surface, so every figure using it ships its numbers as a table.
COLOURS = {"qwen25_7b": viz.CATEGORICAL[0], "v4_flash": viz.CATEGORICAL[3]}
MARKERS = {"qwen25_7b": "o", "v4_flash": "D"}
HUMAN_COLOUR = viz.RED

#: Outcomes are 0-100 scales, so a level error is in scale points.
LEVEL_METRICS = ("mean_human", "mean_synthetic", "variance_ratio", "ovl", "w1")


def load_runs(runs) -> dict[str, pd.DataFrame]:
    """Each run's ``samples.csv``, keyed by run key, in the order given."""
    loaded = {}
    for run in runs:
        path = samples_dir(run) / "samples.csv"
        if not path.exists():
            raise SystemExit(f"no sample at {path}; run build-csv for {run} first")
        loaded[run] = pd.read_csv(path, low_memory=False)
    return loaded


def level_errors(human1: pd.DataFrame, samples: dict) -> pd.DataFrame:
    """Per-outcome distance from the real distribution, model by model.

    Treatment effects are differences, so a constant bias cancels out of them;
    these do not.  This is where the Qwen sample's worst failure showed up — a
    mean absolute level error of 23 points on a 0-100 scale — so it is the first
    place to look for an improvement.
    """
    frames = []
    for run, sample in samples.items():
        table = S.distribution_table(human1, sample)
        if not len(table):
            continue
        per_outcome = (
            table.groupby("outcome")
            .agg(
                mean_human=("mean_human", "mean"),
                mean_synthetic=("mean_synthetic", "mean"),
                variance_ratio=("variance_ratio", "mean"),
                ovl=("ovl", "mean"),
                w1=("w1", "mean"),
            )
            .reset_index()
        )
        per_outcome["level_error"] = (
            per_outcome["mean_synthetic"] - per_outcome["mean_human"]
        ).abs()
        per_outcome["model"] = model_label(run)
        per_outcome["run"] = run
        frames.append(per_outcome)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def subgroup_signal(human1: pd.DataFrame, samples: dict) -> pd.DataFrame:
    """Do the models condition on who they are supposed to be?

    Pooled correlation of subgroup effects, split by whether the moderator was
    *visible* in the transcript.  A model that reads its assigned demographics
    should do better on the moderators it could see than on the ones it could not;
    Qwen2.5-7B did not (0.26 against 0.24), which is the finding this checks.
    """
    rows = []
    for run, sample in samples.items():
        table = S.subgroup_table(human1, sample)
        if not len(table):
            continue
        for visible, group in table.groupby("visible_to_model"):
            weights = group["n_pairs"].to_numpy(float)
            rows.append(
                {
                    "model": model_label(run),
                    "run": run,
                    "moderators": "visible" if visible else "invisible",
                    "n_moderators": len(group),
                    "pearson_r": float(
                        np.average(group["pearson_r"].to_numpy(float), weights=weights)
                    ),
                    "directional_pct": float(
                        np.average(
                            group["directional_pct"].to_numpy(float), weights=weights
                        )
                    ),
                }
            )
    return pd.DataFrame(rows)


def effect_magnitudes(
    human1: pd.DataFrame, human2: pd.DataFrame, samples: dict
) -> pd.DataFrame:
    """How big each submission says the effects are, against how big they are.

    Answers "over- or under-estimating?" directly, which the calibration slope
    does *not*: beta is ``cov(human, predicted) / var(predicted)``, so it moves
    with alignment as well as magnitude and a model can shrink toward the right
    size while beta falls.  Spread ratios separate the two.

    Three reference points, because "the truth" is not observable here:

    ``sd_estimate``
        The spread of the submission's 54 effect estimates.
    ``ratio_vs_human1``
        Against Human 1's *estimated* spread — the number the main report quotes.
        Human 1's estimates carry sampling noise, so this understates the
        exaggeration.
    ``ratio_vs_true``
        Against the noise-corrected spread, ``var(estimate) - mean(se^2)``, which
        is the closest thing to the real between-arm variation.
    """
    reference = S.effects(human1)
    pairs_human = ate_pairs(reference, S.effects(human2))
    sd_human1 = float(pairs_human["estimate_h"].std(ddof=1))
    noise = float(np.mean(pairs_human["se_h"].to_numpy(float) ** 2))
    sd_true = float(np.sqrt(max(sd_human1**2 - noise, 0.0)))

    rows = []
    for label, frame in list(samples.items()) + [("__human2__", human2)]:
        name = (
            "Human replication (Human 2)"
            if label == "__human2__"
            else model_label(label)
        )
        pairs = ate_pairs(reference, S.effects(frame))
        sd = float(pairs["estimate_l"].std(ddof=1))
        rows.append(
            {
                "submission": name,
                "mean_abs_effect": float(pairs["estimate_l"].abs().mean()),
                "sd_estimate": sd,
                "ratio_vs_human1": sd / sd_human1 if sd_human1 else float("nan"),
                "ratio_vs_true": sd / sd_true if sd_true else float("nan"),
            }
        )
    rows.append(
        {
            "submission": "Human 1 (the reference itself)",
            "mean_abs_effect": float(pairs_human["estimate_h"].abs().mean()),
            "sd_estimate": sd_human1,
            "ratio_vs_human1": 1.0,
            "ratio_vs_true": sd_human1 / sd_true if sd_true else float("nan"),
        }
    )
    frame = pd.DataFrame(rows)
    frame.attrs["sd_true"] = sd_true
    frame.attrs["sd_human1"] = sd_human1
    return frame


def party_gaps(human1: pd.DataFrame, samples: dict) -> pd.DataFrame:
    """Republican minus Democrat means in the control arm, ours against theirs.

    This is the *direct* version of the demographic question, and a cleaner one
    than subgroup treatment effects: it asks whether the model puts partisans in
    different places at all, before any intervention. Party is written into every
    question of this instrument by name, so a model that reads its assigned
    identity has no excuse for a flat gap — and one that reads it too eagerly
    produces the stereotyping failure the benchmark's diagnostic exists to catch.
    Both directions are wrong; they are wrong in opposite ways.
    """
    control = human1[human1["condition"] == S.CONTROL]
    table = pd.DataFrame({"outcome": list(oc.OUTCOMES)}).set_index("outcome")

    def gaps(frame: pd.DataFrame, column: str) -> pd.Series:
        arm = frame[frame["condition"] == S.CONTROL]
        found = {}
        for outcome in oc.OUTCOMES:
            if outcome not in arm.columns:
                continue
            means = (
                pd.to_numeric(arm[outcome], errors="coerce")
                .groupby(arm["party_gen"])
                .mean()
            )
            if {"Republican", "Democrat"} <= set(means.index):
                found[outcome] = means["Republican"] - means["Democrat"]
        return pd.Series(found, name=column)

    table = table.join(gaps(control, "human"))
    for run, sample in samples.items():
        table = table.join(gaps(sample, model_label(run)))
    return table.reset_index()


def party_gap_summary(table: pd.DataFrame) -> pd.DataFrame:
    """How big each model's partisan gaps are, and whether they point the right way."""
    rows = []
    for column in table.columns:
        if column in ("outcome", "human"):
            continue
        rows.append(
            {
                "model": column,
                "mean_abs_gap": float(table[column].abs().mean()),
                "sd_gap": float(table[column].std(ddof=1)),
                "corr_with_human_gaps": float(table["human"].corr(table[column])),
                "n_outcomes": int(table[column].notna().sum()),
            }
        )
    rows.append(
        {
            "model": "human (Human 1)",
            "mean_abs_gap": float(table["human"].abs().mean()),
            "sd_gap": float(table["human"].std(ddof=1)),
            "corr_with_human_gaps": 1.0,
            "n_outcomes": int(table["human"].notna().sum()),
        }
    )
    return pd.DataFrame(rows)


def _scatter(pairs_by_run: dict, human_pairs, path: Path) -> None:
    """Predicted against human effects, both models plus the human yardstick.

    The identity line is the target, but a slope of 1 is not what a good predictor
    produces: both axes are noisy estimates of the same unobservable truth, so the
    fitted slope is the reliability of the x-axis, not 1.  Reliability differs by
    submission, which is why the dashed lines (each predictor's own noise removed)
    are the comparable ones.
    """
    viz.style()
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    everything = [human_pairs["estimate_l"], human_pairs["estimate_h"]] + [
        column for pairs in pairs_by_run.values() for column in (pairs["estimate_l"],)
    ]
    limit = float(np.nanmax(np.abs(np.concatenate(everything)))) * 1.15

    fig, ax = plt.subplots(figsize=(6.8, 6.4))
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
    series = [
        (run, pairs, COLOURS[run], MARKERS[run]) for run, pairs in pairs_by_run.items()
    ]
    series.append(("human", human_pairs, HUMAN_COLOUR, "x"))
    grid = np.linspace(-limit, limit, 10)
    for key, pairs, colour, marker in series:
        name = "human replication" if key == "human" else model_label(key)
        # A surface ring keeps overlapping points readable.  'x' has no face to
        # ring — matplotlib warns and ignores the edgecolor — so it carries its
        # identity by shape and stroke weight instead.
        ring = {} if marker == "x" else {"edgecolor": viz.SURFACE}
        ax.scatter(
            pairs["estimate_l"],
            pairs["estimate_h"],
            s=40,
            marker=marker,
            color=colour,
            lw=1.4,
            label=name,
            zorder=3,
            **ring,
        )
        clean = pairs.dropna(subset=["estimate_h", "estimate_l"])
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
            ax.plot(
                grid,
                clean["estimate_h"].mean() + (slope / reliability) * (grid - x.mean()),
                color=colour,
                lw=1.5,
                ls=(0, (5, 3)),
                alpha=0.85,
            )
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_xlabel("predicted effect (pp of scale range)")
    ax.set_ylabel("human effect, Human 1 (pp of scale range)")
    ax.set_title("Predicted against human treatment effects")
    handles, labels = ax.get_legend_handles_labels()
    handles.append(Line2D([], [], color=viz.TEXT_MUTED, lw=1.5, ls=(0, (5, 3))))
    labels.append("same slope, predictor noise removed")
    ax.legend(handles, labels, loc="upper left", fontsize=8.5)
    viz.save(fig, path)


def _delta_plot(
    verdict: pd.DataFrame, baseline: str, contender: str, path: Path
) -> None:
    """Paired change per metric, with its 95% interval.

    **Plots the raw signed delta, the same number the table beside it prints.**
    An earlier version flipped the sign of the lower-is-better metrics so that
    "positive" would mean "better" on every row — and then still labelled each row
    with the metric's own convention, so the RMSE panel read "+0.812" next to
    "lower is better" while the table said -0.812. Two conventions in one panel,
    and the figure disagreed with its own table.

    So the sign stays raw and each row keeps its own convention in its label, which
    is where the direction belonged all along.  Nothing to flip mentally, and the
    numbers match the table.
    """
    viz.style()
    import matplotlib.pyplot as plt

    frame = verdict.copy()

    # Small multiples, one panel per metric.  These are measures of different
    # scale — directional agreement in percentage points, correlations in
    # correlation units, RMSE in scale points — so a shared x-axis buries the
    # correlation deltas beside a swing of 11 points, and a second axis is never
    # the answer.  Each panel gets its own scale and the title says so.
    fig, axes = plt.subplots(
        len(frame), 1, figsize=(7.0, 1.05 * len(frame) + 1.7), squeeze=False
    )
    # Each panel carries its own tick labels below it, so the panels need real
    # separation or those labels land on the next panel's marks.
    fig.subplots_adjust(hspace=1.25)
    for ax, row in zip(axes[:, 0], frame.itertuples()):
        span = max(abs(row.delta_lo), abs(row.delta_hi), abs(row.delta)) or 1.0
        limit = 1.45 * span
        ax.axvline(0, color=viz.TEXT_MUTED, lw=1.2)
        ax.plot(
            [row.delta_lo, row.delta_hi],
            [0, 0],
            color=viz.CATEGORICAL[0],
            lw=2.4,
            alpha=0.5,
        )
        ax.scatter(
            [row.delta],
            [0],
            s=58,
            color=viz.CATEGORICAL[0],
            edgecolor=viz.SURFACE,
            lw=1.2,
            zorder=3,
        )
        ax.annotate(
            f"{row.delta:+.3f}",
            (row.delta, 0),
            textcoords="offset points",
            xytext=(0, 12),
            ha="center",
            fontsize=9,
            color=viz.TEXT_SECONDARY,
        )
        ax.set_xlim(-limit, limit)
        ax.set_ylim(-0.5, 0.95)
        ax.set_yticks([])
        ax.tick_params(axis="x", labelsize=8)
        ax.set_ylabel(
            f"{row.metric}\n({'higher' if row.higher_is_better else 'lower'} is better)",
            rotation=0,
            ha="right",
            va="center",
            fontsize=9,
            color=viz.TEXT_SECONDARY,
        )
        for side in ("left", "right", "top"):
            ax.spines[side].set_visible(False)
    axes[0, 0].set_title(
        f"Paired change from {baseline} to {contender}\n"
        "raw signed delta, as in the table — so on a lower-is-better row a "
        "negative delta\nis the improvement · 95% cluster bootstrap · own scale "
        "per metric",
        fontsize=10.5,
    )
    viz.save(fig, path)


def _level_plot(levels: pd.DataFrame, path: Path) -> None:
    """Absolute level error per outcome, model against model.

    Two series, so identity matters: fixed categorical hues, a 2px surface gap
    between adjacent bars, and the numbers repeated in the table beside it.
    """
    viz.style()
    import matplotlib.pyplot as plt

    runs = list(dict.fromkeys(levels["run"]))
    outcomes = (
        levels[levels["run"] == runs[0]]
        .sort_values("level_error", ascending=False)["outcome"]
        .tolist()
    )
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    ax.set_axisbelow(True)  # the house style grids on; bars must sit over it
    width = 0.38
    base = np.arange(len(outcomes))
    for offset, run in zip((-width / 2, width / 2), runs):
        values = [
            float(
                levels.loc[
                    (levels["run"] == run) & (levels["outcome"] == outcome),
                    "level_error",
                ].iloc[0]
            )
            for outcome in outcomes
        ]
        ax.bar(
            base + offset,
            values,
            width=width - 0.02,  # the 0.02 is the surface gap between neighbours
            color=COLOURS[run],
            label=model_label(run),
        )
    ax.set_xticks(base)
    ax.set_xticklabels(outcomes, rotation=30, ha="right")
    ax.set_ylabel("|synthetic mean - human mean|  (0-100 scale)")
    ax.set_title("How far each sample sits from the real distribution")
    ax.legend(fontsize=9)
    viz.save(fig, path)


def diagnostics(runs) -> pd.DataFrame:
    """What each sampler did: throughput, illegal draws, forced answers."""
    import json

    rows = []
    for run in runs:
        path = samples_dir(run) / "run_meta.json"
        if not path.exists():
            continue
        meta = json.loads(path.read_text(encoding="utf-8"))
        draws = meta.get("draws", {})
        rows.append(
            {
                "model": model_label(run),
                "hf_id": meta.get("model"),
                "respondents_per_hour": meta.get("respondents_per_hour"),
                "rejection_rate": draws.get("rejection_rate"),
                "structured_fallbacks": draws.get("structured_fallbacks"),
                "forced_defaults": draws.get("forced"),
                "gpus": (meta.get("engine") or {}).get("tensor_parallel_size", 1),
            }
        )
    return pd.DataFrame(rows)


def _md(frame: pd.DataFrame, floats: int = 3) -> str:
    from .report import md_table

    return md_table(frame, floats=floats)


def generate(
    runs,
    out: Path = REPORT,
    baseline: str = "qwen25_7b",
    contender: str = "v4_flash",
    draws: int = 2000,
) -> dict:
    """Write ``05_model_comparison.md`` and its figures."""
    out.mkdir(parents=True, exist_ok=True)
    plots = out / "plots"
    plots.mkdir(parents=True, exist_ok=True)

    samples = load_runs(runs)
    humans = S.load_humans()
    human1, human2 = half_split(humans, seed=42)

    board, reference = S.leaderboard(
        human1, human2, {model_label(run): sample for run, sample in samples.items()}
    )
    verdict = S.model_comparison(
        human1,
        samples[baseline],
        samples[contender],
        baseline_label=model_label(baseline),
        contender_label=model_label(contender),
        draws=draws,
    )
    levels = level_errors(human1, samples)
    magnitudes = effect_magnitudes(human1, human2, samples)
    subgroups = subgroup_signal(human1, samples)
    gaps = party_gaps(human1, samples)
    gap_summary = party_gap_summary(gaps)
    diag = diagnostics(runs)

    pairs_by_run = {
        run: ate_pairs(reference, S.effects(sample)) for run, sample in samples.items()
    }
    human_pairs = ate_pairs(reference, S.effects(human2))
    _scatter(pairs_by_run, human_pairs, plots / "05_models_vs_human.png")
    if len(verdict):
        _delta_plot(
            verdict,
            model_label(baseline),
            model_label(contender),
            plots / "05_paired_change.png",
        )
    if len(levels):
        _level_plot(levels, plots / "05_level_error.png")

    _write(
        out,
        board,
        verdict,
        levels,
        magnitudes,
        subgroups,
        gaps,
        gap_summary,
        diag,
        samples,
        baseline,
        contender,
    )
    return {
        "board": board,
        "verdict": verdict,
        "levels": levels,
        "magnitudes": magnitudes,
        "subgroups": subgroups,
        "party_gaps": gaps,
        "party_gap_summary": gap_summary,
        "diagnostics": diag,
    }


def _verdict_sentence(verdict: pd.DataFrame, metric: str, contender: str) -> str:
    """One metric, stated with its interval and what it does or does not settle.

    The call is made on the one-sided bootstrap share rather than on whether the
    interval brackets zero.  Several of these metrics are discrete — directional
    agreement moves in steps of 1/54 — so an interval edge can land exactly on
    zero while every single resample points the same way, and reading that as
    "inconclusive" would be the opposite of what the draws say.
    """
    row = verdict[verdict["metric"] == metric]
    if not len(row):
        return f"- `{metric}`: not available."
    row = row.iloc[0]
    share = row["p_contender_better"]
    if share >= 0.975:
        judgement = f"a real improvement — {share:.0%} of resamples favour {contender}."
    elif share <= 0.025:
        judgement = (
            f"a real regression — {1 - share:.0%} of resamples go the other way."
        )
    else:
        judgement = "not settled: the resamples do not agree on a direction."
    return (
        f"- **{metric}**: {row['delta']:+.3f} "
        f"[{row['delta_lo']:+.3f}, {row['delta_hi']:+.3f}] — {judgement}"
    )


def _write(
    out,
    board,
    verdict,
    levels,
    magnitudes,
    subgroups,
    gaps,
    gap_summary,
    diag,
    samples,
    baseline,
    contender,
) -> None:
    base_name, cont_name = model_label(baseline), model_label(contender)
    counts = " · ".join(f"{model_label(run)}: {len(s):,}" for run, s in samples.items())
    clusters = verdict.attrs.get("n_clusters") if len(verdict) else None
    sd_true = magnitudes.attrs.get("sd_true", float("nan"))

    def _row(frame, name, column):
        hit = frame[frame["submission"] == name]
        return float(hit[column].iloc[0]) if len(hit) else float("nan")

    mean_level = (
        levels.groupby("model")["level_error"].mean().round(1).to_dict()
        if len(levels)
        else {}
    )
    rank_columns = [
        c
        for c in (
            "submission",
            "directional_pct",
            "spearman_rho",
            "pearson_r",
            "pearson_adj",
        )
        if c in board.columns
    ]

    lines = [
        f"# Four questions about {cont_name} against {base_name}",
        "",
        "[← main report](README.md)",
        "",
        f"The same {len(next(iter(samples.values()))):,} respondent profiles, sampled twice — "
        f"once with {base_name} and once with {cont_name}, about a 40x jump in",
        "parameters — and both scored against the same real responses. Same seeds, so",
        "the two samples are paired respondent by respondent, which is what lets the",
        "difference between them carry an interval rather than just a sign.",
        "",
        f"Respondents: {counts}.",
        "",
        "**Everything here is read against the human replication**, not against a",
        "perfect score. Human 2 predicting Human 1 is what a fresh sample of this size",
        "achieves, and it is the ceiling the pipeline is chasing — not 1.0.",
        "",
        "## The answers, in short",
        "",
        "| question | answer |",
        "| --- | --- |",
        f"| **1.** Over- or underestimating effect sizes? | **Both overestimate.** "
        f"{base_name} spreads its effects {_row(magnitudes, base_name, 'ratio_vs_true'):.1f}x too wide, "
        f"{cont_name} {_row(magnitudes, cont_name, 'ratio_vs_true'):.1f}x. Scaling up helped here. |",
        f"| **2.** Right rank order and direction? | **{base_name} partly; {cont_name} worse.** "
        f"Rank correlation {_row(board, base_name, 'spearman_rho'):.2f} against "
        f"{_row(board, cont_name, 'spearman_rho'):.2f}, with a real replication at "
        f"{_row(board, 'Human replication (Human 2)', 'spearman_rho'):.2f}. |",
        f"| **3.** Do predictions change with demographics? | **Yes, both.** Real partisans differ by "
        f"{_row(gap_summary.rename(columns={'model': 'submission'}), 'human (Human 1)', 'mean_abs_gap'):.1f} points; "
        f"{base_name} produces {_row(gap_summary.rename(columns={'model': 'submission'}), base_name, 'mean_abs_gap'):.1f} and "
        f"{cont_name} {_row(gap_summary.rename(columns={'model': 'submission'}), cont_name, 'mean_abs_gap'):.1f}. |",
        "| **4.** Using demographics to their advantage? | **No, neither.** The moderators a model "
        "could read predict its subgroup effects no better than the ones it could not. |",
        "",
        "And one precondition before any of them: the bigger model's answers sit far",
        f"closer to where real answers sit — mean absolute level error "
        f"{mean_level.get(base_name, float('nan')):.1f} -> {mean_level.get(cont_name, float('nan')):.1f} points on a 0-100 scale.",
        "",
        "In one line: **scaling the base model about 40x bought a much more realistic",
        "sample and no better prediction of which interventions work.**",
        "",
        "---",
        "",
        "## First, a precondition: are the answers even in the right place?",
        "",
        "Treatment effects are differences, so a constant bias cancels out of them and",
        "none of the four questions below would notice if every synthetic respondent",
        "sat 40 points off. Raw response distributions have no such mercy.",
        "",
        f"**Mean absolute level error fell from {mean_level.get(base_name, float('nan')):.1f} to "
        f"{mean_level.get(cont_name, float('nan')):.1f} points on a 0-100 scale.**",
        "This is the largest single change between the two models, and it is not",
        "marginal: on opposition to bipartisan cooperation the smaller model answered",
        "83 where real Americans answered 21, and the bigger one answers 37. On support",
        "for undemocratic candidates the smaller model said 18 against a real 53; the",
        "bigger one says 56.",
        "",
        (
            _md(
                levels[
                    [
                        "model",
                        "outcome",
                        "mean_human",
                        "mean_synthetic",
                        "level_error",
                        "variance_ratio",
                        "ovl",
                        "w1",
                    ]
                ].sort_values(["outcome", "model"]),
                floats=2,
            )
            if len(levels)
            else ""
        ),
        "",
        "`ovl` is the overlap between the two distributions (1 is identical) and `w1`",
        "the Wasserstein distance in scale points. Both improve on eight of nine",
        "outcomes.",
        "",
        "![Level error by outcome](plots/05_level_error.png)",
        "",
        "---",
        "",
        "## 1. Are the models over- or underestimating effect sizes?",
        "",
        "**Both overestimate, and the bigger model overestimates less.**",
        "",
        _md(magnitudes, floats=3),
        "",
        "The spread of the real between-arm effects, once sampling noise is removed,",
        f"is **{sd_true:.2f} percentage points** — these interventions genuinely do very",
        "little. Against that:",
        "",
        f"- {base_name} spreads its effects **{_row(magnitudes, base_name, 'ratio_vs_true'):.1f}x** too wide",
        f"- {cont_name} spreads them **{_row(magnitudes, cont_name, 'ratio_vs_true'):.1f}x** too wide",
        f"- even a real human replication looks **{_row(magnitudes, 'Human replication (Human 2)', 'ratio_vs_true'):.1f}x** too wide,"
        " because its own estimates carry sampling noise",
        "",
        "So the direction of the error is the same for both models — they make these",
        "messages look far more consequential than they are — and scaling up cut the",
        "exaggeration by roughly a third.",
        "",
        "**Why the calibration slope does not show this.** `beta` in the leaderboard is",
        f"{_row(board, base_name, 'beta'):.3f} for {base_name} and "
        f"{_row(board, cont_name, 'beta'):.3f} for {cont_name} — it got *worse*, which looks",
        "like a contradiction. It is not: beta is `cov(human, predicted) / var(predicted)`,",
        "so it moves with alignment as well as with magnitude. The bigger model's effects",
        "shrank toward the right size *and* became less aligned with the real ones, and",
        "the second effect is the larger. Which is question 2.",
        "",
        "---",
        "",
        "## 2. Do they predict the right rank order and direction?",
        "",
        f"**{base_name} gets part of the ordering right. {cont_name} does worse.**",
        "",
        _md(board[rank_columns], floats=3),
        "",
        "`spearman_rho` is the rank-order question in its purest form and",
        "`directional_pct` the sign question, with 50% as the no-information floor.",
        "",
        f"- rank order: {_row(board, base_name, 'spearman_rho'):.2f} for {base_name}, "
        f"{_row(board, cont_name, 'spearman_rho'):.2f} for {cont_name}, against "
        f"{_row(board, 'Human replication (Human 2)', 'spearman_rho'):.2f} for a real replication",
        f"- direction: {_row(board, base_name, 'directional_pct'):.0f}% and "
        f"{_row(board, cont_name, 'directional_pct'):.0f}%, against "
        f"{_row(board, 'Human replication (Human 2)', 'directional_pct'):.0f}% and a floor of 50%",
        "",
        f"{cont_name}'s directional agreement, {_row(board, cont_name, 'directional_pct'):.1f}%, is the same as",
        'the "predict every effect positive" baseline. That is the sharpest way to put',
        "it: on which interventions help, the bigger model carries about as much",
        "information as a constant guess.",
        "",
        "![Both models against human effects](plots/05_models_vs_human.png)",
        "",
        "The dashed lines remove each predictor's own sampling noise from the slope,",
        "which matters because attenuation depends only on the x-axis and each",
        "submission has a different amount of it. Read those, not the solid ones, when",
        "comparing submissions by eye.",
        "",
        "---",
        "",
        "## 3. Do the models change their predictions based on demographics?",
        "",
        "**Yes, both do — and the bigger model varies far more than real people.**",
        "",
        "This question is about whether the demographics move the answers at all,",
        "regardless of whether they move them correctly. The cleanest measure is the",
        "Republican-minus-Democrat gap in the control arm, before any intervention:",
        "party is named in every question of this instrument, so a model that ignores",
        "its assigned identity has no excuse for a flat gap.",
        "",
        _md(gaps, floats=1),
        "",
        _md(gap_summary, floats=2),
        "",
        f"- real partisans differ by **{_row(gap_summary.rename(columns={'model': 'submission'}), 'human (Human 1)', 'mean_abs_gap'):.1f} points** on average",
        f"- {base_name} produces **{_row(gap_summary.rename(columns={'model': 'submission'}), base_name, 'mean_abs_gap'):.1f}** — about the right size",
        f"- {cont_name} produces **{_row(gap_summary.rename(columns={'model': 'submission'}), cont_name, 'mean_abs_gap'):.1f}** — roughly three times too large",
        "",
        "So neither model is demographically inert on this instrument. The failure is",
        "in *how* they vary, not whether they vary — which is question 4.",
        "",
        "---",
        "",
        "## 4. Are they using demographics to their advantage?",
        "",
        "**No. Neither model turns the demographics it was given into a better",
        "prediction.** Two independent tests, and both come out the same way.",
        "",
        "### Test one: do the partisan gaps point the right way?",
        "",
        "Correlation between each model's nine party gaps and the real ones:",
        f"**{_row(gap_summary.rename(columns={'model': 'submission'}), base_name, 'corr_with_human_gaps'):.2f}** for {base_name}, "
        f"**{_row(gap_summary.rename(columns={'model': 'submission'}), cont_name, 'corr_with_human_gaps'):.2f}** for {cont_name}.",
        "",
        f"{cont_name} is the better of the two here, but look at the signs in the table",
        "above rather than the summary. Real respondents are *less* socially distant",
        "from the out-party if they are Republican (-9.1); both models say the",
        f"opposite, {base_name} mildly (-2.4 is at least the right sign) and {cont_name}",
        "confidently wrong (+9.5). Getting a gap of the right rough size pointed the",
        "wrong way is not an advantage.",
        "",
        "### Test two: do the moderators the model could *see* beat the ones it could not?",
        "",
        "This is the decisive test, and it needs no assumption about what the right",
        "answer is. Three moderators appear in the transcript — gender, race, party —",
        "and two never do: age and education came from the panel supplier and were never",
        "shown. A model that uses what it is told should predict subgroup effects better",
        "for the first group than the second. Neither does:",
        "",
        (
            _md(subgroups.drop(columns=["run"], errors="ignore"), floats=3)
            if len(subgroups)
            else ""
        ),
        "",
        f"- {base_name}: visible {_row(subgroups.rename(columns={'model': 'submission'}).query('moderators == \"visible\"'), base_name, 'pearson_r'):.3f} against invisible "
        f"{_row(subgroups.rename(columns={'model': 'submission'}).query('moderators == \"invisible\"'), base_name, 'pearson_r'):.3f} — no gap at all",
        f"- {cont_name}: visible {_row(subgroups.rename(columns={'model': 'submission'}).query('moderators == \"visible\"'), cont_name, 'pearson_r'):.3f} against invisible "
        f"{_row(subgroups.rename(columns={'model': 'submission'}).query('moderators == \"invisible\"'), cont_name, 'pearson_r'):.3f} — visible does *worse*",
        "",
        "If either model were reading its assigned identity to any useful effect, the",
        "visible row would beat the invisible one. For the smaller model the two are",
        "indistinguishable; for the bigger one the ordering is backwards. The",
        "demographic variation in question 3 is real, and it is noise.",
        "",
        "**The two models fail in opposite directions**, which is worth naming because",
        "the benchmark's diagnostics are built to catch only one of them. A model",
        "answering from a stereotype produces subgroup differences that are too large",
        "and too clean; a model ignoring its assigned identity produces almost none.",
        f"{cont_name} is the first kind and {base_name} closer to the second, so moving",
        "from one to the other is not simply progress.",
        "",
        "---",
        "",
        "## So did the bigger model win? The paired verdict",
        "",
        "Both models answered the same instrument about the same interventions and are",
        "scored against the same reference, so their errors move together. Each",
        "bootstrap draw resamples one set of intervention clusters and rescores *both*,",
        "which estimates the difference far more precisely than either score — and is",
        "why this table, not the leaderboard, is the verdict.",
        "",
    ]
    if len(verdict):
        lines += [
            _md(
                verdict[
                    [
                        "metric",
                        base_name,
                        cont_name,
                        "delta",
                        "delta_lo",
                        "delta_hi",
                        "p_contender_better",
                    ]
                ]
            ),
            "",
            f"Clusters resampled: {clusters}. `delta` is signed raw, so on `rmse` — the",
            "one row where lower is better — a negative delta is the improvement.",
            "",
        ]
        settled = [
            m
            for m in verdict["metric"]
            if not 0.025
            < float(verdict.loc[verdict["metric"] == m, "p_contender_better"].iloc[0])
            < 0.975
        ]
        if settled:
            lines += [_verdict_sentence(verdict, m, cont_name) for m in settled]
        else:
            lines += [
                "Every row is unsettled — no metric's resamples agree on a direction —"
                " so there are no per-metric conclusions to list.",
            ]
        lines += [
            "",
            "**Nothing clears its interval.** With six intervention clusters to resample,",
            "this study cannot certify a difference of the size at stake, so the honest",
            'reading of the effect metrics is "no improvement, possibly a regression",',
            "not a demonstrated regression.",
            "",
            "The one row that comes close is `rmse`, and it is the row most likely to be",
            "misread. It improved because the exaggeration shrank (question 1), not",
            "because the ordering improved (question 2) — squared error rewards a",
            "predictor for moving toward the right scale even as its ranking degrades.",
            "Note also that predicting **no effect at all** scores 1.537 on this metric,",
            "better than either model: when the true effects are barely larger than the",
            "noise, shrinking everything to zero is close to optimal.",
            "",
            "![Paired change per metric](plots/05_paired_change.png)",
            "",
        ]
    else:
        lines += ["Not enough shared intervention clusters to pair on.", ""]

    lines += [
        "---",
        "",
        "## What this implies for the pipeline",
        "",
        "The three things that improved — level accuracy, exaggeration, demographic",
        "responsiveness — are all properties of *how a respondent answers in isolation*,",
        "and those are exactly what a better language model should be expected to fix.",
        "The thing that did not improve is the one the megastudy actually scores:",
        "whether the sample can tell the interventions apart. That is a claim about",
        "counterfactual sensitivity to a paragraph of text, and 40x more parameters",
        "bought none of it.",
        "",
        "So the next thing worth trying is probably not a bigger model. It is a change",
        "to what the model is asked to do — conditioning it more strongly on the",
        "stimulus, or abandoning single-pass simulation for something that reasons about",
        "the message before answering.",
        "",
        "---",
        "",
        "## What the samplers did",
        "",
        _md(diag, floats=4) if len(diag) else "No run metadata found.",
        "",
        "## Caveats",
        "",
        "1. **The two runs differ in KV-cache precision.** DeepSeek-V4-Flash requires",
        "   `fp8_ds_mla` — on an H200 vLLM selects its FlashMLA attention, whose paged",
        "   layout *is* the fp8 format, and it will not start with anything else. The",
        "   checkpoint ships the UE8M0 scales, so this is the precision the model was",
        "   built to run at rather than a compromise, but the Qwen run used bf16 KV and",
        "   that asymmetry cannot be ruled out as a contributor.",
        "2. **Six intervention clusters.** The pure-text rule left 6 of 27 arms, so every",
        "   interval here is wide and the paired bootstrap resamples six things. That is",
        "   why the verdict table settles nothing.",
        "3. **A subset the paper never reports.** Dropping the non-textual arms means the",
        "   human reference is not the study's headline result.",
        "4. **2022 sits inside both models' training windows.** Neither result should be",
        "   read as a clean out-of-sample prediction.",
        "",
    ]
    (out / "05_model_comparison.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
