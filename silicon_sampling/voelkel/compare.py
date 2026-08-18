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
    """Paired improvement per metric, with its 95% interval.

    Signs are flipped for metrics where lower is better, so *positive is always
    better* and the reader does not have to hold two conventions at once.  The raw
    signed deltas stay in the table beside this figure.
    """
    viz.style()
    import matplotlib.pyplot as plt

    frame = verdict.copy()
    direction = np.where(frame["higher_is_better"], 1.0, -1.0)
    frame["improvement"] = frame["delta"] * direction
    # Flipping the sign swaps which end of the interval is which.
    frame["lo"] = np.minimum(
        frame["delta_lo"] * direction, frame["delta_hi"] * direction
    )
    frame["hi"] = np.maximum(
        frame["delta_lo"] * direction, frame["delta_hi"] * direction
    )

    # Small multiples, one panel per metric.  These are measures of different
    # scale — directional agreement in percentage points, correlations in
    # correlation units, RMSE in scale points — so a shared x-axis buries the
    # correlation deltas beside a swing of 11 points, and a second axis is never
    # the answer.  Each panel gets its own scale and the title says so.
    fig, axes = plt.subplots(
        len(frame), 1, figsize=(6.8, 1.05 * len(frame) + 1.5), squeeze=False
    )
    # Each panel carries its own tick labels below it, so the panels need real
    # separation or those labels land on the next panel's marks.
    fig.subplots_adjust(hspace=1.15)
    for ax, row in zip(axes[:, 0], frame.itertuples()):
        span = max(abs(row.lo), abs(row.hi), abs(row.improvement)) or 1.0
        ax.axvline(0, color=viz.TEXT_MUTED, lw=1.2)
        ax.plot([row.lo, row.hi], [0, 0], color=viz.CATEGORICAL[0], lw=2.4, alpha=0.5)
        ax.scatter(
            [row.improvement],
            [0],
            s=58,
            color=viz.CATEGORICAL[0],
            edgecolor=viz.SURFACE,
            lw=1.2,
            zorder=3,
        )
        ax.annotate(
            f"{row.improvement:+.3f}",
            (row.improvement, 0),
            textcoords="offset points",
            xytext=(0, 11),
            ha="center",
            fontsize=9,
            color=viz.TEXT_SECONDARY,
        )
        ax.set_xlim(-1.35 * span, 1.35 * span)
        ax.set_ylim(-0.5, 0.9)
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
        f"Paired improvement of {contender} over {baseline}\n"
        "95% cluster bootstrap · positive = more faithful · own scale per metric",
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
            plots / "05_paired_improvement.png",
        )
    if len(levels):
        _level_plot(levels, plots / "05_level_error.png")

    _write(
        out,
        board,
        verdict,
        levels,
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

    mean_level = (
        levels.groupby("model")["level_error"].mean().round(1).to_dict()
        if len(levels)
        else {}
    )
    level_line = " · ".join(f"{name} {value}" for name, value in mean_level.items())

    lines = [
        "# Does a bigger base model sample more faithfully?",
        "",
        "[← main report](README.md)",
        "",
        f"{base_name} against {cont_name}, both scored against the same real",
        f"responses. Respondents: {counts}. Both models sampled the *same* profiles",
        "with the same seeds, so the comparison is paired respondent by respondent.",
        "",
        "## The leaderboard",
        "",
        _md(
            board[
                [
                    c
                    for c in (
                        "submission",
                        "n_pairs",
                        "directional_pct",
                        "pearson_r",
                        "pearson_adj",
                        "rmse",
                        "alpha",
                        "beta",
                        "beta_adj",
                    )
                    if c in board.columns
                ]
            ]
        ),
        "",
        "Read every row against the human replication, not against 1.0: a real",
        "replication of this size is the ceiling the pipeline is chasing.",
        "",
        "## Did it improve? The paired answer",
        "",
        "Both models answered the same instrument about the same interventions and",
        "are scored against the same reference, so their errors move together. Each",
        "bootstrap draw resamples one set of intervention clusters and rescores",
        "*both* — which estimates the difference far more precisely than either",
        "score, and is why this table, not the leaderboard above, is the verdict.",
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
            f"Clusters resampled: {clusters}.",
            "",
        ]
        lines += [_verdict_sentence(verdict, m, cont_name) for m in verdict["metric"]]
        lines += ["", "![Paired improvement](plots/05_paired_improvement.png)", ""]
    else:
        lines += ["Not enough shared intervention clusters to pair on.", ""]

    lines += [
        "![Both models against human effects](plots/05_models_vs_human.png)",
        "",
        "## Levels, not just effects",
        "",
        "Treatment effects are differences, so a constant bias cancels out of them.",
        "Raw response distributions have no such mercy, and this is where the",
        f"{base_name} sample failed worst.",
        "",
        f"Mean absolute level error across outcomes — {level_line} points on a 0-100 scale.",
        "",
    ]
    if len(levels):
        lines += [
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
            ),
            "",
            "![Level error by outcome](plots/05_level_error.png)",
            "",
        ]

    lines += [
        "## Does it condition on who it is supposed to be?",
        "",
        "The sharpest failure of the smaller model was demographic flatness: it",
        "wrote a party identity into the transcript and then answered as if it had",
        "not. If scale fixes anything, the *visible* moderators should now beat the",
        "invisible ones — the model can read the first group and cannot read the",
        "second, so a real conditioning effect has to show up as a gap.",
        "",
    ]
    if len(subgroups):
        lines += [_md(subgroups), ""]
    lines += [
        "Subgroup treatment effects are a hard and noisy target, though, so the",
        "cleaner test is whether the model puts partisans in different places at all,",
        "before any intervention. Party is named in every question of this",
        "instrument, so a flat gap has no excuse:",
        "",
        _md(gaps, floats=1),
        "",
        _md(gap_summary, floats=2),
        "",
        "**The two models fail in opposite directions.** Read `mean_abs_gap` against",
        "the human row: one sample is too flat, the other too stereotyped. A model",
        "answering from a stereotype produces subgroup differences that are too large",
        "and too clean; a model ignoring its assigned identity produces almost none.",
        "The benchmark's diagnostics are built to catch the first, and the second is",
        "the more damaging of the two for subgroup estimates — so moving from one to",
        "the other is not simply progress.",
        "",
    ]
    lines += [
        "## What the samplers did",
        "",
        _md(diag, floats=4) if len(diag) else "No run metadata found.",
        "",
    ]
    (out / "05_model_comparison.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
