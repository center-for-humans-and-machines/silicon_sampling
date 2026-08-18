"""Qwen2.5-7B against DeepSeek-V4-Flash on the Pfänder instrument.

There is no human data here — the megastudy publishes none, by design — so
nothing in this file can say which model is *right*.  What it can say is three
things that matter anyway:

1. **Do the two models agree on which messages work?**  If the ranking of the 16
   interventions is model-independent, it is more likely to be a property of the
   messages than of the sampler.  If it is not, then whichever model we submit is
   the finding, which would be worth knowing before submitting either.
2. **Does the bigger model condition on its assigned demographics?**  This is the
   sharpest failure of the smaller one: it writes a party identity, an income and
   an education into the transcript and then answers as if it had not, leaving
   subgroup differences near zero where real US survey data has some of the
   largest gaps in public opinion.  Scale either fixes this or it does not.
3. **Does it still look like survey data?**  Degeneracy, straightlining and scale
   reliability, side by side.

The faithfulness question proper is answered in the Voelkel report, which has
real responses to score against.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..analysis import distributions as dist
from ..analysis import effects as eff
from ..analysis import moderators as mod
from ..analysis import plotting as viz
from ..models import label as model_label
from . import outcomes as oc
from .paths import REPORT, samples_dir

MODERATORS = tuple(oc.MODERATORS)
PRIMARY = "trust_multidimensional"
CONTROL = "control"

#: Held stable with the Voelkel figures so a reader can carry identity across
#: reports.  Validated as a categorical pair plus red on the light surface.
COLOURS = {"qwen25_7b": viz.CATEGORICAL[0], "v4_flash": viz.CATEGORICAL[3]}
MARKERS = {"qwen25_7b": "o", "v4_flash": "D"}


def load_runs(runs) -> dict[str, pd.DataFrame]:
    """Each run's ``samples.csv``, keyed by run key."""
    loaded = {}
    for run in runs:
        path = samples_dir(run) / "samples.csv"
        if not path.exists():
            raise SystemExit(f"no sample at {path}; run build-csv for {run} first")
        loaded[run] = pd.read_csv(path, low_memory=False)
    return loaded


def effect_tables(samples: dict) -> dict[str, pd.DataFrame]:
    """Intervention effects for every outcome, per model."""
    return {
        run: eff.ate_table(sample, dict(oc.SCALE_RANGE), control=CONTROL)
        for run, sample in samples.items()
    }


def effect_agreement(tables: dict, baseline: str, contender: str) -> pd.DataFrame:
    """Do the models rank the interventions the same way, outcome by outcome?

    Correlated across the 16 interventions within each outcome, so a shared
    "everything helps a bit" level cannot manufacture agreement.  ``sd_ratio``
    says whether the contender's effects are more or less spread out; a model that
    agrees on the ordering but exaggerates the magnitudes is a different animal
    from one that disagrees.
    """
    left, right = tables[baseline], tables[contender]
    rows = []
    for outcome in oc.OUTCOMES:
        a = left[left["outcome"] == outcome].set_index("condition")["estimate"]
        b = right[right["outcome"] == outcome].set_index("condition")["estimate"]
        shared = a.index.intersection(b.index)
        if len(shared) < 3:
            continue
        x, y = a[shared].to_numpy(float), b[shared].to_numpy(float)
        keep = np.isfinite(x) & np.isfinite(y)
        if keep.sum() < 3 or np.std(x[keep]) == 0 or np.std(y[keep]) == 0:
            continue
        rows.append(
            {
                "outcome": outcome,
                "n_interventions": int(keep.sum()),
                "pearson_r": float(np.corrcoef(x[keep], y[keep])[0, 1]),
                "spearman_rho": float(
                    pd.Series(x[keep]).corr(pd.Series(y[keep]), method="spearman")
                ),
                f"mean_{baseline}": float(np.mean(x[keep])),
                f"mean_{contender}": float(np.mean(y[keep])),
                f"sd_{baseline}": float(np.std(x[keep], ddof=1)),
                f"sd_{contender}": float(np.std(y[keep], ddof=1)),
                "sd_ratio": float(np.std(y[keep], ddof=1) / np.std(x[keep], ddof=1)),
                "sign_agreement": float(np.mean(np.sign(x[keep]) == np.sign(y[keep]))),
            }
        )
    return pd.DataFrame(rows)


def demographic_signal(samples: dict) -> pd.DataFrame:
    """How much variance each moderator explains beyond condition, per model.

    ``predictability`` fits ``outcome ~ moderator + condition`` against
    ``outcome ~ condition`` alone, so this is the demographic signal *net* of the
    treatment.  For the Qwen sample the largest value over all six moderators and
    all thirteen outcomes was R² = 0.002 — flat enough that every subgroup
    estimate the submission makes is close to a constant.
    """
    rows = []
    for run, sample in samples.items():
        table = mod.predictability(sample, MODERATORS, list(oc.OUTCOMES))
        if not len(table):
            continue
        best = table.loc[table["r2_moderator"].idxmax()]
        rows.append(
            {
                "model": model_label(run),
                "run": run,
                "max_r2_moderator": float(best["r2_moderator"]),
                "max_at": f"{best['moderator']} on {best['outcome']}",
                "mean_r2_moderator": float(table["r2_moderator"].mean()),
                "median_r2_moderator": float(table["r2_moderator"].median()),
            }
        )
    return pd.DataFrame(rows)


def partisan_gap(samples: dict, outcome: str = "belief_post") -> pd.DataFrame:
    """Republican minus Democrat mean, the gap real US data makes enormous.

    Belief in human-caused climate change is one of the largest and most reliable
    partisan divides in American public opinion — routinely tens of points. A
    synthetic sample that puts it near zero is not modelling the respondent it was
    told to be.
    """
    rows = []
    for run, sample in samples.items():
        if outcome not in sample.columns or "party" not in sample.columns:
            continue
        values = pd.to_numeric(sample[outcome], errors="coerce")
        means = values.groupby(sample["party"]).mean()
        if not {"Republican", "Democrat"} <= set(means.index):
            continue
        rows.append(
            {
                "model": model_label(run),
                "run": run,
                "outcome": outcome,
                "republican": float(means["Republican"]),
                "democrat": float(means["Democrat"]),
                "gap": float(means["Republican"] - means["Democrat"]),
            }
        )
    return pd.DataFrame(rows)


def survey_likeness(samples: dict, batteries: dict) -> pd.DataFrame:
    """Degeneracy, straightlining and reliability — is it survey-shaped at all?"""
    rows = []
    for run, sample in samples.items():
        degeneracy = dist.degeneracy(sample, [PRIMARY])
        reliab = dist.scale_reliabilities(sample, batteries)
        flat = dist.straightlining(sample, batteries)
        rows.append(
            {
                "model": model_label(run),
                "run": run,
                "primary_modal_share": float(degeneracy["modal_share"].iloc[0]),
                "primary_multiple_of_10": float(
                    degeneracy["share_multiple_of_10"].iloc[0]
                ),
                "mean_alpha": float(reliab["alpha"].mean()),
                "mean_share_flat": float(flat["share_flat"].mean()),
            }
        )
    return pd.DataFrame(rows)


def diagnostics(runs) -> pd.DataFrame:
    """What each sampler did."""
    rows = []
    for run in runs:
        path = samples_dir(run) / "run_meta.json"
        if not path.exists():
            continue
        meta = json.loads(path.read_text(encoding="utf-8"))
        draws = meta.get("draws", {})
        engine = meta.get("engine") or {}
        rows.append(
            {
                "model": model_label(run),
                "hf_id": meta.get("model"),
                "n": meta.get("sampled"),
                "hours": round((meta.get("seconds") or 0) / 3600, 2),
                "respondents_per_hour": meta.get("respondents_per_hour"),
                "gpus": engine.get("tensor_parallel_size", 1),
                "rejection_rate": draws.get("rejection_rate"),
                "structured_fallbacks": draws.get("structured_fallbacks"),
                "forced_defaults": draws.get("forced"),
            }
        )
    return pd.DataFrame(rows)


def _effect_scatter(tables: dict, baseline: str, contender: str, path: Path) -> None:
    """One model's intervention effects against the other's, on the primary outcome.

    Both axes are the same quantity measured by different samplers, so the
    identity line really is the target here — unlike the Voelkel figure, where the
    axes are a prediction and a noisy human measurement of it.
    """
    viz.style()
    import matplotlib.pyplot as plt

    left = tables[baseline]
    right = tables[contender]
    a = left[left["outcome"] == PRIMARY].set_index("condition")["estimate"]
    b = right[right["outcome"] == PRIMARY].set_index("condition")["estimate"]
    shared = a.index.intersection(b.index)
    x, y = a[shared].to_numpy(float), b[shared].to_numpy(float)

    span = float(np.nanmax(np.abs(np.concatenate([x, y])))) * 1.2 or 1.0
    fig, ax = plt.subplots(figsize=(6.0, 5.8))
    ax.set_axisbelow(True)
    ax.axhline(0, color=viz.TEXT_MUTED, lw=0.8)
    ax.axvline(0, color=viz.TEXT_MUTED, lw=0.8)
    ax.plot(
        [-span, span],
        [-span, span],
        ls="--",
        color=viz.TEXT_MUTED,
        lw=1.0,
        label="identity",
    )
    ax.scatter(
        x,
        y,
        s=46,
        color=viz.CATEGORICAL[0],
        edgecolor=viz.SURFACE,
        lw=1.2,
        zorder=3,
        label="intervention",
    )
    # Label only the extremes: a number on all 16 points would be unreadable, and
    # the ones worth naming are the ones the models disagree about most.
    disagreement = np.abs(y - x)
    for index in np.argsort(disagreement)[-3:]:
        ax.annotate(
            str(shared[index]),
            (x[index], y[index]),
            textcoords="offset points",
            xytext=(6, 5),
            fontsize=8,
            color=viz.TEXT_SECONDARY,
        )
    ax.set_xlim(-span, span)
    ax.set_ylim(-span, span)
    ax.set_xlabel(f"{model_label(baseline)} effect (pp)")
    ax.set_ylabel(f"{model_label(contender)} effect (pp)")
    ax.set_title(f"Intervention effects on {PRIMARY}")
    ax.legend(fontsize=9, loc="upper left")
    viz.save(fig, path)


def _md(frame: pd.DataFrame, floats: int = 3) -> str:
    from .report import md_table

    return md_table(frame, floats=floats)


def generate(
    runs,
    out: Path = REPORT,
    baseline: str = "qwen25_7b",
    contender: str = "v4_flash",
) -> dict:
    """Write ``05_model_comparison.md`` and its figure."""
    from .report import BATTERIES

    out.mkdir(parents=True, exist_ok=True)
    plots = out / "plots"
    plots.mkdir(parents=True, exist_ok=True)

    samples = load_runs(runs)
    tables = effect_tables(samples)
    agreement = effect_agreement(tables, baseline, contender)
    demographics = demographic_signal(samples)
    gap = partisan_gap(samples)
    likeness = survey_likeness(samples, BATTERIES)
    diag = diagnostics(runs)

    _effect_scatter(tables, baseline, contender, plots / "05_effect_agreement.png")

    base_name, cont_name = model_label(baseline), model_label(contender)
    primary_row = agreement[agreement["outcome"] == PRIMARY]
    primary_r = (
        float(primary_row["pearson_r"].iloc[0]) if len(primary_row) else float("nan")
    )

    lines = [
        f"# {base_name} against {cont_name}",
        "",
        "[← main report](README.md)",
        "",
        "The Pfänder megastudy publishes no human data, so nothing here says which",
        "model is *right* — that question is answered in the",
        "[Voelkel validation](../voelkel_validation/05_model_comparison.md), which has",
        "real responses to score against. What this can say is whether the two",
        "models agree, whether the bigger one reads its assigned demographics, and",
        "whether it still produces something survey-shaped.",
        "",
        "Both models sampled the same 18,000 profiles with the same seeds.",
        "",
        "## Do they agree on which messages work?",
        "",
        "On the primary outcome the two models' 16 intervention effects correlate at",
        f"**r = {primary_r:.3f}**.",
        "",
        _md(agreement),
        "",
        "`sd_ratio` above 1 means the contender spreads the interventions further",
        "apart than the baseline did. That matters on its own: the Voelkel check",
        "found the smaller model's effects **2.6x too spread out**, so more spread",
        "here is not automatically better.",
        "",
        "![Effect agreement](plots/05_effect_agreement.png)",
        "",
        "## Does it condition on who it is supposed to be?",
        "",
        "This is the smaller model's sharpest failure. It writes a party identity, an",
        "income and an education into the transcript and then answers the rest of the",
        "questionnaire as if it had not.",
        "",
        "Variance a moderator explains *beyond condition*, over all six moderators",
        "and all thirteen outcomes:",
        "",
        _md(demographics, floats=5),
        "",
        "And the single sharpest case — belief in human-caused climate change by",
        "party, where real US survey data shows one of the largest and most reliable",
        "gaps in public opinion, routinely tens of points:",
        "",
        _md(gap, floats=2),
        "",
        "## Does it still look like survey data?",
        "",
        _md(likeness, floats=4),
        "",
        "`primary_modal_share` is the fraction of respondents giving the single most",
        "common answer on the primary outcome; `mean_share_flat` the fraction giving",
        "an identical answer to every item of a battery. A model that has stopped",
        "behaving like a respondent shows up in these before it shows up anywhere else.",
        "",
        "## What the samplers did",
        "",
        _md(diag, floats=4) if len(diag) else "No run metadata found.",
        "",
    ]
    (out / "05_model_comparison.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    return {
        "agreement": agreement,
        "demographics": demographics,
        "partisan_gap": gap,
        "survey_likeness": likeness,
        "diagnostics": diag,
    }
