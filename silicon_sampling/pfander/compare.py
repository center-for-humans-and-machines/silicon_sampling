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
    _write(
        out,
        agreement,
        demographics,
        gap,
        likeness,
        diag,
        baseline,
        contender,
        {run: len(sample) for run, sample in samples.items()},
    )
    return {
        "agreement": agreement,
        "demographics": demographics,
        "partisan_gap": gap,
        "survey_likeness": likeness,
        "diagnostics": diag,
    }


def _write(
    out,
    agreement,
    demographics,
    gap,
    likeness,
    diag,
    baseline,
    contender,
    n_by_run,
) -> None:
    base_name, cont_name = model_label(baseline), model_label(contender)

    def _cell(frame, key, column, key_column="model"):
        hit = frame[frame[key_column] == key]
        return float(hit[column].iloc[0]) if len(hit) else float("nan")

    primary = agreement[agreement["outcome"] == PRIMARY]
    primary_r = float(primary["pearson_r"].iloc[0]) if len(primary) else float("nan")
    median_r = (
        float(agreement["pearson_r"].median()) if len(agreement) else float("nan")
    )
    median_sd_ratio = (
        float(agreement["sd_ratio"].median()) if len(agreement) else float("nan")
    )
    counts = " · ".join(f"{model_label(r)}: {n:,}" for r, n in n_by_run.items())

    lines = [
        "# The same four questions, on the Pfänder instrument",
        "",
        "[← main report](README.md)",
        "",
        f"{base_name} against {cont_name} over the same 18,000 respondent profiles with",
        "the same seeds, so the two samples are paired respondent by respondent.",
        f"Respondents: {counts}.",
        "",
        "**The megastudy publishes no human data, by design.** So this report cannot say",
        "which model is *right* about anything. Two of the four questions still get a",
        "full answer here, one gets a useful partial answer, and one has to be sent",
        "next door to the [Voelkel validation](../voelkel_validation/05_model_comparison.md),",
        "which has real responses to score against.",
        "",
        "## The answers, in short",
        "",
        "| question | can this study answer it? | what it says |",
        "| --- | --- | --- |",
        f"| **1.** Over- or underestimating effect sizes? | Partly — no truth to compare to, "
        f"but the two models can be compared to each other | {cont_name}'s effects are much "
        f"smaller: median spread ratio **{median_sd_ratio:.2f}** of {base_name}'s. Voelkel says "
        "both still overestimate, so this is a move in the right direction. |",
        f"| **2.** Right rank order and direction? | No — but it can ask whether the ranking "
        f"is a property of the *messages* or of the *model* | **It is the model.** The two "
        f"models' 16 intervention effects correlate at **r = {primary_r:.2f}** on the primary "
        f"outcome (median {median_r:.2f} across outcomes). At most one of them can be right. |",
        f"| **3.** Do predictions change with demographics? | Yes, fully | **Much more than "
        f"before.** Largest variance a moderator explains beyond condition: "
        f"**{_cell(demographics, base_name, 'max_r2_moderator'):.4f} -> "
        f"{_cell(demographics, cont_name, 'max_r2_moderator'):.4f}**. |",
        "| **4.** Using demographics to their advantage? | **No — needs ground truth** | See "
        "Voelkel: neither model's demographic variation points the right way. |",
        "",
        "---",
        "",
        "## 1. How big are the effects each model reports?",
        "",
        f"**{cont_name} reports much smaller effects than {base_name} on almost every",
        "outcome.** Without human data this cannot be scored, but it is the same",
        "direction the Voelkel check measures against real responses, where both models",
        "overestimate and the bigger one overestimates less.",
        "",
        _md(agreement, floats=3),
        "",
        "`sd_ratio` is the spread of the contender's 16 intervention effects over the",
        f"baseline's. Below 1 means {cont_name} spreads the messages less far apart. It is",
        f"below 1 on {int((agreement['sd_ratio'] < 1).sum())} of {len(agreement)} outcomes,",
        f"with a median of **{median_sd_ratio:.2f}** — so the bigger model thinks these",
        "messages do roughly a third to two thirds of what the smaller one thought.",
        "",
        f"On the primary outcome the mean effect falls from "
        f"**{_cell(agreement, PRIMARY, f'mean_{baseline}', 'outcome'):.2f}** to "
        f"**{_cell(agreement, PRIMARY, f'mean_{contender}', 'outcome'):.2f}** scale points.",
        "",
        "---",
        "",
        "## 2. Do the two models agree on which messages work?",
        "",
        "**Barely — which means the intervention ranking is a property of the sampler,",
        "not of the messages.**",
        "",
        "This is the strongest thing this study can say about rank order without human",
        "data. If two samplers agreed closely, the ranking would at least be a stable",
        "feature of the stimuli. They do not:",
        "",
        f"- primary outcome (`{PRIMARY}`): **r = {primary_r:.2f}**, rank correlation "
        f"{float(primary['spearman_rho'].iloc[0]) if len(primary) else float('nan'):.2f}",
        f"- across all 13 outcomes the median correlation is **{median_r:.2f}**, ranging from "
        f"{agreement['pearson_r'].min():.2f} to {agreement['pearson_r'].max():.2f}",
        f"- on {int((agreement['pearson_r'] < 0).sum())} of {len(agreement)} outcomes the two "
        "models are *negatively* correlated — they disagree about the sign of the ranking",
        "",
        "So at most one of these two samples is tracking the real ordering, and the",
        "Voelkel scoring says that whichever it is, it is not doing it well: rank",
        "correlation with real effects was 0.31 for the smaller model and 0.19 for the",
        "bigger one, against 0.40 for a fresh human sample.",
        "",
        "**What this means for a submission.** The 16-message ranking this pipeline",
        "produces should not be read as a property of the messages. Change the base",
        "model and you get a substantially different ranking, with no way to tell from",
        "inside the study which one to believe.",
        "",
        "![Effect agreement on the primary outcome](plots/05_effect_agreement.png)",
        "",
        "`sign_agreement` in the table above is the fraction of the 16 interventions the",
        "two models at least push in the same direction — high on the trust outcomes,",
        "near chance on the policy ones.",
        "",
        "---",
        "",
        "## 3. Do the models change their predictions based on demographics?",
        "",
        f"**Yes, and {cont_name} does so far more than {base_name}.** This was the",
        "smaller model's sharpest failure: it wrote a party identity, an income and an",
        "education into the transcript and then answered as if it had not.",
        "",
        "Variance a moderator explains *beyond condition*, across all six moderators and",
        "all thirteen outcomes:",
        "",
        _md(demographics.drop(columns=["run"], errors="ignore"), floats=5),
        "",
        "And the sharpest single case — belief in human-caused climate change by party,",
        "one of the largest and most reliable divides in US public opinion, routinely",
        "tens of points:",
        "",
        _md(gap.drop(columns=["run"], errors="ignore"), floats=2),
        "",
        f"{base_name} produced a **{abs(_cell(gap, base_name, 'gap')):.1f}-point** gap where",
        "reality has tens. That was the finding that made the first sample's subgroup",
        f"estimates close to constants. {cont_name} produces",
        f"**{abs(_cell(gap, cont_name, 'gap')):.1f} points** — still short of the real divide, but an",
        "order of magnitude closer, and in the right direction.",
        "",
        "---",
        "",
        "## 4. Are they using demographics to their advantage?",
        "",
        "**This study cannot tell**, and it is worth being clear about why rather than",
        "reaching for a proxy. Question 3 shows the demographics move the answers; whether",
        "they move them *correctly* needs real subgroup responses to compare against, and",
        "the megastudy publishes none.",
        "",
        "[The Voelkel validation answers it](../voelkel_validation/05_model_comparison.md):",
        "**no, neither model.** The moderators a model could read in the transcript",
        "predict its subgroup effects no better than the two it never saw — dead even for",
        f"{base_name}, and backwards for {cont_name}. So the larger demographic",
        "responsiveness measured above should be read as larger variation, not better",
        "variation, until something demonstrates otherwise.",
        "",
        "---",
        "",
        "## Does it still look like survey data?",
        "",
        "A model that has stopped behaving like a respondent shows up here before it",
        "shows up in any effect estimate.",
        "",
        _md(likeness.drop(columns=["run"], errors="ignore"), floats=4),
        "",
        "`primary_modal_share` is the fraction of respondents giving the single most",
        "common answer on the primary outcome, `mean_share_flat` the fraction giving an",
        "identical answer to every item of a battery, and `mean_alpha` the average",
        "internal consistency of the multi-item scales.",
        "",
        f"Both samples pass. {cont_name} straightlines somewhat more "
        f"({_cell(likeness, cont_name, 'mean_share_flat'):.1%} of battery profiles flat against "
        f"{_cell(likeness, base_name, 'mean_share_flat'):.1%}) and rounds to multiples of ten "
        "slightly more often, neither at a level that would make the sample unusable. Scale",
        "reliability is essentially unchanged.",
        "",
        "---",
        "",
        "## What the samplers did",
        "",
        _md(diag, floats=4) if len(diag) else "No run metadata found.",
        "",
        "## Caveats",
        "",
        "1. **No human data, by design.** Nothing here is validated against ground truth;",
        "   questions 2 and 4 are answered next door or not at all.",
        "2. **The two runs differ in KV-cache precision.** DeepSeek-V4-Flash requires",
        "   `fp8_ds_mla` — vLLM's FlashMLA attention for this model *is* the fp8 layout and",
        "   will not start otherwise — while the Qwen run used bf16 KV. The checkpoint",
        "   ships the UE8M0 scales, so this is the model's native precision rather than a",
        "   compromise, but the asymmetry cannot be ruled out as a contributor.",
        "3. **Gender, age and race were pre-filled** from the preregistered census quotas;",
        "   education, income and party were generated by the model. The moderator",
        "   analysis above mixes both kinds.",
        "4. **The throughput row covers only the final resumed pass**, not the whole run:",
        "   a job killed at its wall-time limit writes no `run_meta.json`, so the",
        "   respondents-per-hour and rejection figures describe the segment that finished",
        "   cleanly.",
        "",
    ]
    (out / "05_model_comparison.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
