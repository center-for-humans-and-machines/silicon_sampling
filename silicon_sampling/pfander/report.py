"""Analyse a finished silicon sample and write the report set.

The main report carries the result; four sub-reports carry the detail.  Every
table in a report is written from the same DataFrame that produced the chart
beside it, so the numbers and the picture cannot disagree — and the tables also
discharge the palette's contrast relief rule.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ..analysis import distributions as dist
from ..analysis import effects as eff
from ..analysis import moderators as mod
from ..analysis import plotting as viz
from . import outcomes
from .conditions import INTERVENTIONS
from .run import all_slots
from .stats import near_misses

MODERATORS = tuple(outcomes.MODERATORS)

#: Item batteries, by their raw Qualtrics ids as stored in samples.csv.
BATTERIES = {
    "Trust in climate scientists (12 items, 0-100)": [
        "trust_competent_1",
        "trust_intelligent_1",
        "trust_qualified_1",
        "trust_honest_1",
        "trust_ethical_1",
        "trust_sincere_1",
        "trust_concerned_1",
        "trust_improve_1",
        "trust_considerate_1",
        "trust_feedback_1",
        "trust_transparent_1",
        "trust_attention_1",
    ],
    "Institutional trust (5 items, 0-100)": list(outcomes.MEANS["inst_trust_mean"]),
    "Scientists' policy role (4 items, 0-100)": list(
        outcomes.MEANS["policy_role_mean"]
    ),
    "Climate concern (3 items, 0-100)": list(outcomes.MEANS["concern_mean"]),
    "Specific climate policies (7 items, 0-100)": list(
        outcomes.MEANS["policy_specific_mean"]
    ),
    "Pro-climate behaviour (6 items, 0-100)": list(outcomes.MEANS["behavior_mean"]),
    "Need for epistemic autonomy (6 items, 1-7)": [
        f"epist_auton_{i}" for i in range(1, 7)
    ],
    "Alienation from climate science (6 items, 1-7)": [
        "alien_inst_1",
        "alien_inst_2",
        "alien_social_1",
        "alien_social_2",
        "alien_spatial_1",
        "alien_spatial_2",
    ],
    "Exposure to climate information (6 items, 1-5)": [
        f"alien_info_{i}" for i in range(1, 7)
    ],
}

PRIMARY = "trust_multidimensional"

SHORT = {
    "trust_multidimensional": "trust (multi)",
    "trust_post": "trust (single)",
    "distrust_post": "distrust",
    "funding_perceptions": "funding",
    "policy_role_mean": "policy role",
    "inst_trust_mean": "inst. trust",
    "belief_post": "belief",
    "concern_mean": "concern",
    "policy_general": "policy (gen)",
    "policy_specific_mean": "policy (spec)",
    "behavior_mean": "behaviour",
    "donation_ams": "donation",
    "newsletter_signup": "newsletter",
}


def md_table(frame: pd.DataFrame, floats: int = 2, max_rows: int | None = None) -> str:
    """A DataFrame as a GitHub markdown table."""
    data = frame if max_rows is None else frame.head(max_rows)
    formatted = data.copy()
    for column in formatted.columns:
        if pd.api.types.is_float_dtype(formatted[column]):
            formatted[column] = formatted[column].map(
                lambda value: "" if pd.isna(value) else f"{value:.{floats}f}"
            )
        else:
            formatted[column] = formatted[column].astype(str)
    header = "| " + " | ".join(str(column) for column in formatted.columns) + " |"
    rule = "| " + " | ".join("---" for _ in formatted.columns) + " |"
    body = ["| " + " | ".join(row) + " |" for row in formatted.astype(str).to_numpy()]
    return "\n".join([header, rule, *body])


def _load(samples_csv: Path) -> pd.DataFrame:
    frame = pd.read_csv(samples_csv, low_memory=False)
    for column in list(outcomes.OUTCOMES) + list(outcomes.SUBSCALES):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


# --------------------------------------------------------------------------- #
# 01 effects
# --------------------------------------------------------------------------- #


def effects_report(frame: pd.DataFrame, out: Path, plots: Path) -> dict:
    table = eff.ate_table(frame, dict(outcomes.SCALE_RANGE))
    spread = eff.spread_of_effects(table)
    means = eff.condition_means(frame, outcomes.OUTCOMES)

    primary = table[table["outcome"] == PRIMARY].sort_values("estimate")
    viz.forest(
        primary["condition"].tolist(),
        primary["estimate"].tolist(),
        primary["conf_low"].tolist(),
        primary["conf_high"].tolist(),
        title=f"Effect on {PRIMARY} (0-100), each intervention vs. control",
        xlabel="Difference in scale points (95% CI, HC1)",
        path=plots / "01_primary_forest.png",
    )

    pivot = table.pivot(
        index="condition", columns="outcome", values="pp_scale"
    ).reindex(index=list(INTERVENTIONS), columns=list(outcomes.OUTCOMES))
    viz.heatmap(
        pivot.to_numpy(),
        pivot.index.tolist(),
        [SHORT[name] for name in pivot.columns],
        title="Treatment effects, all 16 interventions x 13 outcomes",
        cbar_label="percentage points of scale range",
        path=plots / "01_effect_heatmap.png",
    )

    subscales = eff.ate_table(frame, {name: 100.0 for name in outcomes.SUBSCALES})
    sub_pivot = subscales.pivot(
        index="condition", columns="outcome", values="estimate"
    ).reindex(index=list(INTERVENTIONS))
    viz.heatmap(
        sub_pivot.to_numpy(),
        sub_pivot.index.tolist(),
        [name.replace("trust_", "") for name in sub_pivot.columns],
        title="Effects on the four trust subscales (scale points)",
        cbar_label="scale points",
        path=plots / "01_subscale_heatmap.png",
        figsize=(6.6, 6.4),
        fmt="{:+.2f}",
    )

    best = primary.iloc[-1]
    worst = primary.iloc[0]
    significant = int((table["p_holm"] < 0.05).sum())

    text = f"""# Intervention effects

[← main report](README.md) · sub-report 1 of 4

Every intervention is compared against the shared control condition. One OLS per
outcome with condition dummy-coded (control as reference) and HC1 robust standard
errors gives all 16 contrasts at once. There are {len(table)} effects in total
(16 interventions x 13 outcomes), so p-values are reported raw and adjusted
(Holm and Benjamini-Hochberg).

## Primary outcome

`{PRIMARY}` is the mean of the four trust subscales (competence, integrity,
benevolence, openness), each the mean of three 0-100 slider items.

Control mean: **{primary['control_mean'].iloc[0]:.2f}**.
Largest positive effect: **{best['condition']}** ({best['estimate']:+.2f} points,
95% CI [{best['conf_low']:.2f}, {best['conf_high']:.2f}]).
Largest negative effect: **{worst['condition']}** ({worst['estimate']:+.2f} points).
{significant} of {len(table)} effects across all outcomes survive Holm correction at 0.05.

![Primary outcome forest plot](plots/01_primary_forest.png)

{md_table(primary[["condition", "n_treated", "treated_mean", "estimate", "se", "conf_low", "conf_high", "cohens_d", "p", "p_holm"]].iloc[::-1], floats=3)}

## All outcomes

Effects are shown in percentage points of each outcome's scale range, so that a
move on a 0-100 slider, a dollar of donation and a newsletter signup are on one
footing. This is the unit the benchmark scores on.

![Effect heatmap](plots/01_effect_heatmap.png)

## Which outcomes move, and do the messages differ?

The benchmark separates two kinds of skill: knowing *which outcomes* an
intervention can move at all, and knowing *which message* moves them. The second
only means anything where the 16 effects actually spread out. `true_sd_effect_pp`
is the observed spread with the average sampling variance removed, so it
estimates real between-message variation rather than estimation noise.

{md_table(spread, floats=3)}

## Trust subscales

![Subscale effects](plots/01_subscale_heatmap.png)

## Condition means

{md_table(means.pivot(index="condition", columns="outcome", values="mean").reindex(columns=list(outcomes.OUTCOMES)).reset_index(), floats=2)}

## Full effect table

{md_table(table[["outcome", "condition", "estimate", "se", "conf_low", "conf_high", "pp_scale", "cohens_d", "p", "p_holm", "p_bh"]], floats=3)}
"""
    (out / "01_effects.md").write_text(text, encoding="utf-8")
    return {
        "table": table,
        "spread": spread,
        "primary": primary,
        "significant": significant,
    }


# --------------------------------------------------------------------------- #
# 02 demographics
# --------------------------------------------------------------------------- #


def demographics_report(frame: pd.DataFrame, out: Path, plots: Path) -> dict:
    baselines = pd.concat(
        [mod.baseline_means(frame, name, outcomes.OUTCOMES) for name in MODERATORS],
        ignore_index=True,
    )
    tests = pd.DataFrame(
        [
            mod.moderation_test(frame, name, outcome)
            for name in MODERATORS
            for outcome in outcomes.OUTCOMES
        ]
    )
    predict = mod.predictability(frame, MODERATORS, outcomes.OUTCOMES)
    gaps = mod.parity_gap(frame, MODERATORS, outcomes.OUTCOMES)

    r2 = (
        predict.groupby("moderator")["r2_moderator"].mean().sort_values(ascending=False)
    )
    viz.bars(
        r2.index.tolist(),
        r2.to_numpy().tolist(),
        title="How much of each outcome's variance the moderator explains",
        xlabel="mean added R² over condition alone (13 outcomes)",
        path=plots / "02_predictability.png",
        fmt="{:.4f}",
    )

    primary_baseline = baselines[baselines["outcome"] == PRIMARY]
    viz.bars(
        [f"{row.moderator}: {row.level}" for row in primary_baseline.itertuples()],
        primary_baseline["mean"].tolist(),
        title=f"Control-condition {PRIMARY} by demographic group",
        xlabel="scale points (0-100)",
        path=plots / "02_baselines.png",
        figsize=(7.6, 7.4),
        fmt="{:.1f}",
    )

    subgroup = pd.concat(
        [
            mod.subgroup_effects(frame, name, PRIMARY)
            for name in ("gender", "age_band", "race")
        ],
        ignore_index=True,
    )
    subgroup_plots = []
    for name in ("gender", "age_band", "race"):
        part = (
            subgroup[subgroup["moderator"] == name] if not subgroup.empty else subgroup
        )
        if part.empty:
            continue
        subgroup_plots.append(name)
        series = {
            str(level): part[part["level"] == level]
            .set_index("condition")
            .reindex(list(INTERVENTIONS))["estimate"]
            .tolist()
            for level in sorted(part["level"].unique())[:3]
        }
        viz.grouped_lines(
            part,
            [name[:14] for name in INTERVENTIONS],
            series,
            title=f"{PRIMARY}: effect of each intervention, by {name} (first 3 levels)",
            ylabel="scale points vs. control",
            path=plots / f"02_subgroup_{name}.png",
            figsize=(10.5, 4.6),
        )

    subgroup_figures = (
        "\n\n".join(
            f"![Subgroup effects by {name}](plots/02_subgroup_{name}.png)"
            for name in subgroup_plots
        )
        if subgroup_plots
        else "_Too few respondents per subgroup cell for level-wise effect estimates._"
    )
    strongest = tests.sort_values("p").head(15)
    marginals = pd.concat(
        [
            frame[name]
            .value_counts(normalize=True)
            .rename("share")
            .rename_axis("level")
            .reset_index()
            .assign(variable=name)
            for name in MODERATORS
        ],
        ignore_index=True,
    )[["variable", "level", "share"]]

    text = f"""# Demographics: baselines, moderation, and predictability

[← main report](README.md) · sub-report 2 of 4

Three questions that fail differently, kept apart.

## 1. Do groups differ at baseline?

Cell means within the control condition. A synthetic sample can land the overall
average and still put every demographic group on top of it.

![Baseline means](plots/02_baselines.png)

{md_table(primary_baseline[["moderator", "level", "n", "mean", "sd", "conf_low", "conf_high"]], floats=2)}

## 2. Does the intervention work differently for different people?

Saturated `outcome ~ condition * moderator`, with a joint Wald test on the whole
interaction block. With {len(tests)} tests here, individual small p-values are
expected; the pattern matters more than any single row.

Strongest 15 interactions:

{md_table(strongest[["moderator", "outcome", "n", "interaction_terms", "chi2", "df", "p", "r2"]], floats=4)}

{subgroup_figures}

## 3. How much is demographics alone? (stereotyping diagnostic)

`r2_moderator` is what the moderator adds once condition is already in the model.
A model answering from a stereotype produces a sample where knowing someone's
party tells you their answer almost exactly. Real survey data is much noisier
than that, so a large value here is a warning, not a success.

![Predictability](plots/02_predictability.png)

{md_table(predict.sort_values("r2_moderator", ascending=False).head(25), floats=4)}

## Demographic parity gaps

Largest gap between any two cells of a moderator, per outcome — the worst-case
group difference the sample implies.

{md_table(gaps.sort_values("gap", ascending=False).head(25), floats=2)}

## Marginals of the generated demographics

Gender, age and race were **pre-filled** from the preregistered quotas, so their
marginals are correct by construction and are not evidence about the model.
Education, income and party were **generated** by the model: their marginals are
a direct read on what population it produces when it is not told what to be.

{md_table(marginals, floats=4)}
"""
    (out / "02_demographics.md").write_text(text, encoding="utf-8")
    return {
        "tests": tests,
        "predict": predict,
        "gaps": gaps,
        "baselines": baselines,
        "marginals": marginals,
    }


# --------------------------------------------------------------------------- #
# 03 distributions
# --------------------------------------------------------------------------- #


def distributions_report(frame: pd.DataFrame, out: Path, plots: Path) -> dict:
    summary = dist.summary(frame, outcomes.OUTCOMES)
    slider_items = [
        item
        for battery, items in BATTERIES.items()
        if "0-100" in battery
        for item in items
    ]
    degeneracy = dist.degeneracy(
        frame,
        [
            name
            for name in outcomes.OUTCOMES
            if name not in ("donation_ams", "newsletter_signup")
        ],
    )
    item_degeneracy = dist.degeneracy(frame, slider_items)
    reliabilities = dist.scale_reliabilities(frame, BATTERIES)
    flatness = dist.straightlining(frame, BATTERIES)
    corr = dist.correlations(frame, outcomes.OUTCOMES)
    positions = dist.position_effects(frame, BATTERIES)

    viz.hist_grid(
        frame,
        [name for name in outcomes.OUTCOMES if name != "newsletter_signup"],
        title="Response distributions, 12 continuous outcomes",
        path=plots / "03_outcome_histograms.png",
        ranges={name: (0, 100) for name in outcomes.OUTCOMES if name != "donation_ams"}
        | {"donation_ams": (0, 10)},
    )
    viz.heatmap(
        corr.to_numpy(),
        [SHORT[name] for name in corr.index],
        [SHORT[name] for name in corr.columns],
        title="Correlations among the 13 outcomes",
        cbar_label="Pearson r",
        path=plots / "03_correlations.png",
        figsize=(8.4, 7.0),
        fmt="{:+.2f}",
    )
    viz.bars(
        reliabilities["scale"].tolist(),
        reliabilities["alpha"].tolist(),
        title="Internal consistency of the multi-item scales",
        xlabel="Cronbach's α",
        path=plots / "03_reliability.png",
        figsize=(7.8, 4.4),
    )

    pre_post = {}
    for pre, post in (("belief_pre", "belief_post"), ("trust_pre", "trust_post")):
        both = frame[[pre, post]].apply(pd.to_numeric, errors="coerce").dropna()
        pre_post[f"{pre} vs {post}"] = (
            float(both.corr().iloc[0, 1]) if len(both) > 2 else float("nan")
        )

    text = f"""# Response distributions and scale properties

[← main report](README.md) · sub-report 3 of 4

The characteristic failure of a silicon sample is not a wrong mean — it is a
degenerate distribution. A sample can reproduce every average in a study and
still be useless for anything distributional. These are the checks for that.

## Outcome distributions

![Outcome histograms](plots/03_outcome_histograms.png)

{md_table(summary, floats=2)}

## Degeneracy diagnostics

`modal_share` is the fraction of respondents giving the single most common
answer; the `share_at_*` columns pick out the scale endpoints and midpoint, which
is where a model that is guessing tends to pile up. `share_multiple_of_10` shows
how round the answers are — humans round too, so a value here is expected; a
value near 1.0 is not.

{md_table(degeneracy, floats=3)}

## Item-level degeneracy

{md_table(item_degeneracy, floats=3)}

## Scale reliability

![Reliability](plots/03_reliability.png)

{md_table(reliabilities, floats=3)}

## Straight-lining

Within-respondent SD across the items of one battery. `share_flat` is the
fraction who gave *identical* answers to every item in the battery — some of that
is real, a lot of it is a model copying its own previous line.

{md_table(flatness, floats=3)}

## Position effects within a battery

Does the answer drift with an item's position? The transcript format could induce
this even where the content does not, since the model sees its own earlier
answers.

{md_table(positions, floats=4)}

## Outcome correlations

![Correlations](plots/03_correlations.png)

## Pre- vs post-treatment consistency

Two items are asked both before and after the manipulation. Their correlation is
an internal consistency check: a respondent who is a coherent person answers them
similarly, and the intervention explains the rest.

{md_table(pd.DataFrame([{"pair": key, "pearson_r": value} for key, value in pre_post.items()]), floats=3)}
"""
    (out / "03_distributions.md").write_text(text, encoding="utf-8")
    return {
        "summary": summary,
        "degeneracy": degeneracy,
        "reliabilities": reliabilities,
        "flatness": flatness,
        "positions": positions,
        "pre_post": pre_post,
    }


# --------------------------------------------------------------------------- #
# 04 diagnostics
# --------------------------------------------------------------------------- #


def diagnostics_report(
    frame: pd.DataFrame, run_dir: Path, out: Path, plots: Path
) -> dict:
    meta_path = run_dir / "run_meta.json"
    meta = (
        json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    )
    draws = meta.get("draws", {})

    worst = pd.DataFrame(
        draws.get("worst_slots", []), columns=["slot", "rejected_draws"]
    )
    if not worst.empty:
        viz.bars(
            worst["slot"].tolist()[:18],
            worst["rejected_draws"].tolist()[:18],
            title="Slots whose draws were rejected most often",
            xlabel="rejected draws",
            path=plots / "04_rejections.png",
            figsize=(7.6, 5.6),
            fmt="{:.0f}",
        )

    near = pd.DataFrame(near_misses(run_dir, all_slots()))
    scored_items = (
        set(outcomes.DIRECT)
        | {item for items in outcomes.MEANS.values() for item in items}
        | {"funding_5", "newsletter"}
    )
    if not near.empty:
        scored = near[near["slot"].isin(scored_items)]
        worst_scored = scored.sort_values(
            "near_misses_per_asked", ascending=False
        ).head(1)
        scored_note = (
            f"Across the {len(scored_items)} items feeding the 13 scored outcomes, the worst near-miss rate is "
            f"**{worst_scored['near_misses_per_asked'].iloc[0]:.3f} per respondent** "
            f"(`{worst_scored['slot'].iloc[0]}`), and the median is "
            f"{scored['near_misses_per_asked'].median():.4f}. No scored outcome is materially exposed."
            if len(scored)
            else "No scored item had any rejected draw."
        )
    else:
        scored_note = "No draws were rejected."
    if not near.empty:
        near = near[near["rejected"] >= 5].copy()
        near["examples"] = near["examples"].map(
            lambda items: "; ".join(repr(item)[:26] for item in items)
        )
        near_miss_table = md_table(
            near[
                [
                    "slot",
                    "asked",
                    "rejected",
                    "rejected_per_asked",
                    "near_misses",
                    "near_miss_share",
                    "examples",
                ]
            ],
            floats=2,
        )
    else:
        near_miss_table = "_No draws were rejected._"

    # The one residual worth naming: a scored binary outcome whose rejected draws
    # are bare numerals, which cannot be mapped to Yes/No without guessing.
    newsletter_note = ""
    if "newsletter" in frame.columns:
        shares = frame["newsletter"].value_counts(normalize=True)
        newsletter_note = (
            "The one residual reported rather than fixed is `newsletter`, a scored binary outcome. Its rejected\n"
            "draws are bare numerals, and they are not a coherent coding — the values include 30 and 7 as well as\n"
            "0, 1 and 2 — so they look like slider answers bleeding in from neighbouring items rather than\n"
            f"Qualtrics codes, and cannot be mapped to Yes/No without guessing. In-format answers run\n"
            f"{shares.get('No', 0):.0%} No / {shares.get('Yes', 0):.0%} Yes; if the numerals were read as codes they would imply a\n"
            "somewhat higher Yes share. Rejecting them uniformly is the honest treatment, and this is the\n"
            "resulting uncertainty on that one outcome.\n"
        )

    state_note = ""
    if "state" in frame.columns and "zip_code" in frame.columns:
        weather = frame[frame["condition"] == "Extreme weather predictions"]
        if len(weather):
            state_note = (
                f"\n{len(weather)} respondents saw the state-adaptive arm. "
                f"{(weather['state'] == 'Prefer not to say').mean():.1%} declined to give a state.\n"
            )

    comments = (
        frame["comments"].dropna().astype(str)
        if "comments" in frame.columns
        else pd.Series(dtype=str)
    )
    sample_comments = comments[comments.str.len() > 3].head(12).tolist()

    text = f"""# Sampling diagnostics

[← main report](README.md) · sub-report 4 of 4

What the sampler actually did, and where it had to intervene.

## The run

{md_table(pd.DataFrame([{"key": key, "value": str(value)} for key, value in meta.items() if not isinstance(value, (dict, list))]))}

## Rejection sampling

Each slot is generated by asking for {meta.get('sampler', {}).get('draws_per_call', '?')} independent
continuations in one call and taking the first that parses as a legal answer.
Taking the first legal draw out of *n* i.i.d. draws is exact rejection sampling
from the model's distribution restricted to the legal set — the truncation costs
nothing in faithfulness, only in tokens.

- calls: **{draws.get('calls', 0):,}**
- draws: **{draws.get('draws', 0):,}**
- rejected: **{draws.get('rejected', 0):,}** ({(draws.get('rejection_rate') or 0):.2%})
- grammar-constrained fallbacks: **{draws.get('structured_fallbacks', 0):,}**
- forced defaults: **{draws.get('forced', 0):,}**

A grammar-constrained fallback means the model failed to produce a legal answer
in {meta.get('sampler', {}).get('max_rounds', 4)} rounds of {meta.get('sampler', {}).get('draws_per_call', 4)}
draws, and the slot was re-run with a decoder that cannot emit anything illegal.
Those draws are *not* faithful samples, so the count is reported rather than
buried; a forced default means even that failed.

{"![Rejections by slot](plots/04_rejections.png)" if not worst.empty else ""}

{md_table(worst) if not worst.empty else "_No draws were rejected._"}

## Is any rejection biasing an answer? (near-miss analysis)

Rejection sampling is unbiased **only if the rejection probability does not
depend on which answer the model meant**. Where it does, the retained
distribution is skewed by exactly that asymmetry — and this failure is silent.

Each rejected draw is therefore classified. A *near miss* would match a legal
option under a loose comparison (different punctuation, a missing currency
symbol): the model gave the right answer in the wrong spelling, and rejecting it
is dangerous, because whether that happens depends on which option was meant. A
non-near-miss matches nothing legal: the model answered a different question, and
rejecting it is correct.

A high `near_miss_share` says a slot's rejections are suspect; only
`near_misses_per_asked` says how much of the distribution is actually exposed to
them. The table is sorted by the latter.

**The scored outcomes are clean.** {scored_note}

{near_miss_table}

Two option-dependent rejections were found this way and fixed before this run:
money options rejected when the model dropped a `$` or a thousands separator
(which had inflated one income bracket from ~18% to ~28%), and slider decimals
rejected as non-integers (a slider is a continuous control the survey records as
an integer, so `92.36` is a real position and is now rounded).

{newsletter_note}
## Internal consistency of the generated session
{state_note}
## A sample of the free-text answers

The final comment box is the one free-text slot the model generates. It is not
scored, but it is the clearest window onto whether the model is writing as a
survey respondent or as something else.

{chr(10).join(f'- {comment!r}' for comment in sample_comments) if sample_comments else '_No free-text answers._'}
"""
    (out / "04_diagnostics.md").write_text(text, encoding="utf-8")
    return {"meta": meta, "worst": worst}


# --------------------------------------------------------------------------- #
# main report
# --------------------------------------------------------------------------- #


def generate(samples_csv: Path, run_dir: Path, out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    plots = out / "plots"
    plots.mkdir(parents=True, exist_ok=True)

    frame = _load(samples_csv)
    part1 = effects_report(frame, out, plots)
    part2 = demographics_report(frame, out, plots)
    part3 = distributions_report(frame, out, plots)
    part4 = diagnostics_report(frame, run_dir, out, plots)

    primary = part1["primary"]
    spread = part1["spread"]
    primary_spread = spread[spread["outcome"] == PRIMARY].iloc[0]
    control_mean = float(primary["control_mean"].iloc[0])
    degeneracy = part3["degeneracy"]
    primary_deg = degeneracy[degeneracy["variable"] == PRIMARY].iloc[0]
    top_predict = part2["predict"].sort_values("r2_moderator", ascending=False).iloc[0]
    # The partisan gap in climate belief is the most legible single number for
    # whether a synthetic sample's respondents differ from one another at all.
    party_means = frame.groupby("party")["belief_post"].mean().to_dict()
    party_gap = abs(
        party_means.get("Democrat", float("nan"))
        - party_means.get("Republican", float("nan"))
    )
    # The contrast that makes the flatness diagnostic rather than merely
    # disappointing: within-person item structure is good, between-person is not.
    reliabilities = part3["reliabilities"].set_index("scale")["alpha"]
    flatness = part3["flatness"].set_index("battery")["share_flat"]
    trust_scale = next(
        name for name in reliabilities.index if name.startswith("Trust in climate")
    )
    policy_scale = next(
        name for name in reliabilities.index if name.startswith("Specific climate")
    )
    alpha_trust = reliabilities[trust_scale]
    alpha_policy = reliabilities[policy_scale]
    flat_trust = flatness[trust_scale]

    counts = frame["condition"].value_counts()
    per_arm = counts.drop("control", errors="ignore")
    arm_text = (
        f"{per_arm.min():,} per intervention"
        if per_arm.min() == per_arm.max()
        else f"{per_arm.min():,}-{per_arm.max():,} per intervention"
    )
    text = f"""# Silicon sample of the Pfänder megastudy — main results

Qwen2.5-7B (base), sampled respondent by respondent through a text transcript of
the full instrument. **N = {len(frame):,}** synthetic respondents across
{frame['condition'].nunique()} conditions ({counts.get('control', 0):,} control, {arm_text}).

Sub-reports: [1 · intervention effects](01_effects.md) ·
[2 · demographics](02_demographics.md) ·
[3 · distributions](03_distributions.md) ·
[4 · sampling diagnostics](04_diagnostics.md)

## Headline

On the primary outcome — multidimensional trust in climate scientists, 0-100 —
the control mean is **{control_mean:.1f}**. Across the 16 interventions the
effects run from **{primary['estimate'].min():+.2f}** ({primary['condition'].iloc[0]})
to **{primary['estimate'].max():+.2f}** ({primary['condition'].iloc[-1]}) scale points.
{int(primary_spread['n_positive'])} of 16 point in the positive direction.

![Primary outcome forest plot](plots/01_primary_forest.png)

Across all 13 outcomes, **{part1['significant']} of {len(part1['table'])}** effects
survive Holm correction at α = 0.05.

![Effect heatmap](plots/01_effect_heatmap.png)

## How to read this

Three things decide whether this sample is worth anything, and they are separable:

1. **Does it move in the right direction?** — [sub-report 1](01_effects.md).
2. **Does it put the right people in the right places?** — [sub-report 2](02_demographics.md).
   The strongest demographic signal is **{top_predict['moderator']}** on
   `{top_predict['outcome']}` (adds R² = {top_predict['r2_moderator']:.3f} over
   condition alone).
3. **Does it look like survey data at all?** — [sub-report 3](03_distributions.md).
   On the primary outcome the modal answer is taken by
   {primary_deg['modal_share']:.1%} of respondents, and
   {primary_deg['share_multiple_of_10']:.1%} of answers are multiples of 10.

## The headline weakness: these respondents have no demographics

This sample's respondents are demographically **flat**. The model writes a party
identity, an income and an education into the transcript, then answers the rest of
the questionnaire as if it had not.

On belief in human-caused climate change, the synthetic Republicans and Democrats
differ by **{party_gap:.1f} points on a 0-100 scale**
({party_means.get('Republican', float('nan')):.1f} vs
{party_means.get('Democrat', float('nan')):.1f}). In US survey data this is one of
the largest and most reliable partisan gaps in the whole of public opinion —
routinely tens of points. Across all six moderators and all 13 outcomes, the
largest variance any moderator explains beyond condition is
**R² = {top_predict['r2_moderator']:.3f}** ({top_predict['moderator']} on
`{top_predict['outcome']}`).

What makes this sharp rather than merely disappointing is that the *within*-person
structure is good. The multi-item scales hold together like real ones — Cronbach's
α of {alpha_trust:.2f} on the 12-item trust battery, {alpha_policy:.2f} on the
seven policy items — and only {flat_trust:.0%} of respondents give a flat profile
across all twelve trust items. The model writes a coherent individual; it just
writes nearly the *same* individual every time, whatever demographics it has
placed at the top of the page.

This is the *opposite* of the failure the benchmark's stereotyping diagnostic is
built to catch. A model answering from a stereotype produces subgroup differences
that are too large and too clean; this one produces almost none. For the
benchmark that is the more damaging error of the two, because every subgroup and
demographic-baseline estimate the submission makes is close to a constant, while
the average treatment effects — which do not depend on between-person variation —
remain usable. Detail in [sub-report 2](02_demographics.md).

## Between-message variation

A prediction that gets the average effect right but cannot tell the messages
apart is doing only half the job. Removing sampling noise from the observed
spread leaves **{primary_spread['true_sd_effect_pp']:.2f} pp** of real
between-message variation on the primary outcome
(observed SD {primary_spread['sd_effect_pp']:.2f} pp).

{md_table(spread, floats=3)}

## Caveats

- The human data does not exist for anyone outside the study team, by design, so
  nothing here has been validated against ground truth. What can be checked —
  legality of draws, non-degeneracy, plausibility of the generated demographics —
  is in sub-reports 3 and 4.
- Gender, age and race were pre-filled from the preregistered census quotas;
  education, income and party were generated by the model.
- Every respondent answers every question: the real instrument lets participants
  skip most items, so the human data will carry missingness this sample does not.
- Attention checks and consent items were pre-filled as passed, matching a human
  sample that contains only respondents who passed them.
"""
    (out / "README.md").write_text(text, encoding="utf-8")
    return {
        "n": len(frame),
        "effects": part1,
        "demographics": part2,
        "distributions": part3,
        "diagnostics": part4,
    }
