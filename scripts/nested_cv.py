"""Nested leave-one-study-out cross-validation of the effect side of the recipe.

Every earlier evaluation in this project held out *parameters* while choosing
other things on all studies at once.  ``loso.py`` fits a shrinkage factor on two
studies and scores it on the third, which is honest as far as it goes, but the
ensemble membership, the within-outcome factor and the structural donor were all
picked by looking at the mean across all of them -- so the numbers that came out
were contaminated by selection even though each individual score was out-of-fold.

This closes that.  For each held-out study, **every free choice is fitted on the
others only**: which runs' effects to average, the within-outcome shrink factor,
the global shrink factor, which run supplies the level and the residual spread,
and the residual scale.  The assembled recipe is then scored once on the held-out
study.

**The selection rule is fixed before looking at any held-out score**, and it is
lexicographic rather than a weighted composite, because the leaderboard sorts on
pooled Pearson r:

1. membership and within-outcome factor maximise **training** pooled ``pearson_r``
2. the global factor is set so the **training** calibration slope is exactly 1
3. the donor and residual scale minimise **training** level and dispersion error

Four folds cannot support an interval on the *fold mean*, and none is reported
for it.  What can be reported, and now is, is the benchmark's own uncertainty
interval: a cluster bootstrap over interventions within each fold, which is the
preregistered interval every leaderboard row will carry.

**This grades effects only.**  ``scripts/nested_benchmark.py`` is the companion
that assembles a real :class:`~silicon_sampling.calibration.recipes.Recipe` per
fold and scores it on all four benchmark sections; the two agree on Section 1 to
within 0.002, which is what makes the fast grid here trustworthy.

Run: ``python scripts/nested_cv.py [--bootstrap 1000] [studies...]``
"""

from __future__ import annotations

import argparse
import itertools
import warnings

import numpy as np
import pandas as pd

from silicon_sampling import models as MODELS
from silicon_sampling.benchmark import distributions as DIST
from silicon_sampling.benchmark import metrics as MET
from silicon_sampling.benchmark.reference import ate_pairs, half_split
from silicon_sampling.calibration import folds as F

warnings.filterwarnings("ignore")

RUNS = ("qwen25_7b_v3", "qwen25_72b_v3", "v4_flash_v3", "muse_glimmer_30b")

#: Candidate effect ensembles: every non-empty subset of the runs a fold actually
#: has.  One run per model is all these studies have, so the shipped seven-run
#: average cannot be represented here; the gap is handled by the measured
#: reliability difference reported at the end, not by pretending.
#:
#: The subsetting is per fold and is *reported*, because it is currently doing
#: real work: DeepSeek-V4-Flash has no CCC sample, CCC is in the training set of
#: every fold, and so no membership containing V4 is a candidate anywhere.  That
#: is the deliberate pre-V4 state, not an oversight — but a silent version of it
#: would read as "the search considered V4 and rejected it", which is false.
WITHIN_GRID = tuple(np.round(np.arange(0.0, 1.01, 0.1), 2))


def memberships(available: tuple[str, ...]) -> tuple[tuple[str, ...], ...]:
    """Every non-empty subset of *available*, shortest first, in ``RUNS`` order."""
    out = []
    for size in range(1, len(available) + 1):
        out.extend(itertools.combinations(available, size))
    return tuple(out)


#: Sample frames and their effect tables, read once and re-used across splits.
_CACHE: dict[str, dict] = {}


def _samples(study: F.FoldStudy) -> dict:
    """This study's silicon samples and their effect tables, cached.

    Reading four studies' samples is most of the runtime, and none of it depends
    on which half of the humans is the reference -- so a multi-split run pays it
    once rather than once per split.
    """
    if study.name in _CACHE:
        return _CACHE[study.name]
    scored = set(study.design.outcomes)
    frames, effects, controls = {}, {}, {}
    for run in RUNS:
        key = MODELS.resolve_run(study.samples_dir, run)
        if key is None:
            continue
        sample = study.prepare(
            pd.read_csv(study.samples_dir(key) / "samples.csv", low_memory=False)
        )
        frames[run] = sample
        table = study.effects(sample)
        effects[run] = table[table["outcome"].isin(scored)]
        controls[run] = sample[sample["condition"].astype(str) == study.design.control]
    humans = study.prepare(study.load_humans())
    _CACHE[study.name] = {"effects": effects, "controls": controls, "humans": humans}
    return _CACHE[study.name]


def load(study: F.FoldStudy, seed: int = 42) -> dict:
    """Human halves, per-run effect pairs, and per-run control arms.

    Every frame -- human and synthetic -- goes through the study's ``prepare``
    first, so both sides carry the same condition vocabulary and, on CCC, the
    same donation budget.

    ``seed`` picks the half-split.  The benchmark fixes one preregistered split
    and scores against it, so a single seed is the right model of what Pfänder
    will do -- but it is the wrong tool for *choosing between recipes*, because
    the reference half's sampling noise is shared by every variant and one draw
    of it can reorder them.  ``main(--splits N)`` averages over N seeds for that
    reason, and reports both.
    """
    cached = _samples(study)
    human1, human2 = half_split(cached["humans"], seed=seed)
    reference = study.effects(human1)
    scored = set(study.design.outcomes)
    reference = reference[reference["outcome"].isin(scored)]
    pairs, controls = {}, dict(cached["controls"])
    for run, frame in cached["effects"].items():
        pairs[run] = (
            ate_pairs(reference, frame)
            .dropna(subset=["estimate_h", "estimate_l"])
            .set_index(["outcome", "condition"])
        )
    index = None
    for frame in pairs.values():
        index = frame.index if index is None else index.intersection(frame.index)
    replication_all = study.effects(human2)
    replication = ate_pairs(
        reference, replication_all[replication_all["outcome"].isin(scored)]
    ).dropna(subset=["estimate_h", "estimate_l"])
    # The ceiling has to be measured on the pairs the models are measured on, or
    # it is a different question answered on a different grid.
    replication = (
        replication.set_index(["outcome", "condition"]).loc[index].reset_index()
    )
    return {
        "study": study,
        "cfg": {"name": study.name, "scales": dict(study.design.outcomes)},
        "pairs": pairs,
        "index": index,
        "controls": controls,
        "human_control": human1[
            human1[study.design.condition_col].astype(str) == study.design.control
        ],
        "replication": replication,
    }


def effect_vector(
    data: dict, membership: tuple[str, ...], within: float
) -> pd.DataFrame:
    """Averaged, within-outcome-shrunk pairs frame for one study.

    The standard error travels with the estimate.  Averaging ``m`` runs of the
    same target divides the sampling variance by ``m``, and the within-outcome
    contraction multiplies the deviation -- and hence its error -- by ``within``.
    Carrying the first member's raw ``se_l`` instead, as an earlier version did,
    left ``beta_adj`` reading a reliability the vector being scored does not have.
    """
    index = data["index"]
    base = data["pairs"][membership[0]].loc[index].copy()
    stack = np.mean(
        [data["pairs"][r].loc[index, "estimate_l"].to_numpy() for r in membership],
        axis=0,
    )
    variance = np.mean(
        [data["pairs"][r].loc[index, "se_l"].to_numpy() ** 2 for r in membership],
        axis=0,
    ) / len(membership)
    out = base.reset_index()
    out["estimate_l"] = stack
    out["se_l"] = np.sqrt(variance)
    if within != 1.0:
        grouped = out.groupby("outcome")["estimate_l"]
        mean = grouped.transform("mean")
        out["estimate_l"] = mean + within * (out["estimate_l"] - mean)
        # The per-outcome mean is itself an average of the cells, so contracting
        # toward it leaves a little more error than a bare scaling; treating the
        # mean as fixed is the standard approximation and is used here.
        out["se_l"] = out["se_l"] * within
    return out


def section1(pairs: pd.DataFrame) -> dict:
    out = dict(MET.pooled_metrics(pairs, include_rmse=True))
    out.update(MET.run_calibration_pooled(pairs))
    out.update(MET.adjusted_metrics(pairs))
    return out


def interval(pairs: pd.DataFrame, draws: int) -> dict:
    """The benchmark's own uncertainty interval: a cluster bootstrap over arms."""
    if not draws:
        return {}
    got = MET.cluster_bootstrap(pairs, section1, cluster="condition", draws=draws)
    return {
        key: value
        for key, value in got.items()
        if key.startswith(("pearson_r", "spearman_rho", "directional_pct"))
    }


def control_metrics(data: dict, donor: str, scale: float) -> dict:
    """Level error and the four shape metrics, on the donor's control arm.

    Residuals are rescaled about the arm mean and then clipped to the scale, and
    the mean is restored afterwards -- the shipped recipe recomposes and clips the
    same way, and a version that clipped without restoring the mean quietly
    reported a level error the submission would not have.
    """
    cfg, human = data["cfg"], data["human_control"]
    synth = data["controls"][donor]
    level, shapes = [], []
    for outcome, span in cfg["scales"].items():
        if outcome not in human.columns or outcome not in synth.columns:
            continue
        left = pd.to_numeric(human[outcome], errors="coerce").dropna()
        right = pd.to_numeric(synth[outcome], errors="coerce").dropna()
        if len(left) < 30 or len(right) < 30:
            continue
        centre = right.mean()
        right = (centre + scale * (right - centre)).clip(0.0, span)
        right = (right + (centre - right.mean())).clip(0.0, span)
        level.append(abs(right.mean() - left.mean()) / span * 100)
        shapes.append(DIST.compare_distributions(left, right, lo=0.0, hi=span))
    if not shapes:
        return {}
    frame = pd.DataFrame(shapes)
    return {
        "level_err_pp": float(np.mean(level)),
        "variance_ratio": float(frame["variance_ratio"].mean()),
        "ovl": float(frame["ovl"].mean()),
        "ks": float(frame["ks"].mean()),
        "w1": float(frame["w1"].mean()),
    }


def training_r(train: list[dict], membership: tuple[str, ...], within: float) -> float:
    pooled = pd.concat(
        [effect_vector(d, membership, within) for d in train], ignore_index=True
    )
    return float(np.corrcoef(pooled["estimate_h"], pooled["estimate_l"])[0, 1])


def training_kappa(
    train: list[dict], membership: tuple[str, ...], within: float
) -> float:
    pooled = pd.concat(
        [effect_vector(d, membership, within) for d in train], ignore_index=True
    )
    return float(
        np.cov(pooled["estimate_h"], pooled["estimate_l"], ddof=1)[0, 1]
        / np.var(pooled["estimate_l"], ddof=1)
    )


def best_within(train: list[dict], membership: tuple[str, ...]) -> float:
    """The within factor that maximises training r for *this* membership."""
    return float(max(WITHIN_GRID, key=lambda w: training_r(train, membership, w)))


def fit_on_training(train: list[dict], available: tuple[str, ...]) -> dict:
    """Every free choice, decided on the training studies alone."""
    best = None
    for membership in memberships(available):
        for within in WITHIN_GRID:
            r = training_r(train, membership, within)
            if best is None or r > best[0]:
                best = (r, membership, within)
    _, membership, within = best
    kappa = training_kappa(train, membership, within)

    donor_best = None
    for donor in available:
        if any(donor not in d["controls"] for d in train):
            continue
        ratios = []
        for d in train:
            for outcome, span in d["cfg"]["scales"].items():
                if (
                    outcome not in d["human_control"].columns
                    or outcome not in d["controls"][donor].columns
                ):
                    continue
                left = pd.to_numeric(
                    d["human_control"][outcome], errors="coerce"
                ).dropna()
                right = pd.to_numeric(
                    d["controls"][donor][outcome], errors="coerce"
                ).dropna()
                if len(left) < 30 or len(right) < 30 or left.std() == 0:
                    continue
                ratios.append(right.std() / left.std())
        scale = float(1.0 / np.mean(ratios)) if ratios else 1.0
        got = [control_metrics(d, donor, scale) for d in train]
        got = [g for g in got if g]
        if not got:
            continue
        loss = float(np.mean([g["level_err_pp"] for g in got])) / 10.0 + float(
            np.mean([abs(g["variance_ratio"] - 1) for g in got])
        )
        if donor_best is None or loss < donor_best[0]:
            donor_best = (loss, donor, scale)
    _, donor, scale = donor_best
    return {
        "membership": membership,
        "within": float(within),
        "kappa": kappa,
        "donor": donor,
        "residual_scale": scale,
    }


#: Recipes whose structure is fixed a priori, so only scale parameters are fitted.
#:
#: This distinction matters more than it looks.  The shipped recipe does not
#: *choose* its membership or its within-outcome factor from data -- it averages
#: every Qwen run there is, and takes 0.5 as a default.  Letting the training
#: folds pick those two instead adds fitting noise the real recipe never pays, so
#: scoring only the fully-fitted variant understates it.
#:
#: "Pre-committed" is doing a little less work than it sounds like.
#: ``WITHIN_SHRINK`` 0.5 was adopted after leave-one-study-out picked it on 3 of 3
#: folds, so it is a *prior about the form* -- shrink toward the outcome profile --
#: with a magnitude that was fitted once, globally, on these same studies.  The
#: selection optimism that buys is small and measured: over {0.3, 0.5, 1.0} the
#: chosen value scores 0.005 above the mean of the set.  The 2x2 below charges the
#: magnitude in full anyway, which is the conservative reading.
FIXED = {
    "Qwen pair, within 0.5 (shipped design)": (("qwen25_7b_v3", "qwen25_72b_v3"), 0.5),
    "Qwen pair, no within shrink": (("qwen25_7b_v3", "qwen25_72b_v3"), 1.0),
    "Qwen pair, within 0.3": (("qwen25_7b_v3", "qwen25_72b_v3"), 0.3),
    "7B alone, within 0.5": (("qwen25_7b_v3",), 0.5),
    "Qwen pair + Muse, within 0.5": (
        ("qwen25_7b_v3", "qwen25_72b_v3", "muse_glimmer_30b"),
        0.5,
    ),
    "Qwen pair + Muse, no within shrink": (
        ("qwen25_7b_v3", "qwen25_72b_v3", "muse_glimmer_30b"),
        1.0,
    ),
    "Muse alone, within 0.5": (("muse_glimmer_30b",), 0.5),
}


#: Membership *rules* rather than membership choices.
#:
#: The distinction is the one the previous round got wrong.  Reading four fold
#: means and then declaring "average the Qwens and Muse" is a selection made on
#: the held-out data, and it will not transfer to Pfaender.  A rule -- "average
#: every run that clears this bar on the training studies" -- is decided once, and
#: each fold instantiates it without ever seeing its own answer.  So the rule can
#: be scored honestly even though the memberships it produces differ per fold.
def rule_all(train: list[dict], available: tuple[str, ...]) -> tuple[str, ...]:
    """Average everything there is.  No selection at all, so nothing to leak."""
    return available


def rule_positive(train: list[dict], available: tuple[str, ...]) -> tuple[str, ...]:
    """Average every run whose own training correlation is positive.

    The bar exists because a run can be anti-correlated with the truth -- V4-Flash
    is, on two studies -- and averaging that in is worse than dropping it.  Falls
    back to the single best run if nothing clears zero, so the rule always returns
    something.
    """
    scored = [(training_r(train, (run,), 1.0), run) for run in available]
    keep = tuple(run for r, run in scored if r > 0)
    return keep if keep else (max(scored)[1],)


RULES = {
    "rule: average all available, within 0.5": (rule_all, 0.5),
    "rule: positive training r, within 0.5": (rule_positive, 0.5),
    "rule: positive training r, no within shrink": (rule_positive, 1.0),
}


def reliability(estimate, se) -> float:
    """Share of an effect vector's spread that is signal rather than draw noise."""
    estimate = np.asarray(estimate, dtype=float)
    se = np.asarray(se, dtype=float)
    spread = np.nanvar(estimate, ddof=1)
    if not spread > 0:
        return float("nan")
    return float(max(0.0, 1.0 - np.nanmean(se**2) / spread))


def evaluate(
    data: dict, names: list[str], available: tuple[str, ...], bootstrap: int
) -> tuple[list[dict], list[dict]]:
    """Score every variant on every fold, for one half-split of the humans."""
    rows, chosen = [], []
    for held in names:
        train = [data[n] for n in names if n != held]
        fit = fit_on_training(train, available)
        chosen.append(
            {
                "held out": held,
                **{
                    k: (",".join(v).replace("_v3", "") if isinstance(v, tuple) else v)
                    for k, v in fit.items()
                },
            }
        )

        test = effect_vector(data[held], fit["membership"], fit["within"])
        test["estimate_l"] = fit["kappa"] * test["estimate_l"]
        test["se_l"] = fit["kappa"] * test["se_l"]
        row = {"held out": held, **section1(test), **interval(test, bootstrap)}
        row.update(control_metrics(data[held], fit["donor"], fit["residual_scale"]))
        row["what"] = "recipe, fitted on the other three"
        rows.append(row)

        rep = data[held]["replication"]
        rrow = {
            "held out": held,
            **section1(rep),
            **interval(rep, bootstrap),
            "what": "human replication",
        }
        rows.append(rrow)

        for run in available:
            single = effect_vector(data[held], (run,), 1.0)
            rows.append(
                {
                    "held out": held,
                    **section1(single),
                    "what": f"single: {run.replace('_v3', '')}, uncalibrated",
                }
            )

        # The 2x2 the Pfander prediction is built on: which of the two
        # data-informed choices -- membership and the within-outcome factor -- is
        # load-bearing, and what the recipe scores when each is charged as fitted
        # rather than granted as a prior.  The "both fitted" corner is the
        # ``recipe`` row above and the "both fixed" corner is the shipped design.
        #
        # The within factor of the mixed corner is refitted **for the prior
        # membership**, not carried over from the joint search: the joint argmax
        # belongs to a different ensemble, and reusing it charges the prior
        # membership for a factor nothing chose for it.
        prior = ("qwen25_7b_v3", "qwen25_72b_v3")
        if all(r in available for r in prior):
            for label, membership, within in (
                (
                    "2x2: membership by prior, within fitted",
                    prior,
                    best_within(train, prior),
                ),
                ("2x2: membership fitted, within fixed 0.5", fit["membership"], 0.5),
            ):
                k = training_kappa(train, membership, within)
                te = effect_vector(data[held], membership, within)
                te["estimate_l"] = k * te["estimate_l"]
                te["se_l"] = k * te["se_l"]
                rows.append(
                    {
                        "held out": held,
                        **section1(te),
                        **interval(te, bootstrap),
                        "what": label,
                    }
                )

        # Membership decided by a pre-committed rule, instantiated on training.
        for label, (rule, within) in RULES.items():
            membership = rule(train, available)
            k = training_kappa(train, membership, within)
            te = effect_vector(data[held], membership, within)
            te["estimate_l"] = k * te["estimate_l"]
            te["se_l"] = k * te["se_l"]
            row = {
                "held out": held,
                **section1(te),
                **interval(te, bootstrap),
                "what": label,
            }
            row.update(control_metrics(data[held], fit["donor"], fit["residual_scale"]))
            rows.append(row)
            chosen.append(
                {
                    "held out": held,
                    "membership": ",".join(membership).replace("_v3", ""),
                    "within": within,
                    "kappa": k,
                    "donor": "",
                    "residual_scale": float("nan"),
                    "rule": label,
                }
            )

        # Structure fixed a priori; only the global factor is fitted on training.
        for label, (membership, within) in FIXED.items():
            if any(r not in available for r in membership):
                continue
            k = training_kappa(train, membership, within)
            te = effect_vector(data[held], membership, within)
            te["estimate_l"] = k * te["estimate_l"]
            te["se_l"] = k * te["se_l"]
            rows.append(
                {
                    "held out": held,
                    **section1(te),
                    **interval(te, bootstrap),
                    "what": label,
                }
            )

    return rows, chosen


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bootstrap", type=int, default=0)
    parser.add_argument(
        "--splits",
        type=int,
        default=1,
        help=(
            "average over this many half-splits of the humans. 1 (the default) "
            "reproduces the single preregistered-style split the benchmark uses; "
            "more is the right tool for comparing variants, because the reference "
            "half's noise is shared by all of them and one draw can reorder them."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("studies", nargs="*", default=None)
    args = parser.parse_args(argv)

    pd.set_option("display.width", 250)
    studies = F.load_folds(args.studies or None)
    seeds = [args.seed + offset for offset in range(max(1, args.splits))]
    data = {study.name: load(study, seed=seeds[0]) for study in studies}
    names = list(data)

    # A run is a candidate only where every study has it.  Report the exclusions:
    # a silently narrowed search reads as a search that considered and rejected.
    available = tuple(r for r in RUNS if all(r in data[n]["pairs"] for n in names))
    print("=== run availability ===\n")
    for run in RUNS:
        have = [n for n in names if run in data[n]["pairs"]]
        missing = [n for n in names if run not in data[n]["pairs"]]
        mark = "candidate" if run in available else "EXCLUDED"
        note = f"  (missing: {', '.join(missing)})" if missing else ""
        print(f"  {run:20s} {mark:10s} present in {len(have)}/{len(names)}{note}")
    print(
        f"\n  {len(memberships(available))} candidate ensembles "
        f"x {len(WITHIN_GRID)} within-shrink values searched per fold"
    )
    print("\n  scored grid per study (pairs, after dropping redundant outcomes):")
    for name in names:
        study = data[name]["study"]
        dropped = f"  dropped: {', '.join(study.dropped)}" if study.dropped else ""
        print(
            f"    {name:9s} {len(data[name]['index']):4d} pairs "
            f"= {len(study.design.outcomes)} outcomes "
            f"x {len(study.design.conditions) - 1} arms{dropped}"
        )
    print()

    rows, chosen = evaluate(data, names, available, args.bootstrap)
    across = None
    if len(seeds) > 1:
        # The variant comparison, averaged over splits.  Every variant is scored
        # against the *same* reference half within a split, so the reference's
        # sampling noise is common to all of them -- which is exactly why a single
        # split can reorder variants that differ by less than that noise, and why
        # averaging over splits estimates the ordering far better than it
        # estimates any one score.
        collected = []
        for seed in seeds:
            per_seed = {study.name: load(study, seed=seed) for study in studies}
            got, _ = evaluate(per_seed, names, available, 0)
            frame = pd.DataFrame(got)
            frame["seed"] = seed
            collected.append(frame)
        across = pd.concat(collected, ignore_index=True)
    print(
        "=== what the training folds chose, never having seen the held-out study ===\n"
    )
    print(
        pd.DataFrame(chosen).to_string(index=False, float_format=lambda v: f"{v:.3f}")
    )

    table = pd.DataFrame(rows)
    s1 = [
        "pearson_r",
        "spearman_rho",
        "directional_pct",
        "pearson_within",
        "pearson_adj",
        "rmse",
        "rmse_adj",
        "alpha",
        "beta",
    ]
    s3 = ["level_err_pp", "variance_ratio", "ovl", "ks", "w1"]
    print("\n\n=== held-out effect recovery (Section 1) ===\n")
    print(
        table.pivot_table(
            index="what", columns="held out", values="pearson_r"
        ).to_string(float_format=lambda v: f"{v:6.3f}")
    )
    if args.bootstrap and "pearson_r_lo" in table.columns:
        print(
            f"\n95% cluster-bootstrap intervals over interventions "
            f"({args.bootstrap} draws), per fold:\n"
        )
        band = table.dropna(subset=["pearson_r_lo"])
        band = band.assign(
            interval=band.apply(
                lambda r: f"{r['pearson_r']:6.3f} [{r['pearson_r_lo']:6.3f},"
                f" {r['pearson_r_hi']:6.3f}]",
                axis=1,
            )
        )
        print(
            band.pivot_table(
                index="what", columns="held out", values="interval", aggfunc="first"
            ).to_string()
        )
    print("\nfull metric set, recipe rows only:\n")
    keep = table[table["what"] == "recipe, fitted on the other three"]
    print(
        keep[
            ["held out"] + [c for c in s1 if c in keep] + [c for c in s3 if c in keep]
        ].to_string(index=False, float_format=lambda v: f"{v:7.3f}")
    )
    print(
        "\nshipped design, per fold "
        "(structure pre-committed, only the global factor fitted):\n"
    )
    sd = table[table["what"] == "Qwen pair, within 0.5 (shipped design)"]
    print(
        sd[["held out"] + [c for c in s1 if c in sd]].to_string(
            index=False, float_format=lambda v: f"{v:7.3f}"
        )
    )
    print("\nhuman replication on the same folds:\n")
    hum = table[table["what"] == "human replication"]
    print(
        hum[["held out"] + [c for c in s1 if c in hum]].to_string(
            index=False, float_format=lambda v: f"{v:7.3f}"
        )
    )
    print("\n\n=== fold means (four folds: still too few for an interval) ===\n")
    order = [
        "recipe, fitted on the other three",
        "2x2: membership by prior, within fitted",
        "2x2: membership fitted, within fixed 0.5",
        *RULES,
        *FIXED,
        *[f"single: {r.replace('_v3', '')}, uncalibrated" for r in RUNS],
        "human replication",
    ]
    for what in order:
        sub = table[table["what"] == what]
        if sub.empty:
            continue
        cells = " ".join(
            f"{c}={sub[c].mean():+.3f}" for c in s1 if c in sub and sub[c].notna().any()
        )
        print(f"  {what:42s} {cells}")

    if across is not None:
        print(f"\n\n=== averaged over {len(seeds)} half-splits of the humans ===\n")
        print(
            "  A single split is what Pfander will do, so the block above is the\n"
            "  right model of one score.  It is the wrong tool for ranking variants:\n"
            "  they share a reference half, so one draw of its noise moves them\n"
            "  together and can reorder any pair separated by less than it.\n"
        )
        summary = (
            across.groupby(["what", "seed"])["pearson_r"]
            .mean()
            .groupby("what")
            .agg(["mean", "std", "min", "max"])
        )
        summary["se"] = summary["std"] / np.sqrt(len(seeds))
        order = [
            w
            for w in [
                "recipe, fitted on the other three",
                "2x2: membership by prior, within fitted",
                "2x2: membership fitted, within fixed 0.5",
                *RULES,
                *FIXED,
                *[f"single: {r.replace('_v3', '')}, uncalibrated" for r in RUNS],
                "human replication",
            ]
            if w in summary.index
        ]
        print(
            summary.loc[order, ["mean", "se", "std", "min", "max"]].to_string(
                float_format=lambda v: f"{v:7.3f}"
            )
        )
        single = across[across["seed"] == seeds[0]].groupby("what")["pearson_r"].mean()
        beats = []
        base = "single: qwen25_7b, uncalibrated"
        if base in summary.index:
            for what in order:
                if what in (base, "human replication"):
                    continue
                delta = (
                    across[across["what"] == what].groupby("seed")["pearson_r"].mean()
                    - across[across["what"] == base].groupby("seed")["pearson_r"].mean()
                )
                beats.append(
                    {
                        "variant": what,
                        "mean delta vs raw 7B": delta.mean(),
                        "splits where it wins": f"{int((delta > 0).sum())}/{len(seeds)}",
                        "delta at seed 42": single.get(what, float("nan"))
                        - single.get(base, float("nan")),
                    }
                )
            print("\n  against an uncalibrated Qwen2.5-7B:\n")
            print(
                pd.DataFrame(beats).to_string(
                    index=False, float_format=lambda v: f"{v:+7.3f}"
                )
            )

    # How much of each side's spread is signal, measured from the standard errors
    # rather than assumed.  This is what licenses -- or refuses -- the step from a
    # fold mean to a Pfander prediction: correlations scale as the square root of
    # the predictor's reliability, so a fold whose model vector is noisier than
    # the submission's understates what the submission will score.
    print("\n\n=== measured reliability of each side's effect vector ===\n")
    rel_rows = []
    for name in names:
        d = data[name]
        rep = d["replication"]
        rel_rows.append(
            {
                "study": name,
                "side": "human reference half",
                "reliability": reliability(rep["estimate_h"], rep["se_h"]),
            }
        )
        rel_rows.append(
            {
                "study": name,
                "side": "human replication half",
                "reliability": reliability(rep["estimate_l"], rep["se_l"]),
            }
        )
        for run in available:
            single = effect_vector(d, (run,), 1.0)
            rel_rows.append(
                {
                    "study": name,
                    "side": run.replace("_v3", ""),
                    "reliability": reliability(single["estimate_l"], single["se_l"]),
                }
            )
        prior = tuple(r for r in ("qwen25_7b_v3", "qwen25_72b_v3") if r in available)
        if len(prior) == 2:
            pair = effect_vector(d, prior, 1.0)
            rel_rows.append(
                {
                    "study": name,
                    "side": "AVG qwen pair",
                    "reliability": reliability(pair["estimate_l"], pair["se_l"]),
                }
            )
    rel = pd.DataFrame(rel_rows)
    print(
        rel.pivot_table(index="side", columns="study", values="reliability").to_string(
            float_format=lambda v: f"{v:6.3f}"
        )
    )
    pair_mean = rel[rel["side"] == "AVG qwen pair"]["reliability"].mean()
    if pair_mean == pair_mean:
        print(
            f"\n  one-run-each Qwen pair, mean over folds: {pair_mean:.3f}\n"
            "  (compare against the same quantity measured on Pfander's own runs;\n"
            "   the ratio of the two square roots is the only defensible bridge\n"
            "   from a fold mean to a Pfander prediction.)"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
