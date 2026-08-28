"""Nested leave-one-study-out cross-validation of the whole recipe.

Every earlier evaluation in this project held out *parameters* while choosing
other things on all three studies at once.  ``loso.py`` fits a shrinkage factor on
two studies and scores it on the third, which is honest as far as it goes, but the
ensemble membership, the within-outcome factor and the structural donor were all
picked by looking at the mean across all three -- so the numbers that came out
were contaminated by selection even though each individual score was out-of-fold.

This closes that.  For each held-out study, **every free choice is fitted on the
other two only**:

* which runs' effects to average
* the within-outcome shrink factor
* the global shrink factor
* which run supplies the level, the residual spread and the demographic offsets
* the residual scale

and the assembled recipe is then scored once on the held-out study, across the
metric families the benchmark reports rather than on the sort key alone.

**The selection rule is fixed before looking at any held-out score**, and it is
lexicographic rather than a weighted composite, because the leaderboard sorts on
pooled Pearson r:

1. membership and within-outcome factor maximise **training** pooled ``pearson_r``
2. the global factor is set so the **training** calibration slope is exactly 1
3. the donor and residual scale minimise **training** level and dispersion error

Three folds cannot support an interval and none is reported.  What they can show
is whether the recipe beats its own components on data it never saw.

Run: ``python scripts/nested_cv.py``
"""

from __future__ import annotations

import itertools
import warnings

import numpy as np
import pandas as pd

from silicon_sampling import models as MODELS
from silicon_sampling.benchmark import distributions as DIST
from silicon_sampling.benchmark import metrics as MET
from silicon_sampling.benchmark.reference import ate_pairs, half_split

warnings.filterwarnings("ignore")

RUNS = ("qwen25_7b_v3", "qwen25_72b_v3", "v4_flash_v3", "muse_glimmer_30b")

#: Candidate effect ensembles: every non-empty subset of the runs a fold actually
#: has.  One run per model is all these studies have, so the shipped eight-run
#: average cannot be represented here; the gap is handled in the report by the
#: measured reliability difference, not by pretending.
#:
#: The subsetting is per fold and is *reported*, because it is currently doing
#: real work: DeepSeek-V4-Flash has no CCC sample yet, CCC is in the training set
#: of every fold, and so no membership containing V4 is a candidate anywhere.
#: That is the deliberate pre-V4 state, not an oversight — but a silent version of
#: it would read as "the search considered V4 and rejected it", which is false.
WITHIN_GRID = tuple(np.round(np.arange(0.0, 1.01, 0.1), 2))


def memberships(available: tuple[str, ...]) -> tuple[tuple[str, ...], ...]:
    """Every non-empty subset of *available*, shortest first, in ``RUNS`` order."""
    out = []
    for size in range(1, len(available) + 1):
        out.extend(itertools.combinations(available, size))
    return tuple(out)


def study_config() -> list[dict]:
    from silicon_sampling.ccc import outcomes as co
    from silicon_sampling.goldwert import outcomes as go
    from silicon_sampling.voelkel import outcomes as vo

    return [
        {
            "name": "Voelkel",
            "score": "silicon_sampling.voelkel.score",
            "paths": "silicon_sampling.voelkel.paths",
            "scales": {k: 100.0 for k in vo.OUTCOMES},
            "moderators": ("party_gen", "gender", "race", "education", "age_band"),
        },
        {
            "name": "ICPC",
            "score": "silicon_sampling.icpc.score",
            "paths": "silicon_sampling.icpc.paths",
            "scales": {"belief": 100.0, "policy": 100.0, "sharing": 1.0, "wept": 8.0},
            "moderators": ("gender", "education", "age_band", "ideology_band"),
        },
        {
            "name": "Goldwert",
            "score": "silicon_sampling.goldwert.score",
            "paths": "silicon_sampling.goldwert.paths",
            "scales": dict(go.SCORED),
            "moderators": ("party", "gender", "education", "age_band"),
        },
        {
            "name": "CCC",
            "score": "silicon_sampling.ccc.score",
            "paths": "silicon_sampling.ccc.paths",
            "scales": dict(co.SCORED),
            "moderators": ("party", "gender", "race", "education", "age_band"),
        },
    ]


def load(cfg: dict) -> dict:
    """Human halves, per-run effect pairs, and per-run control arms."""
    score = __import__(cfg["score"], fromlist=["x"])
    paths = __import__(cfg["paths"], fromlist=["x"])
    human1, human2 = half_split(score.load_humans())
    reference = score.effects(human1)
    pairs, controls = {}, {}
    for run in RUNS:
        key = MODELS.resolve_run(paths.samples_dir, run)
        if key is None:
            continue
        sample = pd.read_csv(paths.samples_dir(key) / "samples.csv", low_memory=False)
        pairs[run] = (
            ate_pairs(reference, score.effects(sample))
            .dropna(subset=["estimate_h", "estimate_l"])
            .set_index(["outcome", "condition"])
        )
        mask = (
            sample["condition"].astype(str).str.contains("ontrol", case=False, na=False)
        )
        controls[run] = sample[mask]
    index = None
    for frame in pairs.values():
        index = frame.index if index is None else index.intersection(frame.index)
    column = next(c for c in ("condName", "condition", "cond") if c in human1.columns)
    hmask = human1[column].astype(str).str.contains("ontrol", case=False, na=False)
    replication = ate_pairs(reference, score.effects(human2)).dropna(
        subset=["estimate_h", "estimate_l"]
    )
    return {
        "cfg": cfg,
        "pairs": pairs,
        "index": index,
        "controls": controls,
        "human_control": human1[hmask],
        "replication": replication,
    }


def effect_vector(
    data: dict, membership: tuple[str, ...], within: float
) -> pd.DataFrame:
    """Averaged, within-outcome-shrunk pairs frame for one study."""
    index = data["index"]
    base = data["pairs"][membership[0]].loc[index].copy()
    stack = np.mean(
        [data["pairs"][r].loc[index, "estimate_l"].to_numpy() for r in membership],
        axis=0,
    )
    out = base.reset_index()
    out["estimate_l"] = stack
    if within != 1.0:
        grouped = out.groupby("outcome")["estimate_l"]
        mean = grouped.transform("mean")
        out["estimate_l"] = mean + within * (out["estimate_l"] - mean)
    return out


def section1(pairs: pd.DataFrame) -> dict:
    out = dict(MET.pooled_metrics(pairs, include_rmse=True))
    out.update(MET.run_calibration_pooled(pairs))
    out.update(MET.adjusted_metrics(pairs))
    return out


def control_metrics(data: dict, donor: str, scale: float) -> dict:
    """Level error and the four shape metrics, on the donor's control arm."""
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
        right = right.mean() + scale * (right - right.mean())
        right = right.clip(0.0, span)
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


def fit_on_training(train: list[dict], available: tuple[str, ...]) -> dict:
    """Every free choice, decided on the training studies alone."""
    best = None
    for membership in memberships(available):
        for within in WITHIN_GRID:
            frames = [effect_vector(d, membership, within) for d in train]
            pooled = pd.concat(frames, ignore_index=True)
            r = float(np.corrcoef(pooled["estimate_h"], pooled["estimate_l"])[0, 1])
            if best is None or r > best[0]:
                best = (r, membership, within)
    _, membership, within = best
    pooled = pd.concat(
        [effect_vector(d, membership, within) for d in train], ignore_index=True
    )
    human = pooled["estimate_h"].to_numpy()
    ours = pooled["estimate_l"].to_numpy()
    kappa = float(np.cov(human, ours, ddof=1)[0, 1] / np.var(ours, ddof=1))

    donor_best = None
    for donor in available:
        if any(donor not in d["controls"] for d in train):
            continue
        # residual scale that matches human dispersion on the training studies
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
        "within": within,
        "kappa": kappa,
        "donor": donor,
        "residual_scale": scale,
    }


#: Recipes whose structure is fixed a priori, so only scale parameters are fitted.
#:
#: This distinction matters more than it looks.  The shipped recipe does not
#: *choose* its membership or its within-outcome factor from data -- it averages
#: every Qwen run there is, and takes 0.5 as a pre-committed default.  Letting the
#: training folds pick those two instead adds fitting noise the real recipe never
#: pays, so scoring only the fully-fitted variant understates it.
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
    scored = []
    for run in available:
        pooled = pd.concat(
            [effect_vector(d, (run,), 1.0) for d in train], ignore_index=True
        )
        scored.append(
            (float(np.corrcoef(pooled["estimate_h"], pooled["estimate_l"])[0, 1]), run)
        )
    keep = tuple(run for r, run in scored if r > 0)
    return keep if keep else (max(scored)[1],)


RULES = {
    "rule: average all available, within 0.5": (rule_all, 0.5),
    "rule: positive training r, within 0.5": (rule_positive, 0.5),
    "rule: positive training r, no within shrink": (rule_positive, 1.0),
}


def main(argv: list[str] | None = None) -> int:
    pd.set_option("display.width", 250)
    wanted = list(argv or [])
    configs = study_config()
    if wanted:
        known = {c["name"] for c in configs}
        unknown = [w for w in wanted if w not in known]
        if unknown:
            raise SystemExit(
                f"unknown study: {', '.join(unknown)} (have {sorted(known)})"
            )
        configs = [c for c in configs if c["name"] in wanted]
    data = {cfg["name"]: load(cfg) for cfg in configs}
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
        f"x {len(WITHIN_GRID)} within-shrink values searched per fold\n"
    )

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
        row = {"held out": held, **section1(test)}
        row.update(control_metrics(data[held], fit["donor"], fit["residual_scale"]))
        row["what"] = "recipe, fitted on the other two"
        rows.append(row)

        rep = data[held]["replication"]
        rrow = {"held out": held, **section1(rep), "what": "human replication"}
        rrow.update(control_metrics(data[held], fit["donor"], fit["residual_scale"]))
        for key in ("level_err_pp", "variance_ratio", "ovl", "ks", "w1"):
            rrow.pop(key, None)
        rows.append(rrow)

        for run in available:
            single = effect_vector(data[held], (run,), 1.0)
            srow = {
                "held out": held,
                **section1(single),
                "what": f"single: {run.replace('_v3', '')}, uncalibrated",
            }
            rows.append(srow)

        # Membership decided by a pre-committed rule, instantiated on training.
        for label, (rule, within) in RULES.items():
            membership = rule(train, available)
            tr = pd.concat(
                [effect_vector(d, membership, within) for d in train],
                ignore_index=True,
            )
            k = float(
                np.cov(tr["estimate_h"], tr["estimate_l"], ddof=1)[0, 1]
                / np.var(tr["estimate_l"], ddof=1)
            )
            te = effect_vector(data[held], membership, within)
            te["estimate_l"] = k * te["estimate_l"]
            row = {"held out": held, **section1(te), "what": label}
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
            tr = pd.concat(
                [effect_vector(d, membership, within) for d in train], ignore_index=True
            )
            k = float(
                np.cov(tr["estimate_h"], tr["estimate_l"], ddof=1)[0, 1]
                / np.var(tr["estimate_l"], ddof=1)
            )
            te = effect_vector(data[held], membership, within)
            te["estimate_l"] = k * te["estimate_l"]
            rows.append({"held out": held, **section1(te), "what": label})

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
    print("\nfull metric set, recipe rows only:\n")
    keep = table[table["what"] == "recipe, fitted on the other two"]
    print(
        keep[
            ["held out"] + [c for c in s1 if c in keep] + [c for c in s3 if c in keep]
        ].to_string(index=False, float_format=lambda v: f"{v:7.3f}")
    )
    print(
        "\nshipped design, per fold (structure pre-committed, only the global factor fitted):\n"
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
        "recipe, fitted on the other two",
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
        print(f"  {what:38s} {cells}")
    return 0


if __name__ == "__main__":
    import sys

    raise SystemExit(main(sys.argv[1:]))
