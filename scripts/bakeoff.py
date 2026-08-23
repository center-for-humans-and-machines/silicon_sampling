"""Score every calibration recipe on every scored analysis, against real humans.

The point of doing this on Voelkel rather than on Pfänder is that Voelkel publishes
participant-level responses, so every number here is a comparison against truth
rather than an internal consistency check.  The point of scoring *all* the
analyses rather than the leaderboard's sort key is that the benchmark is
multi-objective, and the calibrations reach different metrics almost disjointly —
a recipe that improves the sort key and wrecks the variance ratio is not obviously
better than one that does neither, and only a table wide enough to show both makes
that visible.

Two things to read carefully in the output.

**The human replication row is the yardstick, not 1.0.** It is a fresh half sample
of real people predicting the other half, so it bounds what any predictor can
achieve at this sample size. Voelkel's true between-arm effects are barely larger
than the noise in a half sample, so several metrics are close to unimprovable.

**Effects and everything else move independently.** Recipes that differ only in
which run supplies the level or the demographic offsets have *identical* Section-1
numbers by construction, and differ only from `level_err` rightwards. That is the
design working, not a bug in the table.

Run: ``python scripts/bakeoff.py [--bootstrap 0] [--subgroups]``
"""

from __future__ import annotations

import argparse

import pandas as pd

from silicon_sampling.benchmark import scored as SC
from silicon_sampling.benchmark.reference import half_split
from silicon_sampling.calibration import recipes as R
from silicon_sampling.calibration import tier1 as T1
from silicon_sampling.voelkel import outcomes as voc
from silicon_sampling.voelkel import paths as VP
from silicon_sampling.voelkel import score as VS

#: The runs with a Voelkel sample on disk, best ranker first.
RUNS = ("qwen25_7b", "qwen25_72b", "v4_flash")

#: Voelkel's moderators in codebook order — the first level of each is the
#: reference for every interaction and stereotyping coefficient, so the order is
#: load-bearing rather than cosmetic.
MODERATORS = {
    "gender": ("Male", "Female"),
    "age_band": ("18-29", "30-44", "45-59", "60+"),
    "race": ("White", "Black", "Hispanic", "Asian", "Other"),
    "education": ("No college", "Some college", "College", "Postgrad"),
    "party_gen": ("Republican", "Democrat", "Independent"),
}


def voelkel_instrument() -> T1.Instrument:
    """Voelkel as the calibration layer needs to see it."""
    return T1.Instrument(
        scales=dict(voc.OUTCOMES),
        control=VS.CONTROL,
        moderators=tuple(MODERATORS),
        binary=(),
        composites={},
    )


def voelkel_design(human: pd.DataFrame) -> SC.ScoredDesign:
    """Voelkel as the scoring layer needs to see it.

    Moderator levels are taken from the human data rather than from the constant
    above wherever the two disagree, because the scored reference is the human
    frame and a level it does not contain cannot be a reference level.
    """
    moderators = {}
    for name, declared in MODERATORS.items():
        if name not in human.columns:
            continue
        present = [level for level in declared if (human[name] == level).any()]
        extra = sorted(set(human[name].dropna().unique()) - set(present))
        moderators[name] = tuple(present + extra)
    conditions = [VS.CONTROL] + sorted(set(human["condition"].unique()) - {VS.CONTROL})
    return SC.ScoredDesign(
        outcomes=dict(voc.OUTCOMES),
        control=VS.CONTROL,
        moderators=moderators,
        conditions=conditions,
    )


def candidates() -> list[R.Recipe]:
    """The menu, ordered so each row differs from the one above in one way."""
    grounded = "v4_flash"
    best = ("qwen25_7b", "qwen25_72b")
    return [
        R.uncalibrated("qwen25_7b"),
        R.uncalibrated("qwen25_72b"),
        R.uncalibrated("v4_flash"),
        R.Recipe(name="avg:q7b+q72b", effects_from=best),
        R.Recipe(name="avg:all three", effects_from=(*best, grounded)),
        R.Recipe(name="avg + within-shrink 0.5", effects_from=best, within_shrink=0.5),
        R.Recipe(name="avg + shrink 0.46", effects_from=best, shrink=0.46),
        R.Recipe(
            name="avg + shrink 0.159 (Voelkel kappa)", effects_from=best, shrink=0.159
        ),
        R.Recipe(
            name="avg + v4 level",
            effects_from=best,
            level_from=grounded,
            residuals_from=grounded,
        ),
        R.Recipe(
            name="avg + v4 level & demographics",
            effects_from=best,
            level_from=grounded,
            offsets_from=grounded,
            residuals_from=grounded,
        ),
        # The two families are separable, so the combination should inherit the
        # best of each: Section 1 from the effect transforms, Sections 3 and 10-12
        # from the component swap.  Included to check that it actually does.
        R.Recipe(
            name="COMBINED: avg + within 0.5 + v4 context",
            effects_from=best,
            within_shrink=0.5,
            level_from=grounded,
            offsets_from=grounded,
            residuals_from=grounded,
        ),
        R.Recipe(
            name="COMBINED + shrink 0.46",
            effects_from=best,
            within_shrink=0.5,
            shrink=0.46,
            level_from=grounded,
            offsets_from=grounded,
            residuals_from=grounded,
        ),
    ]


#: The scored numbers worth a column, by their exact key in ``leaderboard_row``.
#: The full row carries 279 keys for this study — every outcome, intervention,
#: moderator and level separately — so a bake-off table has to name the summaries
#: rather than glob for them.  Globbing is how the first version of this script
#: silently produced a column of NaN.
KEEP = {
    "pearson_r": "pearson_r",
    "spearman_rho": "spearman_rho",
    "directional_pct": "directional_pct",
    "pearson_within": "pearson_within",
    "pearson_adj": "pearson_adj",
    "rmse": "rmse",
    "alpha": "alpha",
    "beta": "beta",
    "var_ratio": "median/shape/variance_ratio",
    "ovl": "median/shape/ovl",
    "ks": "median/shape/ks",
    "w1": "median/shape/w1",
    "baseline_rmse": "median/baseline/rmse",
    "parity_dpd": "median/parity/dpd",
    "parity_worst": "max/parity/worst_abs_err",
    "stereo_coef_rmse": "median/stereo/coef_rmse",
    "stereo_r2_gap": "median/stereo/r2_gap",
}

SECTION1 = (
    "pearson_r",
    "spearman_rho",
    "directional_pct",
    "pearson_within",
    "pearson_adj",
    "rmse",
    "alpha",
    "beta",
)


def pick(row: dict) -> dict:
    """Pull the summary columns, and fail loudly on a key that is not there."""
    missing = [key for key in KEEP.values() if key not in row]
    if missing:
        raise KeyError(f"leaderboard_row has no {missing}; it has {len(row)} keys")
    return {name: row[key] for name, key in KEEP.items()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bootstrap", type=int, default=0)
    parser.add_argument("--subgroups", action="store_true")
    args = parser.parse_args()

    humans = VS.load_humans()
    human1, human2 = half_split(humans)
    design = voelkel_design(human1)
    instrument = voelkel_instrument()
    runs = {
        name: pd.read_csv(VP.samples_dir(name) / "samples.csv", low_memory=False)
        for name in RUNS
    }

    sides = SC.reference_sides(human1, design, subgroup=args.subgroups)
    rows = []
    for recipe in candidates():
        frame, drift = R.apply(recipe, runs=runs, instrument=instrument)
        worst = float(drift["max_abs_effect_drift"].max())
        row = SC.leaderboard_row(
            human1,
            frame,
            design,
            label=recipe.name,
            sides=sides,
            bootstrap=args.bootstrap,
            subgroups=args.subgroups,
        )
        rows.append({"submission": recipe.name, "drift": worst, **pick(row)})

    replication = SC.leaderboard_row(
        human1,
        human2,
        design,
        label="human replication",
        sides=sides,
        bootstrap=args.bootstrap,
        subgroups=args.subgroups,
    )
    rows.append({"submission": "human replication", "drift": 0.0, **pick(replication)})

    table = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    head = ["submission", "drift", *SECTION1]
    print("=== Section 1: ATE recovery and calibration (pp of scale range) ===")
    print(table[head].to_string(index=False, float_format=lambda v: f"{v:8.3f}"))

    rest = ["submission"] + [c for c in table.columns if c not in head]
    print("\n=== Sections 3 and 10-12: distributions, baselines, stereotyping ===")
    print(table[rest].to_string(index=False, float_format=lambda v: f"{v:8.3f}"))
    print(
        "\nTargets: pearson_r/rho/directional high, rmse low, alpha 0, beta 1,\n"
        "var_ratio 1, ovl 1, ks 0, w1 0, and baseline_rmse / parity / stereo low.\n"
        "Read every column against the human replication row, not against the ideal."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
