"""Nested leave-one-study-out cross-validation of the *shipped recipe*, scored
on every section of the benchmark rather than on effects alone.

``nested_cv.py`` grades effect vectors.  That covers the leaderboard's sort key
and nothing else, and it grades a *reimplementation* of the recipe -- average the
runs, shrink within outcome, shrink globally, borrow a control arm -- rather than
the object :mod:`silicon_sampling.calibration.recipes` actually builds.  Two
consequences followed and both were invisible from inside it:

* three of the benchmark's four sections were never cross-validated at all, so
  the shipped predictions for response distributions, demographic baselines and
  subgroup recovery rested on one in-sample study;
* the validated artefact was not the submitted artefact.  Anything the real
  recipe does that the miniature does not -- the component swap, the residual
  rescale, the binary flip, the composite realignment, the clipping inside
  ``tier1.calibrate`` -- was never on trial.

This closes both.  For each held-out study the free parameters are fitted on the
other three, a **real** :class:`~silicon_sampling.calibration.recipes.Recipe` is
assembled from them, :func:`~silicon_sampling.calibration.recipes.apply` turns it
into a respondent-level frame using only that study's silicon samples, and
:func:`~silicon_sampling.benchmark.scored.leaderboard_row` scores that frame
against the held-out study's human reference half on every scored analysis.

**What still cannot be cross-validated here, stated rather than hidden.** The
shipped entry pins eight of thirteen control-arm levels to external anchors
(TISP and CCC) and blends its party offsets toward external party gaps.  Those
anchors exist for Pfänder and not for the reference studies, so every recipe
scored here is the *unanchored* recipe.  The anchoring is graded separately, by
holding CCC out in ``scripts/score_ccc_holdout.py``.  Reading a level or party
number from this script as if it described the submission would understate it.

Run: ``python scripts/nested_benchmark.py [--bootstrap 1000] [--studies ...]``
"""

from __future__ import annotations

import argparse
import warnings

import numpy as np
import pandas as pd

from silicon_sampling import models as MODELS
from silicon_sampling.benchmark import scored as SC
from silicon_sampling.benchmark.reference import half_split
from silicon_sampling.calibration import effects as E
from silicon_sampling.calibration import folds as F
from silicon_sampling.calibration import recipes as R

warnings.filterwarnings("ignore")

#: Canonical run names, in the order :mod:`nested_cv` uses them.
RUNS = ("qwen25_7b_v3", "qwen25_72b_v3", "v4_flash_v3", "muse_glimmer_30b")

#: The grid the within-outcome factor is fitted over when it is fitted at all.
WITHIN_GRID = tuple(np.round(np.arange(0.0, 1.01, 0.1), 2))

#: The summary columns worth a table, keyed by their name in ``leaderboard_row``.
#:
#: The full row carries several hundred keys per study -- every outcome,
#: intervention, moderator and level separately -- so the summaries have to be
#: named.  Globbing for them is how a column of NaN gets printed as a result.
KEEP = {
    # Section 1 — average treatment effects
    "pearson_r": "pearson_r",
    "spearman_rho": "spearman_rho",
    "directional_pct": "directional_pct",
    "pearson_within": "pearson_within",
    "pearson_adj": "pearson_adj",
    "rmse": "rmse",
    "rmse_adj": "rmse_adj",
    "alpha": "alpha",
    "beta": "beta",
    # Section 2 — subgroup (condition x moderator) effects
    "sub_pearson_r": "subgroup/pooled/pearson_r",
    "sub_spearman_rho": "subgroup/pooled/spearman_rho",
    "sub_directional_pct": "subgroup/pooled/directional_pct",
    "sub_pearson_adj": "subgroup/pooled/pearson_adj",
    # Section 3 — control-arm response distributions
    "variance_ratio": "median/shape/variance_ratio",
    "ovl": "median/shape/ovl",
    "ks": "median/shape/ks",
    "w1": "median/shape/w1",
    # Sections 10-12 — demographic baselines, parity, stereotyping
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
    "rmse_adj",
    "alpha",
    "beta",
)
SECTION2 = (
    "sub_pearson_r",
    "sub_spearman_rho",
    "sub_directional_pct",
    "sub_pearson_adj",
)
SECTION3 = ("variance_ratio", "ovl", "ks", "w1")
SECTIONS_10_12 = (
    "baseline_rmse",
    "parity_dpd",
    "parity_worst",
    "stereo_coef_rmse",
    "stereo_r2_gap",
)


class Fold:
    """One study, loaded once: humans split, runs read, reference sides fitted."""

    def __init__(self, study: F.FoldStudy, subgroups: bool = True) -> None:
        self.study = study
        self.name = study.name
        self.design = study.design
        self.dropped_moderators: list[str] = []
        self.instrument = study.instrument
        humans = study.prepare(study.load_humans())
        self.human1, self.human2 = half_split(humans)
        self.reference = study.effects(self.human1)
        self.runs: dict[str, pd.DataFrame] = {}
        for run in RUNS:
            key = MODELS.resolve_run(study.samples_dir, run)
            if key is None:
                continue
            self.runs[run] = study.prepare(
                pd.read_csv(study.samples_dir(key) / "samples.csv", low_memory=False)
            )
        if subgroups:
            self.design = self._restrict_moderators()
        self.sides = SC.reference_sides(self.human1, self.design, subgroup=subgroups)
        # Effect vectors, on the grid the fitting search runs over.
        self.effects = {
            run: R.effect_table(frame, self.instrument).set_index(
                ["outcome", "condition"]
            )
            for run, frame in self.runs.items()
        }
        self.human_effects = self.reference.set_index(["outcome", "condition"])
        index = None
        for table in self.effects.values():
            index = table.index if index is None else index.intersection(table.index)
        self.index = index.intersection(self.human_effects.index)

    def _restrict_moderators(self):
        """Keep only moderator levels that every side actually has.

        The benchmark asserts the subgroup grid rather than joining loosely, and
        it is right to: a level present on one side and not the other silently
        drops that cell from the test set.  Levels are therefore intersected
        across the reference half and every run before any model is fitted, and a
        moderator that loses all but one level is dropped with a note.  This is a
        fold-construction step, not a scoring choice -- Pfänder's submission
        format fixes every level in advance, so nothing here applies there.
        """
        frames = [self.human1, self.human2, *self.runs.values()]
        keep: dict[str, tuple[str, ...]] = {}
        for name, levels in self.design.moderators.items():
            present = set(levels)
            for frame in frames:
                if name not in frame.columns:
                    present = set()
                    break
                seen = set(frame[name].dropna().astype(str).unique())
                counts = frame[name].astype(str).value_counts()
                present &= {
                    level
                    for level in seen
                    if counts.get(level, 0) >= self.design.min_group_n
                }
            if len(present) >= 2:
                keep[name] = tuple(level for level in levels if level in present)
            else:
                self.dropped_moderators.append(name)
        from dataclasses import replace

        return replace(self.design, moderators=keep)

    def vector(self, membership: tuple[str, ...], within: float) -> pd.DataFrame:
        """The averaged, within-shrunk effect vector, paired with the humans.

        Built with the same functions the recipe uses, so a fitting search cannot
        optimise one transform and the submission apply a different one.
        """
        recipe = R.Recipe(name="fit", effects_from=membership, within_shrink=within)
        targets = (
            R.effect_table(self.runs[membership[0]], self.instrument)
            if len(membership) == 1
            else R.averaged_effects(recipe, self.runs, self.instrument)
        )
        if within is not None:
            targets = E.shrink_within_outcome(
                targets, within, reference=self.instrument.control
            )
        targets = targets.set_index(["outcome", "condition"]).loc[self.index]
        return pd.DataFrame(
            {
                "outcome": self.index.get_level_values("outcome"),
                "condition": self.index.get_level_values("condition"),
                "estimate_h": self.human_effects.loc[self.index, "estimate"].to_numpy(),
                "estimate_l": targets["estimate"].to_numpy(),
            }
        )


def pooled_training_r(
    train: list[Fold], membership: tuple[str, ...], within: float
) -> float:
    frame = pd.concat([f.vector(membership, within) for f in train], ignore_index=True)
    return float(np.corrcoef(frame["estimate_h"], frame["estimate_l"])[0, 1])


def memberships(available: tuple[str, ...]) -> tuple[tuple[str, ...], ...]:
    import itertools

    out: list[tuple[str, ...]] = []
    for size in range(1, len(available) + 1):
        out.extend(itertools.combinations(available, size))
    return tuple(out)


def fit_effect_side(
    train: list[Fold], available: tuple[str, ...]
) -> tuple[tuple[str, ...], float, float]:
    """Membership and within-factor by training pooled r; kappa by training slope."""
    best = None
    for membership in memberships(available):
        for within in WITHIN_GRID:
            r = pooled_training_r(train, membership, within)
            if best is None or r > best[0]:
                best = (r, membership, within)
    _, membership, within = best
    frame = pd.concat([f.vector(membership, within) for f in train], ignore_index=True)
    human = frame["estimate_h"].to_numpy()
    ours = frame["estimate_l"].to_numpy()
    kappa = float(np.cov(human, ours, ddof=1)[0, 1] / np.var(ours, ddof=1))
    return membership, float(within), kappa


def fit_kappa(train: list[Fold], membership: tuple[str, ...], within: float) -> float:
    frame = pd.concat([f.vector(membership, within) for f in train], ignore_index=True)
    return float(
        np.cov(frame["estimate_h"], frame["estimate_l"], ddof=1)[0, 1]
        / np.var(frame["estimate_l"], ddof=1)
    )


def fit_structure(train: list[Fold], available: tuple[str, ...]) -> tuple[str, float]:
    """Structural donor and residual scale, on the training studies' control arms.

    The residual scale is the factor that puts the donor's within-arm dispersion
    on the humans'; the donor is the run whose control arm minimises level error
    and dispersion error together, weighted as the shipped search weighted them.
    """
    best = None
    for donor in available:
        if any(donor not in f.runs for f in train):
            continue
        ratios, levels = [], []
        for fold in train:
            control = fold.design.control
            human = fold.human1
            synth = fold.runs[donor]
            hcol = human[fold.design.condition_col].astype(str)
            scol = synth[fold.design.condition_col].astype(str)
            for outcome, span in fold.design.outcomes.items():
                if outcome not in human.columns or outcome not in synth.columns:
                    continue
                left = pd.to_numeric(
                    human.loc[hcol == control, outcome], errors="coerce"
                ).dropna()
                right = pd.to_numeric(
                    synth.loc[scol == control, outcome], errors="coerce"
                ).dropna()
                if len(left) < 30 or len(right) < 30 or left.std() == 0:
                    continue
                ratios.append(right.std() / left.std())
                levels.append(abs(right.mean() - left.mean()) / span * 100)
        if not ratios:
            continue
        scale = float(1.0 / np.mean(ratios))
        loss = float(np.mean(levels)) / 10.0 + abs(float(np.mean(ratios)) * scale - 1)
        if best is None or loss < best[0]:
            best = (loss, donor, scale)
    if best is None:
        return available[0], 1.0
    return best[1], best[2]


def build_recipe(
    name: str,
    membership: tuple[str, ...],
    within: float,
    kappa: float,
    donor: str,
    residual_scale: float,
) -> R.Recipe:
    """A real recipe, with every term the shipped one has that transfers."""
    return R.Recipe(
        name=name,
        effects_from=membership,
        within_shrink=within,
        shrink=kappa,
        level_from=donor,
        offsets_from=donor,
        residuals_from=donor,
        residual_scale=residual_scale,
    )


def score(
    fold: Fold, frame: pd.DataFrame, label: str, bootstrap: int, subgroups: bool
) -> dict:
    row = SC.leaderboard_row(
        fold.human1,
        frame,
        fold.design,
        label=label,
        sides=fold.sides,
        bootstrap=bootstrap,
        subgroups=subgroups,
    )
    missing = [key for key in KEEP.values() if key not in row]
    picked = {
        name: (row[key] if key in row else float("nan")) for name, key in KEEP.items()
    }
    if missing:
        picked["_missing"] = ",".join(missing)
    return picked


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bootstrap", type=int, default=0)
    parser.add_argument("--no-subgroups", action="store_true")
    parser.add_argument("--studies", nargs="*", default=None)
    args = parser.parse_args(argv)
    subgroups = not args.no_subgroups

    pd.set_option("display.width", 260)
    pd.set_option("display.max_columns", 60)

    studies = F.load_folds(args.studies)
    data = {study.name: Fold(study, subgroups=subgroups) for study in studies}
    names = list(data)

    for name in names:
        fold = data[name]
        note = (
            f"  (dropped: {', '.join(fold.dropped_moderators)})"
            if fold.dropped_moderators
            else ""
        )
        print(
            f"  {name:9s} outcomes={len(fold.design.outcomes):2d} "
            f"arms={len(fold.design.conditions):2d} "
            f"moderators={list(fold.design.moderators)}{note}"
        )
    print()

    available = tuple(r for r in RUNS if all(r in data[n].runs for n in names))
    print("=== run availability ===\n")
    for run in RUNS:
        missing = [n for n in names if run not in data[n].runs]
        mark = "candidate" if run in available else "EXCLUDED"
        note = f"  (missing: {', '.join(missing)})" if missing else ""
        print(f"  {run:20s} {mark:10s}{note}")
    print()

    prior = tuple(r for r in ("qwen25_7b_v3", "qwen25_72b_v3") if r in available)
    rows, chosen = [], []
    for held in names:
        fold = data[held]
        train = [data[n] for n in names if n != held]
        membership, within, kappa = fit_effect_side(train, available)
        donor, residual_scale = fit_structure(train, available)
        chosen.append(
            {
                "held out": held,
                "membership": ",".join(membership).replace("_v3", ""),
                "within": within,
                "kappa": round(kappa, 3),
                "donor": donor.replace("_v3", ""),
                "residual_scale": round(residual_scale, 3),
            }
        )

        variants = [
            (
                "recipe, everything fitted on the other three",
                membership,
                within,
                kappa,
            ),
        ]
        if prior:
            variants.append(
                (
                    "recipe, membership by prior, within fitted",
                    prior,
                    within,
                    fit_kappa(train, prior, within),
                )
            )
            variants.append(
                (
                    "recipe, structure pre-committed (shipped design)",
                    prior,
                    R.WITHIN_SHRINK,
                    fit_kappa(train, prior, R.WITHIN_SHRINK),
                )
            )
        variants.append(
            (
                "rule: average all available, within 0.5",
                available,
                R.WITHIN_SHRINK,
                fit_kappa(train, available, R.WITHIN_SHRINK),
            )
        )

        for label, runs_used, w, k in variants:
            recipe = build_recipe(label, runs_used, w, k, donor, residual_scale)
            frame, drift = R.apply(recipe, runs=fold.runs, instrument=fold.instrument)
            worst = float(drift["max_abs_effect_drift"].max())
            rows.append(
                {
                    "held out": held,
                    "what": label,
                    "drift": worst,
                    **score(fold, frame, label, args.bootstrap, subgroups),
                }
            )

        for run in available:
            frame = fold.runs[run]
            rows.append(
                {
                    "held out": held,
                    "what": f"single: {run.replace('_v3', '')}, uncalibrated",
                    "drift": 0.0,
                    **score(fold, frame, run, args.bootstrap, subgroups),
                }
            )

        rows.append(
            {
                "held out": held,
                "what": "human replication",
                "drift": 0.0,
                **score(
                    fold, fold.human2, "human replication", args.bootstrap, subgroups
                ),
            }
        )

    print(
        "=== what the training folds chose, never having seen the held-out study ===\n"
    )
    print(pd.DataFrame(chosen).to_string(index=False))

    table = pd.DataFrame(rows)
    for title, columns in (
        ("Section 1 — average treatment effects", SECTION1),
        ("Section 2 — subgroup effects", SECTION2),
        ("Section 3 — control-arm response distributions", SECTION3),
        ("Sections 10-12 — baselines, parity, stereotyping", SECTIONS_10_12),
    ):
        have = [c for c in columns if c in table.columns]
        print(f"\n\n=== {title}: fold means ===\n")
        summary = (
            table.groupby("what", sort=False)[have]
            .mean()
            .reindex(list(dict.fromkeys(table["what"])))
        )
        print(summary.to_string(float_format=lambda v: f"{v:8.3f}"))

    print("\n\n=== per fold, sort key only ===\n")
    print(
        table.pivot_table(
            index="what", columns="held out", values="pearson_r"
        ).to_string(float_format=lambda v: f"{v:7.3f}")
    )
    if "_missing" in table.columns and table["_missing"].notna().any():
        print("\nmissing leaderboard keys:", sorted(set(table["_missing"].dropna())))
    print(
        "\nEvery recipe row here is the UNANCHORED recipe: the level and party-gap\n"
        "anchors the submission uses exist for Pfänder and not for these studies.\n"
        "Section 3 and Sections 10-12 are correspondingly pessimistic for the entry."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
