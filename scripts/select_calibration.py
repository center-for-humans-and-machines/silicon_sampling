"""Run the calibration bake-off under held-out scoring, and print the verdict.

Today this runs leave-one-condition-out on Voelkel, because Voelkel is the only
study with both a silicon sample and real responses.  Once ICPC and Goldwert land
the same candidates get scored leave-one-*study*-out, which is the fold structure
that actually matters — a calibration that transfers between intervention arms of
one study has shown much less than one that transfers between studies.

The distinction is worth keeping in the output rather than in a comment, so the
fold family is printed with the results and the report can quote it.

Run: ``python scripts/select_calibration.py [--run qwen25_7b]``
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from silicon_sampling.benchmark.reference import ate_pairs, half_split
from silicon_sampling.calibration import effects as E
from silicon_sampling.calibration import select as SEL
from silicon_sampling.voelkel import paths as P
from silicon_sampling.voelkel import score as S

#: The metric the choice is made on, declared before any fitting happens.
TARGET = "pearson_r"


def voelkel_pairs(run: str) -> pd.DataFrame:
    humans = S.load_humans()
    human1, _ = half_split(humans)
    sample = pd.read_csv(P.samples_dir(run) / "samples.csv", low_memory=False)
    return ate_pairs(S.effects(human1), S.effects(sample)).dropna(
        subset=["estimate_h", "estimate_l"]
    )


def as_folds_by_outcome_group(
    pairs: pd.DataFrame, n_folds: int = 3, seed: int = 20260823
) -> list[SEL.Fold]:
    """Folds of several outcomes each, because pooling has to survive into the fold.

    Getting this wrong is easy and fails silently, so it is worth spelling out.
    Every candidate here changes pooled Pearson r only by re-weighting outcomes
    against each other — that is the entire mechanism. So a fold has to contain
    several outcomes for the transform to be visible in it at all:

    * **Leave-one-condition-out** gives each outcome a single observation, the
      per-outcome mean equals that observation, and within-outcome shrinkage is
      the *identity*.
    * **Leave-one-outcome-out** gives the fold one outcome, so within-outcome
      shrinkage is an *affine* map of that outcome's values — and Pearson r is
      affine-invariant.

    Both were tried, and both returned bit-identical held-out scores for all five
    candidates: not "no calibration helps", but "this fold family cannot see any
    calibration". Grouping outcomes keeps several in each fold and restores the
    contrast. The real answer is leave-one-*study*-out, where a fold is a whole
    study and carries all its outcomes; this is the within-study stand-in until
    ICPC and Goldwert land.
    """
    outcomes = sorted(pairs["outcome"].unique())
    order = np.random.default_rng(seed).permutation(len(outcomes))
    groups: list[list[str]] = [[] for _ in range(n_folds)]
    for position, index in enumerate(order):
        groups[position % n_folds].append(outcomes[index])
    folds = []
    for index, group in enumerate(groups):
        if not group:
            continue
        held = pairs[pairs["outcome"].isin(group)]
        folds.append(
            SEL.Fold(
                name=f"fold{index}:" + ",".join(sorted(group)),
                predicted=held.rename(columns={"estimate_l": "estimate", "se_l": "se"})[
                    ["outcome", "condition", "estimate", "se"]
                ],
                reference=held[["outcome", "condition", "estimate_h", "se_h"]],
            )
        )
    return folds


def training_effects(training: list[SEL.Fold]) -> pd.DataFrame:
    """The training folds' pairs, stacked, for fitting a parameter on."""
    frames = []
    for fold in training:
        merged = fold.reference.merge(fold.predicted, on=["outcome", "condition"])
        frames.append(merged)
    return pd.concat(frames, ignore_index=True)


def build_candidates() -> list[SEL.Candidate]:
    """The menu, each declaring what it nests inside and what it costs."""
    return [
        SEL.Candidate("raw", lambda table: table, n_parameters=0),
        SEL.Candidate(
            "global_shrink",
            lambda table: table,  # replaced by the fitted transform
            n_parameters=1,
            nests_inside="raw",
            notes="fixes beta and rmse; provably cannot move pearson_r",
        ),
        SEL.Candidate(
            "within_shrink_0.5",
            lambda table: E.shrink_within_outcome(table, 0.5),
            n_parameters=0,
            nests_inside="raw",
            notes="fixed at a conservative 0.5 rather than fitted",
        ),
        SEL.Candidate(
            "within_shrink_fitted",
            lambda table: table,  # replaced by the fitted transform
            n_parameters=1,
            nests_inside="within_shrink_0.5",
            notes="does fitting the factor beat the conservative default?",
        ),
        SEL.Candidate(
            "flatten_noise_outcomes",
            lambda table: E.flatten_outcomes(table, _noise_outcomes(table), factor=0.2),
            n_parameters=0,
            nests_inside="raw",
            notes="needs no ground truth; diagnostic is our own true_effect_sd",
        ),
    ]


def _noise_outcomes(table: pd.DataFrame) -> set[str]:
    """Outcomes whose between-message spread is indistinguishable from noise."""
    spread = E.true_effect_sd(table)
    return {name for name, value in spread.items() if value <= 1e-9}


def fitter(training: list[SEL.Fold], candidate: SEL.Candidate):
    """Fit a candidate's parameter on the training folds only."""
    if candidate.name == "global_shrink":
        train = training_effects(training)
        factor = E.optimal_shrinkage(train["estimate"], train["estimate_h"])
        return lambda table: E.global_shrink(table, factor)
    if candidate.name == "within_shrink_fitted":
        train = training_effects(training)
        best, best_score = 0.5, -np.inf
        for factor in np.arange(0.1, 1.05, 0.05):
            moved = E.shrink_within_outcome(train, factor)
            score = SEL.score_pairs(
                train.assign(estimate_l=moved["estimate"], se_l=moved["se"])
            ).get(TARGET, -np.inf)
            if score > best_score:
                best, best_score = factor, score
        return lambda table: E.shrink_within_outcome(table, best)
    return candidate.transform


def fitted_parameters(folds: list[SEL.Fold]) -> dict:
    """Each candidate's out-of-fold parameter estimates, for the stability check."""
    shrink, within = [], []
    for held_out in folds:
        training = [fold for fold in folds if fold.name != held_out.name]
        train = training_effects(training)
        shrink.append(E.optimal_shrinkage(train["estimate"], train["estimate_h"]))
        best, best_score = 0.5, -np.inf
        for factor in np.arange(0.1, 1.05, 0.05):
            moved = E.shrink_within_outcome(train, factor)
            score = SEL.score_pairs(
                train.assign(estimate_l=moved["estimate"], se_l=moved["se"])
            ).get(TARGET, -np.inf)
            if score > best_score:
                best, best_score = factor, score
        within.append(best)
    return {"global_shrink_k": shrink, "within_shrink_factor": within}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default="qwen25_7b")
    parser.add_argument("--folds", type=int, default=3)
    args = parser.parse_args()

    pairs = voelkel_pairs(args.run)
    folds = as_folds_by_outcome_group(pairs, n_folds=args.folds)
    print(f"study: Voelkel | run: {args.run}")
    print(f"fold family: grouped-outcome, {len(folds)} folds")
    print("  A fold must hold SEVERAL outcomes or no candidate is visible: these")
    print("  transforms move pooled r only by re-weighting outcomes against each")
    print("  other, so one-outcome folds make them affine (r is affine-invariant)")
    print("  and one-condition folds make them the identity. Both were tried and")
    print("  both scored every candidate identically.")
    print("  This is a within-study stand-in; leave-one-STUDY-out needs ICPC and")
    print("  Goldwert, and is the fold family that actually matters.\n")

    results = SEL.evaluate(build_candidates(), folds, fit=fitter)
    per_fold = results.pivot_table(index="candidate", columns="fold", values=TARGET)
    print(f"held-out {TARGET} per fold:")
    print(per_fold.to_string(float_format=lambda v: f"{v:+.3f}"))

    print(f"\nverdicts (target {TARGET}, ceiling 2 free parameters):")
    verdict = SEL.decide(results, metric=TARGET, max_parameters=2)
    print(
        verdict[
            [
                "candidate",
                "n_parameters",
                "nests_inside",
                "wins",
                "ties",
                "n_folds",
                "metric_mean",
                "verdict",
                "reason",
            ]
        ].to_string(index=False, float_format=lambda v: f"{v:+.3f}")
    )

    print("\nparameter stability across folds (the substitute for an interval):")
    for name, values in fitted_parameters(folds).items():
        summary = SEL.parameter_stability(values)
        print(f"  {name:24s} {summary}")

    print("\nother metrics, pooled in-sample, as constraints rather than targets:")
    table = pairs.rename(columns={"estimate_l": "estimate", "se_l": "se"})[
        ["outcome", "condition", "estimate", "se"]
    ]
    for label, moved in (
        ("raw", table),
        (
            "global_shrink(fitted)",
            E.global_shrink(
                table, E.optimal_shrinkage(table["estimate"], pairs["estimate_h"])
            ),
        ),
        ("within_shrink(0.5)", E.shrink_within_outcome(table, 0.5)),
    ):
        scored = SEL.score_pairs(
            pairs.assign(estimate_l=moved["estimate"], se_l=moved["se"])
        )
        print(
            f"  {label:22s} r={scored['pearson_r']:+.3f} rho={scored['spearman_rho']:+.3f} "
            f"dir={scored['directional_pct']:5.1f} rmse={scored['rmse']:.3f} "
            f"beta={scored['beta']:+.3f} alpha={scored['alpha']:+.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
