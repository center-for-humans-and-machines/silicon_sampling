"""Leave-one-study-out selection: does a calibration transfer between studies?

This is the test the whole calibration argument rests on, and it is the one the
project could not run until now, because it needs a silicon sample *and* real
participant responses for several studies at once.

The honest framing first, because it constrains what can be concluded. With three
studies this is a three-point transfer check, not cross-validation: no mean across
three folds has a usable standard error, so none is computed. What three folds can
establish is whether a calibration **dominates** the simpler alternative it nests
inside in every fold, and whether a fitted parameter is **stable** across them.
That is all, and it is reported as such.

Why leave-one-*study*-out matters more than the within-study folds used until now:
a parameter that transfers between the intervention arms of one questionnaire has
shown almost nothing, because the arms share an instrument, a topic, a population
and a year. Between studies all four change. The measurement that motivated this
script is exactly of that kind — the global shrinkage factor is beautifully stable
across Voelkel's own folds (0.137-0.185) and useless between studies, because real
human effect magnitudes differ 4.5-fold across them.

Run: ``python scripts/loso.py [--model qwen25_72b] [--target pearson_r]``
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from silicon_sampling import models as MODELS
from silicon_sampling.benchmark.reference import ate_pairs, half_split
from silicon_sampling.calibration import effects as E
from silicon_sampling.calibration import select as SEL


@dataclass(frozen=True)
class Study:
    """One study's scoring entry points, enough to build a fold from."""

    name: str
    load_humans: Callable[[], pd.DataFrame]
    effects: Callable[[pd.DataFrame], pd.DataFrame]
    samples_dir: Callable[[str], Path]


def registry() -> list[Study]:
    """Every study with both a human reference and a silicon sample on disk.

    Imported lazily and individually, so a study whose package is mid-build does
    not stop the others from being scored.
    """
    found: list[Study] = []

    def add(name: str, module_path: str, paths_path: str) -> None:
        try:
            score = __import__(module_path, fromlist=["x"])
            paths = __import__(paths_path, fromlist=["x"])
        except Exception as exc:  # pragma: no cover - a package mid-build
            print(f"  skipping {name}: {type(exc).__name__}: {exc}")
            return
        found.append(
            Study(
                name=name,
                load_humans=score.load_humans,
                effects=score.effects,
                samples_dir=paths.samples_dir,
            )
        )

    add("voelkel", "silicon_sampling.voelkel.score", "silicon_sampling.voelkel.paths")
    add("icpc", "silicon_sampling.icpc.score", "silicon_sampling.icpc.paths")
    add(
        "goldwert", "silicon_sampling.goldwert.score", "silicon_sampling.goldwert.paths"
    )
    return found


def fold_for(study: Study, model: str) -> tuple[SEL.Fold | None, str | None]:
    """Our effects and the human effects for one study, with the run key used.

    Returns ``(None, None)`` when the study has no sample for this model at all.
    The run key comes back alongside the fold because it is not always the one
    asked for: only ICPC and Goldwert were re-sampled for the ``_v3`` audit, so
    the other studies resolve to their v1 run, and a reader has to be told which
    sample a fold was built from.
    """
    run = MODELS.resolve_run(study.samples_dir, model)
    if run is None:
        return None, None
    path = study.samples_dir(run) / "samples.csv"
    sample = pd.read_csv(path, low_memory=False)
    if len(sample) < 100:
        return None, run
    human1, _ = half_split(study.load_humans())
    pairs = ate_pairs(study.effects(human1), study.effects(sample)).dropna(
        subset=["estimate_h", "estimate_l"]
    )
    if len(pairs) < 6:
        return None, run
    predicted = pairs.rename(columns={"estimate_l": "estimate", "se_l": "se"})[
        ["outcome", "condition", "estimate", "se"]
    ]
    # Outcome names are scoped by study, so a pooled fit cannot silently join
    # two studies' identically-named outcomes.
    predicted = predicted.assign(outcome=study.name + "/" + predicted["outcome"])
    reference = pairs[["outcome", "condition", "estimate_h", "se_h"]].assign(
        outcome=study.name + "/" + pairs["outcome"]
    )
    return SEL.Fold(name=study.name, predicted=predicted, reference=reference), run


def effect_scale(fold: SEL.Fold) -> dict:
    """How big real effects are in this study, against how big ours are.

    The two numbers whose ratio is the global shrinkage factor — reported per
    study because their disagreement is the reason that factor does not transfer.
    """
    merged = fold.reference.merge(fold.predicted, on=["outcome", "condition"])
    return {
        "study": fold.name,
        "n_pairs": len(merged),
        "human_mean_signed": float(merged["estimate_h"].mean()),
        "human_mean_abs": float(merged["estimate_h"].abs().mean()),
        "ours_mean_abs": float(merged["estimate"].abs().mean()),
        "implied_kappa": (
            float(merged["estimate_h"].abs().mean() / merged["estimate"].abs().mean())
            if merged["estimate"].abs().mean() > 0
            else float("nan")
        ),
    }


def candidates() -> list[SEL.Candidate]:
    """The menu that survived the within-study rounds, plus its nesting order."""
    return [
        SEL.Candidate("raw", lambda t: t, n_parameters=0),
        SEL.Candidate(
            "within_shrink_0.5",
            lambda t: E.shrink_within_outcome(t, 0.5),
            n_parameters=0,
            nests_inside="raw",
            notes="conservative fixed factor rather than a fitted one",
        ),
        SEL.Candidate(
            "within_shrink_fitted",
            lambda t: t,
            n_parameters=1,
            nests_inside="within_shrink_0.5",
            notes="does fitting the factor on other studies beat the default?",
        ),
        SEL.Candidate(
            "global_shrink_fitted",
            lambda t: t,
            n_parameters=1,
            nests_inside="raw",
            notes="the factor measured NOT to transfer; included to show it",
        ),
    ]


def training_frame(training: list[SEL.Fold]) -> pd.DataFrame:
    return pd.concat(
        [f.reference.merge(f.predicted, on=["outcome", "condition"]) for f in training],
        ignore_index=True,
    )


def best_within_factor(train: pd.DataFrame, target: str) -> float:
    best, best_score = 0.5, -np.inf
    for factor in np.arange(0.1, 1.05, 0.05):
        moved = E.shrink_within_outcome(train, factor)
        scored = SEL.score_pairs(
            train.assign(estimate_l=moved["estimate"], se_l=moved["se"])
        ).get(target, -np.inf)
        if scored > best_score:
            best, best_score = float(factor), scored
    return best


def fitter(target: str):
    def fit(training: list[SEL.Fold], candidate: SEL.Candidate):
        if candidate.name == "global_shrink_fitted":
            train = training_frame(training)
            factor = E.optimal_shrinkage(train["estimate"], train["estimate_h"])
            return lambda t: E.global_shrink(t, factor)
        if candidate.name == "within_shrink_fitted":
            factor = best_within_factor(training_frame(training), target)
            return lambda t: E.shrink_within_outcome(t, factor)
        return candidate.transform

    return fit


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="qwen25_72b")
    parser.add_argument("--target", default="pearson_r")
    args = parser.parse_args()

    print(f"model: {args.model} | target metric: {args.target}\n")
    folds = []
    for study in registry():
        fold, run = fold_for(study, args.model)
        if fold is None:
            print(f"  {study.name}: no usable sample for {args.model} yet")
            continue
        folds.append(fold)
        note = "" if run == args.model else f"  [from {run}: template unrevised]"
        print(f"  {study.name}: {len(fold.predicted)} pairs{note}")

    if len(folds) < 2:
        print(
            "\nFewer than two studies are sampled, so nothing can be held out."
            "\nThis is a real result about coverage, not a failure — rerun when the"
            "\nremaining samples land."
        )
        return 0

    print("\n=== how big real effects are, per study (pp of scale range) ===")
    scale = pd.DataFrame([effect_scale(f) for f in folds])
    print(scale.to_string(index=False, float_format=lambda v: f"{v:8.3f}"))
    span = scale["implied_kappa"].max() / scale["implied_kappa"].min()
    print(
        f"\nimplied shrinkage factor spans {span:.1f}x across studies"
        f" ({scale['implied_kappa'].min():.2f} to {scale['implied_kappa'].max():.2f})."
        + (
            "  It crosses 1.0, so the direction of the correction is undetermined."
            if scale["implied_kappa"].min() < 1 < scale["implied_kappa"].max()
            else ""
        )
    )

    results = SEL.evaluate(candidates(), folds, fit=fitter(args.target))
    print(f"\n=== held-out {args.target}, leave-one-STUDY-out ({len(folds)} folds) ===")
    print(
        results.pivot_table(
            index="candidate", columns="fold", values=args.target
        ).to_string(float_format=lambda v: f"{v:+.3f}")
    )

    print("\n=== verdicts under the pre-committed rules ===")
    verdict = SEL.decide(results, metric=args.target, max_parameters=2)
    print(
        verdict[
            [
                "candidate",
                "n_parameters",
                "nests_inside",
                "wins",
                "ties",
                "n_folds",
                "verdict",
                "reason",
            ]
        ].to_string(index=False, float_format=lambda v: f"{v:.0f}")
    )

    print("\n=== out-of-fold parameter stability (the substitute for an interval) ===")
    for name, getter in (
        (
            "global_shrink",
            lambda tr: E.optimal_shrinkage(tr["estimate"], tr["estimate_h"]),
        ),
        ("within_shrink", lambda tr: best_within_factor(tr, args.target)),
    ):
        values = []
        for held in folds:
            train = training_frame([f for f in folds if f.name != held.name])
            values.append(float(getter(train)))
        print(f"  {name:14s} {SEL.parameter_stability(values)}")

    print(
        "\nWith this many folds only two things are readable: whether a candidate"
        "\ndominates in every fold, and whether its parameter is stable. No mean"
        "\nacross folds is computed, because none would have a usable interval."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
