"""Build the Pfänder Tier-1 entries and validate every one of them.

The benchmark allows three Tier-1 entries per team and scores all of them, with no
preregistered test between approaches — so hedging costs nothing statistically.
The three built here differ along the axes cross-validation genuinely cannot
resolve at six intervention clusters, rather than being near-duplicates:

``primary``
    The combined recipe: averaged effects from the two best rankers, within-outcome
    shrinkage, and levels, demographic offsets and residual structure from the run
    three independent sources agree is closest to real response levels.  Best on
    the leaderboard's sort key and on the reported distribution and demographic
    sections simultaneously, because those two families of calibration are
    separable.

``secondary-1``
    The same, with a global effect shrinkage toward the mid-climate reference.
    Hedges the one parameter whose right value is genuinely unknown: our effect
    scale sits *below* both climate references and *above* the democratic-norms
    one, so the direction of the correction is undetermined.

``secondary-2``
    The uncalibrated best single ranker.  Insurance against every calibration
    being wrong, and the diagnostic baseline the report measures the others
    against.

**Why no global shrinkage in the primary.**  The factor is not a property of the
sampler, it is the ratio of real effects to ours, and real human intervention
effects differ 4.5-fold between the reference studies (Voelkel 1.125 pp, Goldwert
2.967, ICPC 5.035).  Our averaged Pfänder effects run 2.467 pp after within-outcome
shrinkage, so the implied factor is 0.46 against Voelkel but 1.20 against Goldwert
and 2.04 against ICPC.  Pfänder is a climate study, both climate references sit
above our scale, and so shrinking further would move us away from them.

Run: ``python scripts/build_entries.py [--team-id mpib] [--out-root ...]``
"""

from __future__ import annotations

import argparse
from pathlib import Path


from silicon_sampling.anchors import levels as anchor_levels
from silicon_sampling.calibration import recipes as R
from silicon_sampling.calibration import tier1 as T1
from silicon_sampling.submission import build as SB
from silicon_sampling.submission import check as SC
from silicon_sampling.submission import spec as SP

#: Mid-point of the two climate references, used only by the shrunk variant.
CLIMATE_MID_KAPPA = 1.2

RAW_EXPORT = Path("data/pfander/silicon_sampling/qwen25_7b/samples.csv")


def entries(anchors: dict[str, float]) -> list[tuple[str, R.Recipe]]:
    best = R.BEST_RANKERS
    grounded = R.GROUNDED
    common = dict(
        level_from=grounded,
        offsets_from=grounded,
        residuals_from=grounded,
        within_shrink=0.5,
        flatten_noise=True,
        level_anchors=anchors,
    )
    return [
        ("primary", R.Recipe(name="combined", effects_from=best, **common)),
        (
            "secondary-1",
            R.Recipe(
                name="combined+climate-shrink",
                effects_from=best,
                shrink=CLIMATE_MID_KAPPA,
                **common,
            ),
        ),
        ("secondary-2", R.uncalibrated(best[0])),
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--team-id", default="mpib")
    parser.add_argument("--out-root", default="data/pfander/submission")
    parser.add_argument("--anchor-grade", default="near")
    args = parser.parse_args()

    design = T1.pfander_instrument()
    anchors = anchor_levels.levels(min_grade=args.anchor_grade)
    print(f"level anchors at grade '{args.anchor_grade}': {len(anchors)} outcomes")
    for outcome, value in anchors.items():
        print(f"    {outcome:24s} {value:7.2f}")

    runs = R.load_runs(tuple(dict.fromkeys((*R.BEST_RANKERS, R.GROUNDED))))
    root = Path(args.out_root)
    failures = 0

    for entry, recipe in entries(anchors):
        frame, drift = R.apply(recipe, runs=runs, instrument=design)
        worst = float(drift["max_abs_effect_drift"].max())
        gap = T1.composite_consistency(frame, design)
        print(f"\n=== {entry}: {R.describe(recipe)}")
        print(
            f"    rows {len(frame)}  effect drift {worst:.2e}  composite gap {gap:.5f}"
        )

        out_dir = root / entry
        models = tuple(
            dict.fromkeys((*recipe.effect_runs, *filter(None, [recipe.level_from])))
        )
        result = SB.build_submission(
            frame,
            out_dir=out_dir,
            meta=SB.SubmissionMeta(
                team_id=args.team_id,
                entry=entry,
                abstract=f"{R.describe(recipe)} — {recipe.notes}".strip(" —"),
                models=list(models),
            ),
            raw_export=RAW_EXPORT if RAW_EXPORT.exists() else None,
            template_root=SP.default_template_root(),
            overwrite=True,
        )
        print(f"    wrote {result.predictions.name} ({result.rows} rows)")

        verdict = SC.check_repo(out_dir)
        print(f"    check: {verdict.verdict}  {verdict.counts()}")
        for row in verdict.failures + verdict.warnings:
            print(f"      {row.status}: {row.check} — {row.detail}")
        failures += not verdict.passed

    print("\nall three entries built" if not failures else f"\n{failures} FAILED")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
