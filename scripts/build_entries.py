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
    The same **without** global shrinkage.  Shrinkage is provably neutral on the
    leaderboard's sort key and close to the whole story for RMSE and beta, so this
    entry is the hedge on the one axis where the two differ at all.

``secondary-2``
    The uncalibrated best single ranker.  Insurance against every calibration
    being wrong, and the diagnostic baseline the report measures the others
    against.

**Why the primary shrinks by 0.2.**  Measured on matched pairs in every
(study, model) cell available, the RMSE-optimal factor is 0.159 / 0.125 / 0.112 for
the three models on Voelkel, 0.216 on ICPC and 0.226 on Goldwert for Qwen2.5-72B;
leave-one-study-out puts its out-of-fold estimates at 0.198-0.222.  The *ratio*
transfers even though neither of its parts does — real human effects run 1.1 to
5.0 pp across these studies and ours 2.8 to 5.5, and the quotient stays near 0.2.

An earlier version of this script shipped no shrinkage at all, on the reasoning
that a 4.5-fold spread in absolute human effect magnitudes made no target
transferable.  That compared our effects against human effects computed on *other
studies' pair sets*, which is not a comparison; matched within study it is stable.

Run: ``python scripts/build_entries.py [--team-id mpib] [--out-root ...]``
"""

from __future__ import annotations

import argparse
from pathlib import Path


from silicon_sampling.anchors import ccc as ccc_anchors
from silicon_sampling.anchors import levels as anchor_levels
from silicon_sampling.calibration import recipes as R
from silicon_sampling.calibration import tier1 as T1
from silicon_sampling.submission import build as SB
from silicon_sampling.submission import check as SC
from silicon_sampling.submission import spec as SP

RAW_EXPORT = Path("data/pfander/silicon_sampling/qwen25_7b/samples.csv")


def entries(anchors: dict[str, float]) -> list[tuple[str, R.Recipe]]:
    best = R.BEST_RANKERS
    grounded = R.GROUNDED
    common = dict(
        level_from=grounded,
        offsets_from=grounded,
        residuals_from=grounded,
        within_shrink=R.WITHIN_SHRINK,
        party_offsets_from=R.PARTY_DONOR,
        party_gap_anchors=R.PARTY_GAP_ANCHORS,
        party_gap_weight=R.PARTY_GAP_WEIGHT,
        residual_scale=R.RESIDUAL_SCALE,
        flatten_noise=True,
        level_anchors=anchors,
    )
    return [
        (
            "primary",
            R.Recipe(
                name="combined",
                effects_from=best,
                shrink=R.shrink_for_runs(best),
                **common,
            ),
        ),
        (
            "secondary-1",
            R.Recipe(name="combined-unshrunk", effects_from=best, **common),
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
    # Voelkel et al. 2026 covers six climate outcomes the TISP/CCAM crosswalk does
    # not, two of them on verbatim identical items, so it extends the anchored set
    # from three of thirteen to nine.  The two sources do not overlap, so this is a
    # union rather than a precedence decision.
    ccc_levels = ccc_anchors.levels()
    overlap = set(anchors) & set(ccc_levels)
    if overlap:  # pragma: no cover - a guard; today the two sources are disjoint
        raise SystemExit(
            f"anchor sources disagree about {sorted(overlap)}; decide precedence "
            "explicitly rather than letting dict.update pick"
        )
    anchors.update(ccc_levels)
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
