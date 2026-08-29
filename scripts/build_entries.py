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
import shutil
from pathlib import Path


from silicon_sampling.anchors import ccc as ccc_anchors
from silicon_sampling.anchors import levels as anchor_levels
from silicon_sampling import models as MODELS
from silicon_sampling.calibration import recipes as R
from silicon_sampling.calibration import tier1 as T1
from silicon_sampling.submission import build as SB
from silicon_sampling.submission import check as SC
from silicon_sampling.submission import spec as SP
from silicon_sampling.submission import zenodo as ZEN


#: The raw deposit must be the raw output of the run whose respondents are
#: actually submitted -- ``recipe.template_run``, i.e. the first entry of
#: ``effects_from``, which supplies the rows.  A fixed path here shipped
#: ``qwen25_7b``'s export beside rows drawn from ``qwen25_7b_demo``: a
#: transparency record of a different run than the one on trial.
def raw_export_for(recipe) -> Path:
    """The samples file of the run that supplies this recipe's respondents."""
    return Path("data/pfander/silicon_sampling") / recipe.template_run / "samples.csv"


#: The completed registration form, shared by all three entries.  ``stage_template``
#: copies the *blank* one from the benchmark template and never overwrites an
#: existing file, so the filled version is written over it here.  Items that differ
#: between entries are marked inline in the form rather than split into three files.
REGISTRATION = Path("data/pfander/submission/registration_filled.md")

#: One plain-language paragraph per entry, for ``metadata.json``'s ``abstract`` —
#: which is what Zenodo turns into the permanent record's description.  The recipe
#: string is appended as a second paragraph so the record carries both the readable
#: summary and the exact configuration.
ABSTRACTS = {
    "primary": (
        "Eighteen thousand synthetic survey respondents, each walked through the "
        "study's own questionnaire item by item by open-weight base language "
        "models. A synthetic respondent's answer is overwhelmingly set by where "
        "the response distribution sits rather than by which message they read — "
        "the intervention effect is about 0.03% of a response's variance — and the "
        "benchmark scores those two things separately, so no single model is good "
        "at all of them. Each answer is therefore assembled from four models, each "
        "supplying the one additive term it predicts best: which interventions "
        "move which outcomes, where the control-arm distribution sits, how "
        "demographic groups differ, and how spread out individual answers are. "
        "Eight of thirteen control-arm levels are set to independent published "
        "measurements of the same question rather than to a model's guess. Every "
        "free choice was fitted by nested leave-one-study-out cross-validation "
        "over four external megastudies and none on this study, which publishes "
        "no outcome data.\n\n"
        "The pipeline that produces this entry -- the code, the prompts the "
        "simulated respondents see, the calibration, and the cross-validation "
        "that selected it -- was designed and written by an LLM coding agent "
        "(Claude Code, Claude Opus 5) rather than by people. The human team "
        "chose to use base rather than instruction-tuned models, chose which "
        "ones, chose the external studies used for validation and demographic "
        "anchoring, and required a nested cross-validation."
    ),
    "secondary-1": (
        "As the team's primary entry, but without the final proportional "
        "rescaling of the predicted effects. That rescaling cannot change the "
        "rank ordering of the predictions, only their magnitude, so this entry "
        "isolates the single dimension on which the two differ — it is the same "
        "method on every metric that is invariant to a positive scalar."
        "As with the team's primary entry, the pipeline was designed and written by an LLM coding agent (Claude Code, Claude Opus 5) rather than by people."
    ),
    "secondary-2": (
        "A single open-weight base language model answering the study's "
        "questionnaire as eighteen thousand synthetic respondents, with no "
        "calibration, no ensembling, no external anchoring and no shrinkage. It "
        "is the uncalibrated baseline the team's other two entries are measured "
        "against, submitted so that the value of the calibration is visible in "
        "the field rather than only in our own validation.\n\n"
        "As with the team's other entries, the pipeline was designed and "
        "written by an LLM coding agent (Claude Code, Claude Opus 5) rather "
        "than by people."
    ),
}


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
        flatten_noise=False,
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
        # The registration form asks for "exact identifiers incl. provider, size,
        # version" -- our internal run names are neither, so they are mapped back
        # to the Hugging Face ids the runs actually loaded.  Several runs share a
        # model; the list is of models, not of runs.
        used_runs = tuple(
            dict.fromkeys(
                (
                    *recipe.effect_runs,
                    *filter(
                        None,
                        [
                            recipe.level_from,
                            recipe.offsets_from,
                            recipe.residuals_from,
                            recipe.party_offsets_from,
                        ],
                    ),
                )
            )
        )
        models = tuple(dict.fromkeys(MODELS.MODELS.get(r, r) for r in used_runs))
        # The authoring stage belongs in the machine-readable metadata too, not
        # only in registration.md: the pipeline that produces these predictions
        # was designed and written by an LLM coding agent rather than by people,
        # and a benchmark about LLM capability should be able to see that without
        # reading prose.
        family = (
            (
                "per-respondent simulation, multi-model component hybrid, zero-shot"
                if len(models) > 1
                else "per-respondent simulation, single model, zero-shot"
            )
            + "; pipeline authored by an LLM coding agent (Claude Code / Claude Opus 5)"
        )
        result = SB.build_submission(
            frame,
            out_dir=out_dir,
            meta=SB.SubmissionMeta(
                team_id=args.team_id,
                entry=entry,
                abstract=(
                    f"{ABSTRACTS[entry]}\n\n" f"Configuration: {R.describe(recipe)}."
                ),
                models=list(models),
                approach_family=family,
            ),
            raw_export=(raw if (raw := raw_export_for(recipe)).exists() else None),
            template_root=SP.default_template_root(),
            overwrite=True,
        )
        print(f"    wrote {result.predictions.name} ({result.rows} rows)")

        if REGISTRATION.exists():
            shutil.copyfile(REGISTRATION, out_dir / "registration.md")
            print(f"    registration: {REGISTRATION.name}")
        else:  # pragma: no cover - the blank template stays in place
            print(f"    registration: {REGISTRATION} missing, blank form left in place")

        # `.zenodo.json` controls the permanent Zenodo record a GitHub release
        # creates.  Without it Zenodo auto-generates one with an empty description
        # and no license, for a DOI that cannot be withdrawn.
        payload = ZEN.write_zenodo(out_dir)
        print(f"    zenodo: {payload['title']}")

        verdict = SC.check_repo(out_dir)
        print(f"    check: {verdict.verdict}  {verdict.counts()}")
        for row in verdict.failures + verdict.warnings:
            print(f"      {row.status}: {row.check} — {row.detail}")
        failures += not verdict.passed

    print("\nall three entries built" if not failures else f"\n{failures} FAILED")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
