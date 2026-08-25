"""Named, reproducible calibration recipes for the Pfänder submission.

A recipe is the whole chain from runs on disk to a calibrated respondent-level
frame: which model supplies which term, which effect transform is applied, which
external anchors are trusted.  Keeping them named and declarative rather than
assembled ad hoc at the point of use matters for two reasons.

**The submission has to say what was done to it.**  Up to three Tier-1 entries are
allowed and the interesting ones differ along axes cross-validation cannot
resolve, so each entry needs a stable identity that the report and the
``metadata.json`` abstract can both refer to without re-deriving it.

**A recipe is the unit that gets validated.**  The selection harness scores
candidates out of fold; what it scores has to be the same object that later
produces the file, or the evidence is about something else.  Hence
:func:`apply` takes a :class:`Recipe` and nothing else, and the same call is used
in the bake-off and in the build.

## Why the default recipe is a hybrid rather than a model

The scored analyses read different terms of
``y = level + effect + offset + residual`` almost disjointly, and our two
samplers are good at different terms.  Qwen2.5-7B ranks interventions far better
(pooled Pearson r 0.408 against 0.190 on Voelkel); DeepSeek-V4-Flash is closer on
levels, carries demographic signal Qwen has none of, and produces internally
coherent respondents where Qwen's correlate +0.000 between trust and distrust.

Three independent sources agree on the level point, which is why it is the
default rather than a guess: Voelkel's direct human comparison (mean absolute
level error 8.0 pp against 22.9), the TISP trust battery on Pfänder's own primary
construct (V4-Flash within 2.0-3.1 points, Qwen off by 14.2 and 9.4), and
Goldwert's donation item (V4-Flash 4.726 against a human 4.774, Qwen 2.990).
Different studies, different constructs, same answer.

The honest exception is ``newsletter_signup``, where V4-Flash is *worse*
(0.363 against Goldwert's human 0.243) than Qwen (0.311).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from ..pfander import outcomes as pfander_outcomes
from ..pfander import paths as _pfander_paths
from . import components as C
from . import effects as E
from . import offsets as OFF
from . import tier1

#: Where a model's Pfänder run is filed.  Taken from the study package rather
#: than hardcoded, so a configured data root reaches here too.
RUNS = _pfander_paths.RUNS

#: Outcomes whose observed between-message spread is entirely sampling noise, so
#: their apparent message ranking is variance with no covariance behind it.  This
#: needs no ground truth — the diagnostic is our own noise-corrected effect SD —
#: which is why it is safe to apply to a study with no human data.
NOISE_FLOOR_OUTCOMES = ("belief_post", "donation_ams")


@dataclass(frozen=True)
class Recipe:
    """One calibration, named and complete enough to reproduce."""

    name: str
    #: Which run's condition effects to keep — this is what sets the sort key.
    #: A tuple averages several runs' effect vectors, which is the single largest
    #: free gain measured: on Voelkel, averaging Qwen2.5-7B with Qwen2.5-72B takes
    #: pooled Pearson r from 0.408 to 0.460, better than any individual model and
    #: 90% of a fresh human replication's 0.514.  The first entry supplies the
    #: rows, so it decides the respondent pool and the demographic composition.
    effects_from: str | tuple[str, ...]
    #: Which run supplies control-arm levels, demographic offsets, residuals.
    level_from: str | None = None
    offsets_from: str | None = None
    residuals_from: str | None = None
    #: Global multiplicative shrinkage on the effects, or None to leave them.
    shrink: float | None = None
    #: Shrink the within-outcome deviations, re-weighting toward the profile.
    within_shrink: float | None = None
    #: Per-outcome mean effects to blend toward, and how far (0 = ours, 1 = theirs).
    profile_anchor: dict[str, float] | None = None
    profile_weight: float = 0.0
    #: External control-arm levels, overriding whatever ``level_from`` supplied.
    level_anchors: dict[str, float] | None = None
    #: Per-moderator factors on the demographic offsets.
    offset_scales: dict[str, float] | None = None
    #: Shrink outcomes whose effect spread is pure noise.
    flatten_noise: bool = False
    notes: str = ""
    seed: int = 20260823
    #: Runs actually needed, filled in by :meth:`sources`.
    _extra: tuple[str, ...] = field(default=(), repr=False)

    @property
    def effect_runs(self) -> tuple[str, ...]:
        """The runs whose effect vectors are averaged, in order."""
        if isinstance(self.effects_from, str):
            return (self.effects_from,)
        return tuple(self.effects_from)

    @property
    def template_run(self) -> str:
        """The run whose respondents the submission is made of."""
        return self.effect_runs[0]

    def sources(self) -> tuple[str, ...]:
        """Every run this recipe reads, deduplicated and in a stable order."""
        wanted = [
            *self.effect_runs,
            self.level_from,
            self.offsets_from,
            self.residuals_from,
        ]
        seen: list[str] = []
        for name in wanted:
            if name and name not in seen:
                seen.append(name)
        return tuple(seen)


def load_runs(
    names: tuple[str, ...], root: Path | str = RUNS
) -> dict[str, pd.DataFrame]:
    """Read each run's Tier-1 export, which is the frame a submission is made of."""
    root = Path(root)
    frames = {}
    for name in names:
        path = root / name / "tier1_submission.csv"
        if not path.exists():
            raise FileNotFoundError(f"no Tier-1 export for run {name!r} at {path}")
        frames[name] = pd.read_csv(path)
    return frames


def effect_table(frame: pd.DataFrame, instrument: tier1.Instrument) -> pd.DataFrame:
    """A run's own condition effects, in percentage points of scale range.

    Percentage points because that is the unit the benchmark scores in, so a
    calibration fitted on one study transfers to another without a conversion
    step where a factor of ten could hide.
    """
    rows = []
    for outcome in instrument.outcomes:
        if outcome not in frame.columns:
            continue
        scale = instrument.scales[outcome]
        for condition, value in C.condition_effects(
            frame, outcome, instrument.control
        ).items():
            rows.append(
                {
                    "outcome": outcome,
                    "condition": condition,
                    "estimate": float(value) * (100.0 / scale),
                    "se": 1.0,
                }
            )
    return pd.DataFrame(rows)


def swap_components(
    recipe: Recipe,
    runs: dict[str, pd.DataFrame],
    design: tier1.Instrument,
) -> pd.DataFrame:
    """Take each term from the run the recipe nominates, respecting the format.

    Two things stop this being a plain call to
    :func:`~silicon_sampling.calibration.components.hybrid`, and both silently
    corrupted a submission before they were handled.

    **A recipe that swaps nothing must recompose nothing.**  Recomposition draws
    fresh residuals, so running it on a single run still moves every respondent's
    value by roughly a residual standard deviation.  That is harmless for the
    condition means, which are put back exactly, but it means a no-calibration
    recipe was not returning its input — and the large per-respondent shift then
    clipped against the scale, putting 0.275 points of drift on the primary
    outcome for a recipe that was supposed to do nothing at all.

    **A composite must be rebuilt through its items, not beside them.**  The
    twelve trust items ride along in the submission and the format check compares
    them against the composite.  Recomposing the composite alone leaves the items
    behind, and correcting them afterwards is a shift of the full residual size
    that clips.  Rebuilding the *items* and taking their mean gives the composite
    for free and keeps every value near where it started.

    **A binary outcome cannot be recomposed additively at all** — doing so turned
    a 0/1 column into continuous values whose threshold collapsed the signup rate
    from 0.311 to 0.003.  Binary outcomes are left for
    :func:`~silicon_sampling.calibration.tier1.calibrate_binary`.
    """
    template = recipe.template_run
    swaps = {recipe.level_from, recipe.offsets_from, recipe.residuals_from} - {
        None,
        template,
    }
    if not swaps:
        return runs[template].copy()

    composite_items = {item for items in design.composites.values() for item in items}
    scales = {
        name: scale
        for name, scale in design.scales.items()
        if name not in design.binary and name not in design.composites
    }
    for item in sorted(composite_items):
        if item in runs[template].columns:
            scales[item] = 100.0

    frame, _ = C.hybrid(
        runs,
        scales,
        design.moderators,
        design.control,
        effects_from=template,
        level_from=recipe.level_from,
        offsets_from=recipe.offsets_from,
        residuals_from=recipe.residuals_from,
        seed=recipe.seed,
    )
    for composite, items in design.composites.items():
        present = [item for item in items if item in frame.columns]
        if composite in frame.columns and len(present) == len(items):
            frame[composite] = frame[present].mean(axis=1)
    for name in design.binary:
        if name in runs[template].columns:
            frame[name] = runs[template][name].to_numpy()
    return frame


def averaged_effects(
    recipe: Recipe,
    runs: dict[str, pd.DataFrame],
    design: tier1.Instrument,
) -> pd.DataFrame:
    """Mean effect vector across the runs a recipe nominates.

    Averaging is the largest free improvement measured on Voelkel, and it works
    because the samplers' errors are close to independent: the effect vectors of
    Qwen2.5-7B and Qwen2.5-72B correlate only +0.315 with each other, and
    Qwen2.5-7B with V4-Flash +0.091.  Averaging two such estimates of the same
    underlying signal raises the correlation with that signal, and it does:
    0.408 -> 0.460 for the Qwen pair, against 0.514 for a fresh human half sample.

    It is not monotone in the number of models.  Adding V4-Flash to the Qwen pair
    *lowers* pooled r to 0.440, because it is the weakest ranker (0.190 alone) —
    though it lowers RMSE further, which is the one place the three-way average
    wins outright.  So the membership is a decision, not a default.
    """
    tables = [effect_table(runs[name], design) for name in recipe.effect_runs]
    if len(tables) == 1:
        return tables[0]
    stacked = pd.concat(tables, ignore_index=True)
    return (
        stacked.groupby(["outcome", "condition"], as_index=False)
        .agg(estimate=("estimate", "mean"), se=("se", lambda s: float(s.mean())))
        .reset_index(drop=True)
    )


def apply(
    recipe: Recipe,
    runs: dict[str, pd.DataFrame] | None = None,
    instrument: tier1.Instrument | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Turn a recipe into a calibrated respondent-level frame, plus its audit.

    Order matters and is fixed here rather than left to the caller.  The component
    swap happens first, because the effect transforms are fitted in percentage
    points on whatever effects will actually be submitted; then the effect
    transforms; then the level and offset overrides, which are the terms the
    effect transforms do not touch.  Reversing the first two would fit a shrinkage
    factor to one model's effects and apply it to another's.

    Always check the returned audit.  A non-zero ``max_abs_effect_drift`` means a
    target was not reachable on that outcome's scale, and the two format-bound
    outcomes — the twelve-item trust composite and the binary newsletter — are
    where that shows up first.
    """
    design = instrument or tier1.pfander_instrument()
    runs = runs if runs is not None else load_runs(recipe.sources())
    frame = swap_components(recipe, runs, design)

    targets = (
        effect_table(frame, design)
        if len(recipe.effect_runs) == 1
        else averaged_effects(recipe, runs, design)
    )
    if recipe.profile_weight and recipe.profile_anchor:
        targets = E.substitute_profile(
            targets,
            pd.Series(recipe.profile_anchor),
            recipe.profile_weight,
            reference=design.control,
        )
    if recipe.within_shrink is not None:
        targets = E.shrink_within_outcome(
            targets, recipe.within_shrink, reference=design.control
        )
    if recipe.flatten_noise:
        targets = E.flatten_outcomes(
            targets, set(NOISE_FLOOR_OUTCOMES), factor=0.2, reference=design.control
        )
    if recipe.shrink is not None:
        targets = E.global_shrink(targets, recipe.shrink)

    offsets = None
    if recipe.offset_scales:
        offsets = {
            outcome: OFF.rescale_offsets(
                C.decompose(frame, outcome, design.moderators, design.control).offsets,
                recipe.offset_scales,
            )
            for outcome in design.outcomes
            if outcome not in design.binary and outcome in frame.columns
        }

    return tier1.calibrate(
        frame,
        targets=targets,
        levels=recipe.level_anchors,
        offsets=offsets,
        seed=recipe.seed,
        instrument=design,
    )


#: The effect ensemble that scored best on the leaderboard's sort key.
BEST_RANKERS = ("qwen25_7b", "qwen25_72b")

#: The run three independent sources agree is closest to real response levels.
GROUNDED = "v4_flash"

#: Mean absolute human intervention effect, in pp of scale range, per study.
#: Real effect magnitudes genuinely differ 4.5-fold between these studies, which
#: is why an *absolute* effect target does not transfer.
HUMAN_EFFECT_SCALE = {
    "Voelkel": 1.125,  # democratic norms, 2022
    "Goldwert": 2.967,  # climate advocacy, US megastudy
    "ICPC": 5.035,  # climate belief and policy, US subsample
}

#: The shrinkage factor applied to the averaged effect vector before submission.
#:
#: **Re-derived on the audited (`_v3`) samples; the earlier 0.2 was fitted partly
#: on broken ones.**  The ICPC and Goldwert questionnaires printed slider endpoint
#: labels without their 0-100 range, so those models answered on an implicit 0-10
#: scale and every effect in those two studies came out compressed roughly
#: five-fold.  The shrinkage fitted against that compression was correspondingly
#: too aggressive.
#:
#: Two different factors are defensible and they no longer agree.  Fitting through
#: the origin (Sigma hl / Sigma l^2) minimises RMSE; fitting with an intercept
#: (cov/var) is what drives the benchmark's beta to 1.  On Voelkel these matched to
#: 0.25% because its mean signed human effect is -0.06 pp.  On ICPC and Goldwert
#: the mean signed effect is +3.2 and +2.8 pp -- nearly every arm pushes the same
#: way -- so the through-origin fit absorbs that mean into its slope and the two
#: diverge two-fold (0.46 against 0.27 on ICPC, 0.60 against 0.28 on Goldwert).
#:
#: The choice between them is decided by how flat the RMSE curve is.  Out-of-fold
#: over the three studies, mean RMSE runs 3.099 / 3.056 / 3.028 / 3.018 / 3.024 at
#: k = 0.25 / 0.30 / 0.35 / 0.40 / 0.45 -- a 1.7% spread across the whole range --
#: while beta over the same range runs 1.026 / 0.855 / 0.733 / 0.641 / 0.570.  RMSE
#: barely distinguishes them and beta strongly does, so the factor is set where
#: beta calibrates, at a 2.7% RMSE cost against its own optimum of 0.32.
#:
#: Pearson r is unchanged at +0.398 by every value of k, as it must be: a positive
#: scalar cannot move a correlation.  Shrinkage is worth a large slice of RMSE and
#: all of beta, and provably nothing on the leaderboard's sort key.
GLOBAL_SHRINK = 0.25


def hybrid_default(
    effects_from: str | tuple[str, ...] = BEST_RANKERS,
    grounded: str = GROUNDED,
    shrink: float | None = GLOBAL_SHRINK,
) -> Recipe:
    """The component hybrid: the best rankers' averaged effects, one model's context.

    **``shrink`` defaults to 0.25, after two wrong turns worth recording.**  The
    factor was first taken from Voelkel alone (0.159), then dropped entirely on the
    grounds that real effect magnitudes differ 4.5-fold between studies (Voelkel
    1.125 pp, Goldwert 2.967, ICPC 5.035) so no single target could transfer.  That
    reasoning compared our Pfänder effects against human effects computed on *other
    studies' pair sets* — different instruments, different outcome mixes, different
    arm counts — which is not a comparison at all.

    Matched pair by pair within each study it comes out stable.  The RMSE-optimal
    factor is 0.159 / 0.125 / 0.112 for the three models on Voelkel, 0.216 on ICPC
    and 0.226 on Goldwert, and leave-one-study-out puts its out-of-fold estimates
    at 0.198-0.222 — a spread of 1.12x across three studies.  **The ratio transfers
    even though neither of its parts does**, which is what makes it usable: human
    effects run 1.1 to 5.0 pp and ours 2.8 to 5.5, and the quotient stays near 0.2.

    Note this is the *RMSE-optimal* factor, a regression through the origin, and it
    is much smaller than the factor that would match our effect spread to the
    humans' (0.40 to 0.92).  The two diverge exactly because the correlation is
    low: a predictor that barely correlates should barely predict, so minimising
    squared error shrinks far harder than matching variance.  Shrinkage is provably
    neutral on the leaderboard's sort key either way, so this choice is about RMSE
    and beta only.
    Caveat on the largest reference: ICPC's US control arm is small (n = 669) and
    its effects run hotter than the paper's global headline, so 5.035 is the least
    secure of the three. Excluding it still leaves a 2.6-fold range that spans 1.0.

    The default effect source is the Qwen pair rather than Qwen2.5-7B alone
    because averaging them measured better on every Section-1 metric (pooled r
    0.408 -> 0.460, Spearman 0.311 -> 0.363, directional 61.1 -> 63.0, RMSE
    3.620 -> 2.930).  None of those deltas clears its interval at six intervention
    clusters — nothing does in this study — but all four point the same way and r
    sits at p(higher) = 0.90.
    """
    runs = (effects_from,) if isinstance(effects_from, str) else tuple(effects_from)
    return Recipe(
        name=f"hybrid:{'+'.join(runs)}-effects+{grounded}-context",
        effects_from=runs,
        level_from=grounded,
        offsets_from=grounded,
        residuals_from=grounded,
        shrink=shrink,
        flatten_noise=True,
        notes=(
            "Averaged condition effects from the better rankers; levels, "
            "demographic offsets and residual structure from the model three "
            "independent sources agree is closer to real levels."
        ),
    )


def single_model(run: str, shrink: float | None = 0.159) -> Recipe:
    """One model, calibrated only on its effects — the minimal-assumption entry."""
    return Recipe(
        name=f"single:{run}",
        effects_from=run,
        shrink=shrink,
        flatten_noise=True,
        notes="No component swap; the honest baseline showing what calibration bought.",
    )


def uncalibrated(run: str) -> Recipe:
    """The raw sample, as insurance against every calibration being wrong."""
    return Recipe(name=f"raw:{run}", effects_from=run, notes="No calibration at all.")


def available_runs(root: Path | str = RUNS) -> tuple[str, ...]:
    """Which Pfänder runs have a Tier-1 export on disk right now."""
    root = Path(root)
    if not root.exists():
        return ()
    return tuple(
        sorted(
            path.name
            for path in root.iterdir()
            if (path / "tier1_submission.csv").exists()
        )
    )


def describe(recipe: Recipe) -> str:
    """A one-line human summary, for a report table or a metadata abstract."""
    parts = [f"effects={'+'.join(recipe.effect_runs)}"]
    for label, value in (
        ("levels", recipe.level_from),
        ("offsets", recipe.offsets_from),
        ("residuals", recipe.residuals_from),
    ):
        if value and value != recipe.template_run:
            parts.append(f"{label}={value}")
    if recipe.shrink is not None:
        parts.append(f"shrink={recipe.shrink:g}")
    if recipe.within_shrink is not None:
        parts.append(f"within={recipe.within_shrink:g}")
    if recipe.profile_weight:
        parts.append(f"profile_w={recipe.profile_weight:g}")
    if recipe.level_anchors:
        parts.append(f"anchored={len(recipe.level_anchors)}")
    if recipe.offset_scales:
        parts.append("offsets_rescaled")
    if recipe.flatten_noise:
        parts.append("flattened=" + ",".join(NOISE_FLOOR_OUTCOMES))
    return f"{recipe.name} [{', '.join(parts)}]"


def outcome_names() -> tuple[str, ...]:
    """The thirteen scored outcomes, in the submission's own order."""
    return tuple(pfander_outcomes.OUTCOMES)
