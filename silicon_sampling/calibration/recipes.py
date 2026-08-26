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
from .. import models as MODELS
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
    #: Multiplier on the residual term, correcting shared over-dispersion.
    #:
    #: These models answer with more within-cell spread than real participants:
    #: pooled over the sliders of all three reference studies the synthetic
    #: standard deviation runs about 1.07x the human one, and above it in seven of
    #: the nine (study, model) cells.  Scaling residuals by 0.90 raises the KDE
    #: overlap in **all nine**, by +0.006 to +0.056 and +0.026 on average.
    #:
    #: It touches only the within-cell spread.  Level, condition effects and
    #: demographic offsets are all separate terms of the decomposition and come
    #: through unchanged, so this cannot move the sort key or any effect metric --
    #: it reaches the variance ratio, OVL, KS and W1.
    residual_scale: float = 1.0
    #: Which run supplies the *party* offsets, when that is not ``offsets_from``.
    #:
    #: Party is the one moderator Pfander does not hand the model.  Gender, race
    #: and age are printed in the profile; party is **elicited** at Q16 on page 6,
    #: before almost every outcome, so the party structure in a Pfander sample is
    #: the model's own consistency rather than a demographic it was told to
    #: perform.  That makes it a different measurement from the one the reference
    #: studies grade, where party is given, and the two disagree about which model
    #: is best at it.
    #:
    #: Scored against external estimates of the real US party gap on Pfander's own
    #: outcomes, Qwen2.5-72B's elicited party structure is much the closest --
    #: r = +0.838 and RMSE 7.5 pp over eight outcomes, against DeepSeek-V4-Flash's
    #: +0.487 and 9.8, Qwen2.5-7B's -0.306 and 18.2, and 19.1 for submitting no
    #: party gap at all.  On the moderators that *are* given, V4-Flash remains the
    #: better donor in both reference studies (pooled offset r 0.190 on Voelkel and
    #: 0.177 on Goldwert), so the two are taken from different runs.
    party_offsets_from: str | None = None
    #: Externally estimated Democrat-minus-Republican gaps, in pp of scale range,
    #: and how far to move our own party offsets toward them (0 = ours, 1 = theirs).
    #:
    #: Partial rather than full substitution, because the estimates are good but
    #: not measurements of Pfander: they come from three public datasets, and two
    #: of the three contrast ideology rather than party.
    party_gap_anchors: dict[str, float] | None = None
    party_gap_weight: float = 0.0
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
            self.party_offsets_from,
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
    if not swaps and recipe.residual_scale == 1.0:
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
        residual_scale=recipe.residual_scale,
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

    **Runs are averaged within a model first, then across models**, so each model
    carries the same weight however many seeds of it happen to exist.  Weighting
    every run equally instead would quietly hand Qwen2.5-7B four sevenths of the
    vector as soon as it has one more replicate than Qwen2.5-72B, and the result
    that was actually validated is the balanced one — an equal blend of the two.
    Runs of the same model are recognised by sharing a Hugging Face id, which is
    what makes a seed replicate a replicate rather than a third opinion.
    """
    by_model: dict[str, list[pd.DataFrame]] = {}
    for name in recipe.effect_runs:
        key = MODELS.MODELS.get(name, name)
        by_model.setdefault(key, []).append(effect_table(runs[name], design))

    def mean_of(tables: list[pd.DataFrame]) -> pd.DataFrame:
        if len(tables) == 1:
            return tables[0]
        return (
            pd.concat(tables, ignore_index=True)
            .groupby(["outcome", "condition"], as_index=False)
            .agg(estimate=("estimate", "mean"), se=("se", lambda s: float(s.mean())))
            .reset_index(drop=True)
        )

    return mean_of([mean_of(tables) for tables in by_model.values()])


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
    if recipe.party_offsets_from and "party" in design.moderators:
        donor = runs.get(recipe.party_offsets_from)
        if donor is not None:
            if offsets is None:
                offsets = {
                    outcome: C.decompose(
                        frame, outcome, design.moderators, design.control
                    ).offsets
                    for outcome in design.outcomes
                    if outcome not in design.binary and outcome in frame.columns
                }
            for outcome, table in offsets.items():
                if outcome not in donor.columns:
                    continue
                theirs = C.decompose(
                    donor, outcome, design.moderators, design.control
                ).offsets
                if "party" in theirs:
                    table["party"] = theirs["party"]
    if recipe.party_gap_anchors and recipe.party_gap_weight:
        if offsets is None:
            offsets = {
                outcome: C.decompose(
                    frame, outcome, design.moderators, design.control
                ).offsets
                for outcome in design.outcomes
                if outcome not in design.binary and outcome in frame.columns
            }
        shares = (
            frame.loc[frame["condition"] == design.control, "party"].value_counts()
            if "party" in frame.columns
            else None
        )
        for outcome, table in offsets.items():
            target = recipe.party_gap_anchors.get(outcome)
            if target is None or "party" not in table:
                continue
            wanted = OFF.impose_gap(
                table["party"],
                target / 100.0 * design.scales[outcome],
                "Democrat",
                "Republican",
                shares,
            )
            table["party"] = (1.0 - recipe.party_gap_weight) * table[
                "party"
            ] + recipe.party_gap_weight * wanted

    return tier1.calibrate(
        frame,
        targets=targets,
        levels=recipe.level_anchors,
        offsets=offsets,
        seed=recipe.seed,
        instrument=design,
    )


#: The effect ensemble that scored best on the leaderboard's sort key.
#: The first entry supplies the rows, so it decides who the synthetic respondents
#: are.  It is the quota-demographics run, because a model left to invent its own
#: income and party produces 0.8% of respondents under $30,000 against a real
#: 13.5%, and 18 of them in the control arm against the benchmark's minimum group
#: size of 30 -- for the level that every income interaction is dummy-coded
#: against.  The quota run puts 264 there.
#:
#: Swapping the template costs nothing on the effects: over the 208 pairs the
#: quota run correlates 0.864 with the three elicited seeds, where those seeds
#: correlate 0.846 with each other, so it sits inside the seed-noise band rather
#: than outside it.  Control-arm levels move by half a point.
BEST_RANKERS = (
    "qwen25_7b_demo",
    "qwen25_7b",
    "qwen25_7b_seed2",
    "qwen25_7b_seed3",
    "qwen25_72b_demo",
    "qwen25_72b",
    "qwen25_72b_seed2",
    "qwen25_72b_seed3",
)

#: The run three independent sources agree is closest to real response levels.
GROUNDED = "v4_flash"

#: Which run supplies party offsets; see ``Recipe.party_offsets_from``.
#:
#: The quota-demographics Qwen2.5-72B run, which is the best of the ten measured
#: against the external party gaps: RMSE 7.36 pp and r +0.912, against 7.48 and
#: +0.838 for the same model on invented demographics, 8.82 and +0.838 for
#: DeepSeek-V4-Flash, and 19.09 for submitting no party gap at all.
#:
#: Handing the model its party turns out to *help* every model's gap structure,
#: which is the opposite of what the reference studies suggested would happen:
#: Qwen2.5-7B goes from r -0.306 to +0.559, V4-Flash from +0.487 to +0.838.
PARTY_DONOR = "qwen25_72b_demo"

#: Human within-arm dispersion on Pfander's own outcomes, from TISP.
#:
#: Standard deviations in points of the 0-100 scale, over TISP's US sample, for
#: the items the crosswalk grades ``near`` -- the same questions Pfander asks, so
#: this is the closest thing to Pfander's own human dispersion that exists.
#: ``policy_specific_mean`` is excluded: its TISP mapping is ``construct-only``.
HUMAN_DISPERSION = {
    "trust_multidimensional": 20.62,
    "trust_post": 27.99,
    "policy_role_mean": 26.02,
}

#: Multiplier on the residual term; see ``Recipe.residual_scale``.
#:
#: **Fitted end to end against Pfander's own human dispersion, which reverses an
#: earlier cross-study estimate.**  Measured on the raw samples of the three
#: reference studies, these models are over-dispersed by about 1.07x and scaling
#: residuals by 0.90 raises the KDE overlap in all nine (study, model) cells.  So
#: 0.90 shipped.
#:
#: On Pfander that is wrong twice over.  DeepSeek-V4-Flash's raw dispersion here
#: is already right -- sd ratio 1.014 against the three ``near``-grade TISP
#: anchors, not 1.07 -- because Pfander's headline outcomes are multi-item
#: composites whose averaging removes the very noise the single sliders carried.
#: And the reconstruction shrinks spread further on its own: at a residual scale
#: of 0.90 the built entry came out at 0.846 to 0.926 of the donor's spread, so
#: the submission sat **13% under**-dispersed against TISP.
#:
#: The factor is therefore fitted where it is applied -- on the finished entry,
#: against :data:`HUMAN_DISPERSION` -- rather than transferred from studies with a
#: different instrument.  Both numbers are real; the direct one wins because it
#: measures the target study's own questions.
#:
#: Fitting has to be iterative because the response is **sublinear**: raising the
#: factor from 0.90 to 1.035, a factor of 1.15, moved the entry's dispersion ratio
#: only from 0.870 to 0.962, a factor of 1.106, because residuals pushed outward
#: clip against the ends of the scale.  Three builds land it:
#:
#: =======  =============  ==========
#: factor   mean sd ratio  mean |err|
#: =======  =============  ==========
#: 0.90     0.870          0.130
#: 1.035    0.962          0.071
#: **1.12** **1.011**      **0.057**
#: =======  =============  ==========
#:
#: A single global factor cannot fit three outcomes exactly -- at 1.12 they sit at
#: 1.098, 1.004 and 0.931 -- and per-outcome factors are not worth three
#: parameters fitted on three anchors.
#:
#: Like every other scale factor here it cannot move the sort key.  It reaches the
#: variance ratio, OVL, KS and W1.
RESIDUAL_SCALE = 1.12

#: Externally estimated Democrat-minus-Republican gaps on Pfander's outcomes, in
#: pp of scale range.  Every number comes from public data and none from Pfander.
#:
#: The point of having these at all is that the real party gap is **strongly
#: topic-dependent** and the models apply a roughly uniform one.  Trust in
#: scientists is barely polarised -- TISP's twelve Besley items, the same battery
#: Pfander scores, put the US left-right gap at 4.0 pp -- while climate policy
#: priority is polarised enormously, at 28-52 pp in CCAM.  Qwen2.5-72B gives
#: 14.6 pp for the trust battery and 27.9 for policy: right about the second and
#: nearly four times over on the first.
#:
#: Grades, worst first, because they set the blend weight rather than being hidden:
#:
#: * ``trust_multidimensional``, ``trust_post``, ``policy_role_mean`` -- TISP,
#:   ``near`` grade, 12 / 2 / 4 items.  The strongest evidence here: same battery,
#:   same population, item-for-item.
#: * ``policy_specific_mean`` -- TISP, 5 ``construct-only`` climate-policy items.
#: * ``belief_post`` -- the ICPC tournament's US control arm, an online
#:   experimental sample much like Pfander's, split left against right.
#: * ``concern_mean``, ``policy_general`` -- CCAM's ``worry`` and ``priority``
#:   items, 44.5 and 52.0 pp, shrunk by 0.6.  CCAM is a nationally representative
#:   panel and its gaps run larger than experimental samples': where the two can
#:   be compared, ICPC's belief gap is 0.71 of CCAM's worry gap and Goldwert's
#:   behaviour gap 0.45 of CCAM's discussion gap.
#: * ``behavior_mean`` -- ICPC's sharing gap (7.3) with Goldwert's advocacy gap
#:   (14.0), which bracket it.
#:
#: Two of the three sources contrast **ideology** rather than party, which is why
#: these are blended toward rather than substituted for; see
#: :data:`PARTY_GAP_WEIGHT`.
PARTY_GAP_ANCHORS = {
    "trust_multidimensional": 4.0,
    "trust_post": 11.3,
    "policy_role_mean": 7.4,
    "policy_specific_mean": 13.7,
    "belief_post": 31.7,
    "concern_mean": 26.7,
    "policy_general": 26.7,
    "behavior_mean": 10.0,
}

#: How far to move party offsets toward :data:`PARTY_GAP_ANCHORS`.
#:
#: Half, deliberately.  Full substitution would claim these estimates *are*
#: Pfander's party gaps, which they are not -- different instruments, and two of
#: three contrasting ideology rather than party.  Zero would keep a 14.6 pp party
#: gap on a trust battery that twelve matched items say is 4.0.  Half moves the
#: measured distance to the anchors from RMSE 7.4 pp to about 3.7 while leaving
#: the model's own topic ordering, which is already right (r = +0.835), intact.
PARTY_GAP_WEIGHT = 0.5

#: Mean absolute human intervention effect, in pp of scale range, per study.
#: Real effect magnitudes genuinely differ 4.5-fold between these studies, which
#: is why an *absolute* effect target does not transfer.
HUMAN_EFFECT_SCALE = {
    "Voelkel": 1.125,  # democratic norms, 2022
    "Goldwert": 2.967,  # climate advocacy, US megastudy
    "ICPC": 5.035,  # climate belief and policy, US subsample
}

#: How far to shrink each arm's effect toward its own outcome's mean effect.
#:
#: Leave-one-study-out adopts this at 0.5 for Qwen2.5-7B (3/3 folds) and holds it
#: in the same direction for the others.  On the averaged effect vector it is a
#: clear out-of-fold gain: per-study mean pearson r 0.398 -> 0.426, pooled r
#: 0.446 -> 0.470, and RMSE improves at the same time.
#:
#: The curve is broad rather than peaked -- per-study r is 0.428 / 0.429 / 0.426
#: at 0.3 / 0.4 / 0.5 and pooled r peaks at 0.5 -- so the pre-committed default of
#: 0.5 is kept instead of fitting a magnitude inside the flat region.
WITHIN_SHRINK = 0.5

#: The global multiplicative shrinkage applied *after* :data:`WITHIN_SHRINK`.
#:
#: **The two shrinkages interact, and the constant is only meaningful as a pair
#: with the within factor.**  Fitted out-of-fold across the three studies, the
#: best global factor falls almost linearly as the within factor rises:
#:
#: ===========  =====  =====  =====  =====  =====
#: within        0.2    0.4    0.5    0.8    1.0
#: global k      0.475  0.425  0.375  0.300  0.250
#: nRMSE         0.987  0.997  1.015  1.052  1.076
#: beta          0.992  0.976  1.018  0.994  1.026
#: ===========  =====  =====  =====  =====  =====
#:
#: So 0.25 is right only with no within-shrinkage, and pairing it with 0.5 --
#: which an earlier revision did -- overshoots to beta 1.53.  The shipped pair is
#: (within 0.5, global 0.375).
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
#: diverge two-fold.  The pairing above is chosen on nRMSE + |beta - 1| jointly,
#: which is flat to within 4% across the whole grid.
#:
#: Pearson r is unchanged by every value of the global factor, as it must be: a
#: positive scalar cannot move a correlation.  Global shrinkage is worth a large
#: slice of RMSE and all of beta, and provably nothing on the sort key.
GLOBAL_SHRINK = 0.375

#: Signal and noise variance of one run's Pfander effect vector, per model.
#:
#: Measured over the 208 pairs.  The three Qwen2.5-7B seeds correlate 0.870 /
#: 0.836 / 0.832 with each other, so a single 7B run's reliability is 0.846 and
#: 15% of its effect variance is nothing but which respondents it drew; the
#: Qwen2.5-72B seeds give 0.745.  Splitting each model's total variance on those
#: reliabilities gives the components below.
EFFECT_VARIANCE = {
    "Qwen/Qwen2.5-7B": {"signal": 5.859, "noise": 1.069},
    "Qwen/Qwen2.5-72B": {"signal": 4.528, "noise": 1.550},
}

#: Covariance of two models' *true* effect vectors, keyed by the unordered pair.
#: 3.606 corresponds to a true cross-model correlation of +0.700 -- against +0.713
#: obtained independently by disattenuating the observed cross-correlation.
EFFECT_COVARIANCE = {
    frozenset({"Qwen/Qwen2.5-7B", "Qwen/Qwen2.5-72B"}): 3.606,
}


def ensemble_reliability(runs: tuple[str, ...]) -> float | None:
    """How much of the averaged effect vector is signal rather than sampling noise.

    Computed from variance components rather than looked up, because the answer
    depends on *how* the runs divide between models and not merely how many there
    are: four 7B seeds with two 72B seeds is not the same ensemble as three of
    each.  Averaging is model-balanced (see :func:`averaged_effects`), so each of
    the ``M`` models carries weight ``1/M``; its signal survives that averaging
    and its noise is divided by however many seeds of it there are.

    **The model checks out against a direct measurement.**  It predicts 0.870 for
    a one-run-each ensemble; correlating the shipped 7B+72B ensemble against the
    one built from both models' second seeds gives 0.866, a 0.5% disagreement.

    Returns ``None`` when any run's model has no measured components, since a
    guessed reliability would silently rescale the shrinkage.
    """
    counts: dict[str, int] = {}
    for run in runs:
        counts[MODELS.MODELS.get(run, run)] = (
            counts.get(MODELS.MODELS.get(run, run), 0) + 1
        )
    if not counts or any(model not in EFFECT_VARIANCE for model in counts):
        return None
    weight = 1.0 / len(counts)
    signal = 0.0
    for a in counts:
        for b in counts:
            if a == b:
                signal += weight * weight * EFFECT_VARIANCE[a]["signal"]
            else:
                signal += (
                    weight * weight * EFFECT_COVARIANCE.get(frozenset({a, b}), 0.0)
                )
    noise = sum(
        weight * weight * EFFECT_VARIANCE[model]["noise"] / n
        for model, n in counts.items()
    )
    return signal / (signal + noise) if signal + noise > 0 else None


#: The reliability :data:`GLOBAL_SHRINK` was fitted against: one run per model,
#: which is all the calibration studies have.
SHRINK_FITTED_RELIABILITY = 0.870


def shrink_for_runs(
    runs: tuple[str, ...] | int, base: float | None = None
) -> float | None:
    """:data:`GLOBAL_SHRINK`, adjusted for how noisy an ensemble it is applied to.

    The shrinkage factor is ``cov(h, l) / var(l)``.  Averaging more independent
    runs strips sampling noise out of ``var(l)`` without touching ``cov(h, l)``,
    because that noise is uncorrelated with the human effects -- so the optimal
    factor rises in exact proportion to the reliability.

    This matters because the factor is fitted on the calibration studies, where
    only one run per model exists, and then applied to Pfander, where several do.
    Carrying the one-run-each factor onto a six-run ensemble would over-shrink by
    about 9%.  It cannot touch the sort key either way; a positive scalar cannot
    move a correlation.

    Takes the run names rather than a count, because the reliability depends on
    how they divide between models.  An ``int`` is accepted and ignored, for
    callers that only know how many there are.
    """
    base = GLOBAL_SHRINK if base is None else base
    if base is None or isinstance(runs, int):
        return base
    here = ensemble_reliability(tuple(runs))
    if here is None:  # a model nobody measured; leave the factor alone
        return base
    return base * here / SHRINK_FITTED_RELIABILITY


def hybrid_default(
    effects_from: str | tuple[str, ...] = BEST_RANKERS,
    grounded: str = GROUNDED,
    shrink: float | None = GLOBAL_SHRINK,
    within_shrink: float | None = WITHIN_SHRINK,
    party_offsets_from: str | None = PARTY_DONOR,
    residual_scale: float = RESIDUAL_SCALE,
) -> Recipe:
    """The component hybrid: the best rankers' averaged effects, one model's context.

    **``shrink`` defaults to 0.375, after two wrong turns worth recording.**  The
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
        within_shrink=within_shrink,
        party_offsets_from=party_offsets_from,
        residual_scale=residual_scale,
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
    if recipe.party_offsets_from:
        parts.append(f"party-offsets={recipe.party_offsets_from}")
    if recipe.residual_scale != 1.0:
        parts.append(f"residual-scale={recipe.residual_scale:g}")
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
