"""Measure each model's effect signal and noise from its own seed replicates.

:data:`silicon_sampling.calibration.recipes.EFFECT_VARIANCE` and
:data:`~silicon_sampling.calibration.recipes.EFFECT_COVARIANCE` were derived by
hand and written down as constants. That was fine while there were two models and
the derivation was a paragraph in a docstring. It stops being fine the moment a
third model wants in, because :func:`~silicon_sampling.calibration.recipes.\
ensemble_reliability` returns ``None`` for any model without measured components
— deliberately, since a guessed reliability silently rescales the shrinkage — and
so the membership question cannot even be asked without this measurement.

The method is the one the docstrings describe, made executable:

1. Build each run's Pfänder effect vector over the common set of arm × outcome
   pairs.
2. A model's **reliability** is the mean correlation between its own *generation
   replicates* — runs differing in nothing but the per-respondent RNG seed. Two
   kinds of run are excluded and both matter:

   * a run sharing another's profile-seed set, because that is one draw sampled
     twice rather than two draws (``qwen25_72b_seed3``);
   * a ``_demo`` run, because it differs from its parent in the *prompt* as well
     — its profiles carry given demographics rather than elicited ones — and on
     top of that it reuses the parent's seed column, so part of its draw noise is
     shared. Both effects push its correlation with the parent up: 0.856 for
     Qwen2.5-72B against 0.745 for a genuine replicate. Counting it as a seed
     inflates the reliability and therefore understates the noise.
3. Split the model's total effect variance on that reliability — ``signal =
   rel * total``, ``noise = (1 - rel) * total``.
4. For each pair of models, average the observed correlation over **every**
   cross-model pairing of their replicates, disattenuate it by
   ``sqrt(rel_a * rel_b)`` to get the correlation of their *true* effect vectors,
   and turn that into a covariance with the two signal variances. Averaging
   matters: a single pairing of Qwen2.5-7B against Qwen2.5-72B ranges from 0.492
   to 0.620 across the six available, and picking one gives a covariance of 4.025
   where the mean gives 3.604.

Running it with only Qwen present must reproduce the shipped constants, and the
script says so rather than leaving the reader to check. Anything else means the
derivation in the docstring is not the derivation that produced the numbers.

Run: ``python scripts/variance_components.py``
"""

from __future__ import annotations

import itertools
import warnings

import numpy as np
import pandas as pd

from silicon_sampling import models as MODELS
from silicon_sampling.calibration import recipes as R
from silicon_sampling.calibration import tier1 as T1
from silicon_sampling.pfander import paths as PATHS

warnings.filterwarnings("ignore")

#: Every Pfänder run that could be a seed of some model.
CANDIDATES = (
    "qwen25_7b",
    "qwen25_7b_seed2",
    "qwen25_7b_seed3",
    "qwen25_7b_demo",
    "qwen25_72b",
    "qwen25_72b_seed2",
    "qwen25_72b_seed3",
    "qwen25_72b_demo",
    "v4_flash",
    "v4_flash_demo",
    "muse_glimmer_30b",
    "muse_glimmer_30b_seed2",
    "muse_glimmer_30b_seed3",
)

#: Above this share of identical scored cells, two runs are one draw.
DUPLICATE_AT = 0.40


def load(run: str) -> pd.DataFrame | None:
    path = PATHS.samples_dir(run) / "samples.csv"
    if not path.exists():
        return None
    return pd.read_csv(path, low_memory=False).sort_values("profile_id")


def effect_vector(frame: pd.DataFrame, design) -> pd.Series:
    """Arm effects in pp of each outcome's range, indexed by (outcome, arm)."""
    out = {}
    for outcome, span in design.scales.items():
        if outcome not in frame.columns:
            continue
        means = frame.groupby("condition")[outcome].apply(
            lambda s: pd.to_numeric(s, errors="coerce").mean()
        )
        base = means.get(design.control)
        if base is None or np.isnan(base):
            continue
        for arm, value in means.items():
            if arm != design.control:
                out[(outcome, arm)] = (value - base) / span * 100
    return pd.Series(out)


def identical_share(a: pd.DataFrame, b: pd.DataFrame, design) -> float:
    same = total = 0
    for outcome in design.scales:
        if outcome in a.columns and outcome in b.columns:
            x = pd.to_numeric(a[outcome].to_numpy(), errors="coerce")
            y = pd.to_numeric(b[outcome].to_numpy(), errors="coerce")
            same += int(((x == y) | (np.isnan(x) & np.isnan(y))).sum())
            total += len(x)
    return same / total if total else 0.0


def is_replicate(run: str) -> bool:
    """Whether *run* differs from its model's parent only in the RNG draw.

    ``_demo`` runs do not: they change the prompt, handing the model its
    demographics instead of asking for them.  That is a different treatment, not
    another draw of the same one.
    """
    return not run.endswith("_demo")


def independent_seeds(runs: list[str], frames: dict, design) -> list[str]:
    """Generation replicates only, with same-draw duplicates removed."""
    kept: list[str] = []
    for run in runs:
        if not is_replicate(run):
            continue
        if all(
            identical_share(frames[run], frames[other], design) <= DUPLICATE_AT
            for other in kept
        ):
            kept.append(run)
    return kept


def main() -> int:
    pd.set_option("display.width", 220)
    design = T1.pfander_instrument()

    frames, vectors = {}, {}
    for run in CANDIDATES:
        frame = load(run)
        if frame is None:
            continue
        frames[run] = frame
        vectors[run] = effect_vector(frame, design)
    index = None
    for v in vectors.values():
        index = v.index if index is None else index.intersection(v.index)
    vectors = {k: v.loc[index] for k, v in vectors.items()}
    print(f"{len(vectors)} runs on disk, {len(index)} common arm x outcome pairs\n")

    by_model: dict[str, list[str]] = {}
    for run in vectors:
        by_model.setdefault(MODELS.MODELS[run], []).append(run)

    print("=== per model ===\n")
    components, reliabilities, seeds_by_model = {}, {}, {}
    for model, runs in by_model.items():
        seeds = independent_seeds(runs, frames, design)
        dropped = [r for r in runs if r not in seeds]
        if len(seeds) < 2:
            print(f"  {model:32s} only {len(seeds)} independent draw -- no reliability")
            continue
        pairs = [
            float(np.corrcoef(vectors[a], vectors[b])[0, 1])
            for a, b in itertools.combinations(seeds, 2)
        ]
        rel = float(np.mean(pairs))
        total = float(np.mean([vectors[r].var(ddof=1) for r in seeds]))
        components[model] = {"signal": rel * total, "noise": (1 - rel) * total}
        reliabilities[model] = rel
        seeds_by_model[model] = seeds
        note = f"  (excluded: {', '.join(dropped)})" if dropped else ""
        print(f"  {model}")
        print(f"    independent seeds : {', '.join(seeds)}{note}")
        print(
            f"    pairwise r        : "
            f"{', '.join(f'{p:.3f}' for p in pairs)}   mean {rel:.3f}"
        )
        print(
            f"    total variance    : {total:.3f}"
            f"  ->  signal {components[model]['signal']:.3f}"
            f"  noise {components[model]['noise']:.3f}\n"
        )

    print("=== cross-model, disattenuated ===\n")
    covariances = {}
    for a, b in itertools.combinations(sorted(components), 2):
        # Every replicate of a against every replicate of b.  One pairing is a
        # noisy estimate of the same quantity; the six available for the Qwen
        # pair span 0.492 to 0.620.
        observed = float(
            np.mean(
                [
                    np.corrcoef(vectors[x], vectors[y])[0, 1]
                    for x in seeds_by_model[a]
                    for y in seeds_by_model[b]
                ]
            )
        )
        true = observed / np.sqrt(reliabilities[a] * reliabilities[b])
        cov = true * np.sqrt(components[a]["signal"] * components[b]["signal"])
        covariances[frozenset({a, b})] = cov
        print(
            f"  {a.split('/')[-1]:18s} x {b.split('/')[-1]:18s} "
            f"observed {observed:6.3f}  true {true:6.3f}  covariance {cov:7.3f}"
        )

    print("\n\n=== does this reproduce the shipped constants? ===\n")
    ok = True
    for model, want in R.EFFECT_VARIANCE.items():
        got = components.get(model)
        if got is None:
            print(f"  {model:32s} MISSING")
            ok = False
            continue
        for part in ("signal", "noise"):
            close = abs(got[part] - want[part]) < 0.15
            ok &= close
            print(
                f"  {model:32s} {part:6s} shipped {want[part]:7.3f}  "
                f"measured {got[part]:7.3f}  {'ok' if close else 'DIFFERS'}"
            )
    for key, want in R.EFFECT_COVARIANCE.items():
        got = covariances.get(key)
        label = " x ".join(sorted(m.split("/")[-1] for m in key))
        if got is None:
            print(f"  {label:32s} MISSING")
            ok = False
            continue
        close = abs(got - want) < 0.15
        ok &= close
        print(
            f"  {label:32s} cov    shipped {want:7.3f}  "
            f"measured {got:7.3f}  {'ok' if close else 'DIFFERS'}"
        )
    print(f"\n  {'reproduced' if ok else 'DOES NOT REPRODUCE -- do not use'}")

    print("\n\n=== what a Muse-inclusive ensemble would look like ===\n")
    muse = "meta-models/Muse-Glimmer-30B"
    if muse not in components:
        print("  Muse-Glimmer has fewer than two independent Pfänder draws.")
        print("  Its seeds are still sampling; nothing can be decided yet.")
        return 0
    print("  paste into recipes.EFFECT_VARIANCE / EFFECT_COVARIANCE:\n")
    print(
        f'    "{muse}": {{"signal": {components[muse]["signal"]:.3f}, '
        f'"noise": {components[muse]["noise"]:.3f}}},'
    )
    for key, cov in covariances.items():
        if muse in key:
            other = next(iter(key - {muse}))
            print(f'    frozenset({{"{muse}", "{other}"}}): {cov:.3f},')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
