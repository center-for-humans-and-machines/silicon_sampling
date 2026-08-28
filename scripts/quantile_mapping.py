"""Does mapping our answers onto a *borrowed* human distribution actually help?

Every calibration in this project so far moves one or two moments: a level
anchor moves the mean, a residual scale moves the standard deviation.  The
benchmark also scores ``ovl``, ``ks`` and ``w1``, which are properties of the
whole distribution — and nothing shipped touches those except as a side effect.

Quantile mapping does touch them directly.  Sort our control-arm answers, sort a
human reference, and replace our k-th percentile with the human k-th percentile.
On the reference itself this is exact by construction, so measuring it there
proves nothing at all.  The question that matters is whether the map still helps
when the reference comes from **a different study** — which is the only situation
Pfander is ever in, since Pfander publishes no responses and its five anchorable
outcomes borrow their distributions from CCC.

So the test is cross-study.  ICPC and CCC both measure climate belief and climate
policy support on the same 0-100 range, on separate US samples:

    ICPC belief  n=8,226  mean 70.6  sd 29.7        CCC Belief_Post    n=12,416  mean 66.5  sd 22.6
    ICPC policy  n=8,156  mean 65.5  sd 24.1        CCC Policies_Post  n=12,429  mean 68.7  sd 29.1

For each direction, the map is fitted against the *other* study's humans and
graded against the target study's own humans, which the map never saw.  Four
variants are compared:

``raw``            our answers, untouched
``moment``         shift and scale to the borrowed mean and sd — the shipped kind
``quantile``       full distributional map onto the borrowed distribution
``quantile (own)`` the same map fitted on the target's own humans

The last is not a candidate.  It is the ceiling — what the map would score if the
borrowed distribution were perfect — and the gap between it and ``quantile`` is
the price of borrowing.  If ``quantile`` does not beat ``moment``, the extra
machinery is not worth shipping.

Run: ``python scripts/quantile_mapping.py``
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from silicon_sampling import models as MODELS
from silicon_sampling.benchmark import distributions as DIST

warnings.filterwarnings("ignore")

RUNS = ("qwen25_7b_v3", "qwen25_72b_v3", "muse_glimmer_30b")

#: (study, outcome) cells that measure the same construct on the same range.
#:
#: Matched on construct and scale, not on wording: ICPC's belief composite and
#: CCC's three-item belief battery are different questionnaires asking whether
#: climate change is real and human-caused, both on a 0-100 slider.  That is the
#: same relationship Pfander's anchored outcomes have to CCC's, which is the
#: point — a test on verbatim-identical items would flatter the method relative
#: to how it would actually be used.
PAIRS = (
    (("ICPC", "belief"), ("CCC", "Belief_Post")),
    (("ICPC", "policy"), ("CCC", "Policies_Post")),
)

STUDIES = {
    "ICPC": ("silicon_sampling.icpc.score", "silicon_sampling.icpc.paths"),
    "CCC": ("silicon_sampling.ccc.score", "silicon_sampling.ccc.paths"),
}


def control_values(
    study: str, outcome: str
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Human and per-run control-arm answers for one outcome."""
    score = __import__(STUDIES[study][0], fromlist=["x"])
    paths = __import__(STUDIES[study][1], fromlist=["x"])
    humans = score.load_humans()
    column = next(c for c in ("condName", "condition", "cond") if c in humans.columns)
    mask = humans[column].astype(str).str.contains("ontrol", case=False, na=False)
    human = pd.to_numeric(humans[mask][outcome], errors="coerce").dropna().to_numpy()
    runs = {}
    for run in RUNS:
        key = MODELS.resolve_run(paths.samples_dir, run)
        if key is None:
            continue
        sample = pd.read_csv(paths.samples_dir(key) / "samples.csv", low_memory=False)
        if hasattr(score, "pool_controls"):
            sample = score.pool_controls(sample)
        synth = sample[
            sample["condition"].astype(str).str.contains("ontrol", case=False, na=False)
        ]
        if outcome not in synth.columns:
            continue
        runs[run] = pd.to_numeric(synth[outcome], errors="coerce").dropna().to_numpy()
    return human, runs


def quantile_map(ours: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Replace each of our values with the reference value at the same rank.

    Ranks are taken as mid-ranks so that ties — and these scales are full of them,
    every round number being a spike — map to one reference value rather than
    fanning out across an arbitrary stretch of it.
    """
    ranks = pd.Series(ours).rank(method="average").to_numpy()
    probabilities = (ranks - 0.5) / len(ours)
    return np.quantile(reference, probabilities)


def moment_map(
    ours: np.ndarray, reference: np.ndarray, lo: float, hi: float
) -> np.ndarray:
    """Shift and scale to the reference's mean and sd — the shipped kind."""
    if ours.std(ddof=1) == 0:
        return ours
    scaled = (ours - ours.mean()) / ours.std(ddof=1) * reference.std(
        ddof=1
    ) + reference.mean()
    return np.clip(scaled, lo, hi)


def grade(truth: np.ndarray, ours: np.ndarray, lo: float, hi: float) -> dict:
    shape = DIST.compare_distributions(truth, ours, lo=lo, hi=hi)
    return {
        "level err (pp)": abs(float(ours.mean()) - float(truth.mean()))
        / (hi - lo)
        * 100,
        "sd ratio": float(np.std(ours, ddof=1) / np.std(truth, ddof=1)),
        "ovl": shape["ovl"],
        "ks": shape["ks"],
        "w1": shape["w1"],
    }


def main() -> int:
    pd.set_option("display.width", 250)
    lo, hi = 0.0, 100.0
    rows = []
    for left, right in PAIRS:
        for target, source in ((left, right), (right, left)):
            truth, runs = control_values(*target)
            borrowed, _ = control_values(*source)
            for run, ours in runs.items():
                label = {
                    "pair": f"{target[0]}:{target[1]} <- {source[0]}:{source[1]}",
                    "run": run.replace("_v3", ""),
                }
                for variant, values in (
                    ("raw", ours),
                    ("moment", moment_map(ours, borrowed, lo, hi)),
                    ("quantile", quantile_map(ours, borrowed)),
                    ("quantile (own)", quantile_map(ours, truth)),
                ):
                    rows.append(
                        {**label, "variant": variant, **grade(truth, values, lo, hi)}
                    )

    table = pd.DataFrame(rows)
    metrics = ["level err (pp)", "sd ratio", "ovl", "ks", "w1"]

    print("=== mean over the four cross-study cells x three models ===\n")
    summary = (
        table.groupby("variant")[metrics]
        .mean()
        .reindex(["raw", "moment", "quantile", "quantile (own)"])
    )
    print(summary.to_string(float_format=lambda v: f"{v:8.3f}"))

    print("\n\n=== does quantile beat moment, cell by cell? ===\n")
    print("    ovl is higher-is-better; ks and w1 are lower-is-better\n")
    wide = table.pivot_table(index=["pair", "run"], columns="variant", values="ovl")
    wide = wide[["raw", "moment", "quantile", "quantile (own)"]]
    wide["quantile - moment"] = wide["quantile"] - wide["moment"]
    print(wide.to_string(float_format=lambda v: f"{v:8.3f}"))
    beat = (wide["quantile - moment"] > 0).sum()
    print(f"\n    quantile beat moment on ovl in {beat}/{len(wide)} cells")

    print("\n\n=== how much of the ceiling survives borrowing? ===\n")
    for metric in ("ovl", "ks", "w1"):
        piv = table.pivot_table(index=["pair", "run"], columns="variant", values=metric)
        gain = (piv["quantile"] - piv["raw"]).mean()
        ceiling = (piv["quantile (own)"] - piv["raw"]).mean()
        share = gain / ceiling * 100 if ceiling else float("nan")
        print(
            f"  {metric:4s}  raw {piv['raw'].mean():7.3f}"
            f" -> borrowed {piv['quantile'].mean():7.3f}"
            f" -> own {piv['quantile (own)'].mean():7.3f}"
            f"   ({share:5.1f}% of the available gain)"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
