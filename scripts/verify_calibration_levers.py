"""Measure what each candidate calibration does to the benchmark's own metrics.

Every number in the planning document `docs/plans/2026-08-22-macro-pfander-optimization.md`
comes from here.  The point of the script is that three claims which decide the
whole approach are cheap to check against real data, and two of them turned out
to be wrong in the recon that proposed them:

1. **How much of the human effect vector is between-outcome?**  If most of the
   variance the leaderboard's Pearson r is computed over lives in *which outcome
   moves*, rather than *which message moves it*, then the project's job is to get
   thirteen numbers right, not to rank sixteen messages.

2. **Which calibrations can move Pearson r at all?**  A single global rescale
   provably cannot -- and neither can it touch spearman, directional % or
   ``pearson_adj``, because that one is corrected with the *reference's* standard
   errors, not ours.  A *per-outcome* re-profiling is not a global rescale and
   moves r a great deal.  Conflating the two is the easiest mistake here.

3. **Does averaging two models' effect vectors beat the better model?**  Free if
   true, and the paired cluster bootstrap is the only honest way to ask, since
   both models answered the same instrument about the same interventions.

Run: ``python scripts/verify_calibration_levers.py``
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from silicon_sampling.benchmark import metrics as M
from silicon_sampling.benchmark.reference import ate_pairs, half_split
from silicon_sampling.voelkel import paths as P
from silicon_sampling.voelkel import score as S

RUNS = ("qwen25_7b", "v4_flash")
KEY = ["outcome", "condition"]


def load_pairs() -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """One ATE-pair frame per model, plus the human replication's, against Human 1."""
    humans = S.load_humans()
    human1, human2 = half_split(humans)
    reference = S.effects(human1)
    pairs = {}
    for run in RUNS:
        path = P.samples_dir(run) / "samples.csv"
        if path.exists():
            sample = pd.read_csv(path, low_memory=False)
            pairs[run] = ate_pairs(reference, S.effects(sample)).dropna(
                subset=["estimate_h", "estimate_l"]
            )
    return pairs, ate_pairs(reference, S.effects(human2))


def score(pairs: pd.DataFrame) -> dict:
    """The leaderboard's numbers for one submission."""
    pooled = M.pooled_metrics(pairs)
    return pooled | M.run_calibration_pooled(pairs)


def _row(label: str, pairs: pd.DataFrame) -> str:
    m = score(pairs)
    return (
        f"{label:34s} n={m['n_pairs']:3d} dir={m['directional_pct']:6.2f} "
        f"rho={m['spearman_rho']:+.3f} r={m['pearson_r']:+.3f} "
        f"r_within={m['pearson_within']:+.3f} r_adj={m['pearson_adj']:+.3f} "
        f"rmse={m['rmse']:.3f} beta={m['beta']:+.3f} beta_adj={m['beta_adj']:+.3f} "
        f"alpha={m['alpha']:+.3f}"
    )


def reprofile(pairs: pd.DataFrame, weight: float, anchor=None) -> pd.DataFrame:
    """Blend our per-outcome mean effect toward an external one, keeping deviations.

    ``weight`` 0 leaves the submission untouched; 1 replaces our whole per-outcome
    profile with the anchor's.  Our within-outcome deviations -- the message
    ranking -- survive either way, which is what makes this separable from the
    shrinkage in :func:`shrink_within`.
    """
    out = pairs.copy()
    ours = out.groupby("outcome")["estimate_l"].transform("mean")
    theirs = (
        out.groupby("outcome")["estimate_h"].transform("mean")
        if anchor is None
        else out["outcome"].map(anchor)
    )
    out["estimate_l"] = (weight * theirs + (1 - weight) * ours) + (
        out["estimate_l"] - ours
    )
    return out


def shrink_within(pairs: pd.DataFrame, factor: float) -> pd.DataFrame:
    """Shrink the within-outcome deviations, leaving the per-outcome profile alone.

    This re-weights the pooled correlation toward the component we predict better,
    which is why it raises r even though a *global* shrink cannot.
    """
    out = pairs.copy()
    means = out.groupby("outcome")["estimate_l"].transform("mean")
    out["estimate_l"] = means + factor * (out["estimate_l"] - means)
    out["se_l"] = out["se_l"] * factor
    return out


def shrink_global(pairs: pd.DataFrame, factor: float) -> pd.DataFrame:
    """Scale every predicted effect.  Included to demonstrate what it cannot do."""
    out = pairs.copy()
    out["estimate_l"] = out["estimate_l"] * factor
    out["se_l"] = out["se_l"] * factor
    return out


def average(pairs: dict[str, pd.DataFrame]) -> pd.DataFrame | None:
    """Precision-naive mean of two models' effect vectors on their shared grid."""
    if len(pairs) != 2:
        return None
    left, right = (pairs[name] for name in pairs)
    merged = left.merge(
        right[KEY + ["estimate_l", "se_l"]], on=KEY, suffixes=("_a", "_b")
    )
    out = merged[KEY + ["estimate_h", "se_h"]].copy()
    out["estimate_l"] = 0.5 * (merged["estimate_l_a"] + merged["estimate_l_b"])
    out["se_l"] = np.sqrt(0.25 * (merged["se_l_a"] ** 2 + merged["se_l_b"] ** 2))
    out.attrs["cross_model_r"] = M.pearson(
        merged["estimate_l_a"], merged["estimate_l_b"]
    )
    return out


def decompose(pairs: pd.DataFrame) -> None:
    """How much of the human effect vector is between-outcome, and what that buys."""
    means = pairs.groupby("outcome")["estimate_h"].transform("mean")
    total = pairs["estimate_h"].var(ddof=0)
    print(
        f"  pairs={len(pairs)} outcomes={pairs['outcome'].nunique()} "
        f"conditions={pairs['condition'].nunique()}"
    )
    print(
        f"  var(estimate_h)={total:.4f}  between-outcome={means.var(ddof=0):.4f}  "
        f"share={means.var(ddof=0) / total:.3f}"
    )
    oracle = pairs.assign(estimate_l=means)
    print(
        "  oracle knowing ONLY the per-outcome human mean: "
        f"r={M.pearson(oracle['estimate_h'], oracle['estimate_l']):+.3f} "
        f"rho={M.spearman(oracle['estimate_h'], oracle['estimate_l']):+.3f} "
        f"dir={M.directional_score(oracle['estimate_h'], oracle['estimate_l']):.1f}"
    )


def anchor_break_even(pairs: pd.DataFrame, draws: int = 400, seed: int = 7) -> None:
    """Where a transferred profile starts beating our own.

    The anchor will not be the truth, so the question is not "does the oracle
    help" but "how good must a borrowed profile be before it stops hurting".
    Perturb the true profile to a target correlation and read off the crossing.
    """
    rng = np.random.default_rng(seed)
    truth = pairs.groupby("outcome")["estimate_h"].mean()
    ours = pairs.groupby("outcome")["estimate_l"].mean()
    standard = (truth - truth.mean()) / truth.std()
    baseline = M.pearson(pairs["estimate_h"], pairs["estimate_l"])
    print(f"  corr(our profile, true profile) = {np.corrcoef(ours, truth)[0, 1]:+.3f}")
    print(f"  raw pooled r (the bar to beat)  = {baseline:+.3f}")
    rows = []
    for target in (0.2, 0.4, 0.6, 0.8, 1.0):
        scores = []
        for _ in range(draws):
            noise = rng.normal(size=len(truth))
            noise = (noise - noise.mean()) / noise.std()
            mixed = target * standard + np.sqrt(max(1 - target**2, 0.0)) * noise
            anchor = pd.Series(mixed * truth.std() + truth.mean(), index=truth.index)
            blended = reprofile(pairs, 1.0, anchor)
            scores.append(M.pearson(blended["estimate_h"], blended["estimate_l"]))
        rows.append(
            {
                "rho_anchor_vs_truth": target,
                "pooled_r": np.mean(scores),
                "p10": np.percentile(scores, 10),
                "p90": np.percentile(scores, 90),
                "beats_raw": np.mean(scores) > baseline,
            }
        )
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda v: f"{v:7.3f}"))


def main() -> None:
    pairs, human2 = load_pairs()
    print(f"runs found: {list(pairs)}\n")

    print("=== Baseline (reproduces docs/reports/voelkel_validation) ===")
    for name, frame in pairs.items():
        print(_row(name, frame))
    print(_row("human replication (Human 2)", human2))

    print("\n=== Claim 1: the human effect vector is mostly between-outcome ===")
    decompose(next(iter(pairs.values())))

    print("\n=== Claim 2: which levers move pearson_r ===")
    for name, frame in pairs.items():
        for factor in (0.5, 0.159, 0.05):
            print(_row(f"{name} global k={factor}", shrink_global(frame, factor)))
        for factor in (0.5, 0.2):
            print(_row(f"{name} within k={factor}", shrink_within(frame, factor)))
        for weight in (0.5, 1.0):
            print(_row(f"{name} reprofile w={weight}", reprofile(frame, weight)))

    print("\n=== Claim 3: cross-model averaging ===")
    combined = average(pairs)
    if combined is not None:
        print(
            f"  corr between the two models' effects = {combined.attrs['cross_model_r']:+.3f}"
        )
        print(_row("average of both models", combined))
        best = max(pairs, key=lambda n: score(pairs[n])["pearson_r"])
        delta = M.paired_cluster_bootstrap(
            pairs[best][KEY + ["estimate_h", "se_h", "estimate_l", "se_l"]],
            combined,
            lambda f: {
                "pearson_r": M.pearson(f["estimate_h"], f["estimate_l"]),
                "spearman_rho": M.spearman(f["estimate_h"], f["estimate_l"]),
            },
            cluster="condition",
        )
        print(
            f"  paired cluster bootstrap, average vs {best} (the better single model):"
        )
        for metric in ("pearson_r", "spearman_rho"):
            print(
                f"    {metric}: {delta[f'{metric}_delta']:+.4f} "
                f"[{delta[f'{metric}_delta_lo']:+.4f}, {delta[f'{metric}_delta_hi']:+.4f}] "
                f"p(better)={delta[f'{metric}_p_gt0']:.3f}"
            )

    print("\n=== The break-even a transferred profile has to clear ===")
    anchor_break_even(next(iter(pairs.values())))


if __name__ == "__main__":
    main()
