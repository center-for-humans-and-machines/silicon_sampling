"""Ablate the shipped recipe, one component at a time.

Two halves, because the benchmark scores two disjoint things and no single
reference can grade both.

**The effect side** is graded out-of-fold on the three reference studies, which
publish participant responses.  Pfander does not, so nothing about arm contrasts
can be measured on Pfander itself.

**The structure side** is graded on Pfander directly, against external anchors:
TISP for control-arm levels and dispersion on the outcomes its crosswalk grades
``near``, and the party-gap estimates in ``recipes.PARTY_GAP_ANCHORS``.  This is
the half that carries every distributional and demographic metric, because all
four of those analyses read the control condition only.

Run: ``python scripts/ablate_recipe.py [--half effects|structure|both]``
"""

from __future__ import annotations

import argparse
import warnings

import numpy as np
import pandas as pd

from silicon_sampling import models as MODELS
from silicon_sampling.benchmark.reference import ate_pairs, half_split
from silicon_sampling.calibration import recipes as R
from silicon_sampling.calibration import tier1 as T1

warnings.filterwarnings("ignore")

STUDIES = (
    ("Voelkel", "silicon_sampling.voelkel.score", "silicon_sampling.voelkel.paths"),
    ("ICPC", "silicon_sampling.icpc.score", "silicon_sampling.icpc.paths"),
    ("Goldwert", "silicon_sampling.goldwert.score", "silicon_sampling.goldwert.paths"),
)


def study_pairs(runs: tuple[str, ...]) -> dict[str, pd.DataFrame]:
    """Human effects beside each run's, per study, on the common pair set."""
    out = {}
    for name, mod, pmod in STUDIES:
        score = __import__(mod, fromlist=["x"])
        paths = __import__(pmod, fromlist=["x"])
        human1, _ = half_split(score.load_humans())
        reference = score.effects(human1)
        frames = {}
        for run in runs:
            key = MODELS.resolve_run(paths.samples_dir, run)
            if key is None:
                continue
            sample = pd.read_csv(
                paths.samples_dir(key) / "samples.csv", low_memory=False
            )
            pairs = ate_pairs(reference, score.effects(sample)).dropna(
                subset=["estimate_h", "estimate_l"]
            )
            frames[run] = pairs.set_index(["outcome", "condition"])
        if not frames:
            continue
        index = None
        for frame in frames.values():
            index = frame.index if index is None else index.intersection(frame.index)
        table = pd.DataFrame({"h": frames[runs[0]].loc[index, "estimate_h"]})
        for run, frame in frames.items():
            table[run] = frame.loc[index, "estimate_l"]
        table["outcome"] = [i[0] for i in index]
        out[name] = table.reset_index(drop=True)
    return out


def within_shrink(table: pd.DataFrame, column: str, factor: float) -> np.ndarray:
    """Shrink each arm's effect toward its own outcome's mean effect."""
    grouped = table.groupby("outcome")[column]
    return (
        grouped.transform("mean") + factor * (table[column] - grouped.transform("mean"))
    ).to_numpy()


def effect_ablation() -> pd.DataFrame:
    runs = ("qwen25_7b_v3", "qwen25_72b_v3", "v4_flash_v3")
    data = study_pairs(runs)
    rows = []

    def score(label: str, vectors: dict[str, np.ndarray], kappa: float | None) -> None:
        entry = {"variant": label}
        for study, table in data.items():
            h = table["h"].to_numpy()
            pred = vectors[study]
            if kappa is not None:
                pred = kappa * pred
            entry[study] = float(np.corrcoef(h, pred)[0, 1])
        entry["mean r"] = float(np.mean([entry[s] for s in data]))
        rows.append(entry)

    for run in runs:
        score(
            f"single: {run.replace('_v3', '')}",
            {s: t[run].to_numpy() for s, t in data.items()},
            None,
        )
    avg = {
        s: 0.5 * (t["qwen25_7b_v3"] + t["qwen25_72b_v3"]).to_numpy()
        for s, t in data.items()
    }
    score("average of the two Qwens", avg, None)
    three = {
        s: (t["qwen25_7b_v3"] + t["qwen25_72b_v3"] + t["v4_flash_v3"]).to_numpy() / 3
        for s, t in data.items()
    }
    score("average of all three (V4 included)", three, None)
    for factor in (0.3, 0.5, 0.7):
        vec = {}
        for s, t in data.items():
            tmp = t.assign(avg=avg[s])
            vec[s] = within_shrink(tmp, "avg", factor)
        score(f"average + within-outcome shrink {factor}", vec, None)
    vec = {}
    for s, t in data.items():
        tmp = t.assign(avg=avg[s])
        vec[s] = within_shrink(tmp, "avg", R.WITHIN_SHRINK)
    score(f"SHIPPED + global shrink {R.GLOBAL_SHRINK}", vec, R.GLOBAL_SHRINK)

    human = {"variant": "human replication"}
    for name, mod, pmod in STUDIES:
        score_mod = __import__(mod, fromlist=["x"])
        h1, h2 = half_split(score_mod.load_humans())
        pairs = ate_pairs(score_mod.effects(h1), score_mod.effects(h2)).dropna(
            subset=["estimate_h", "estimate_l"]
        )
        human[name] = float(np.corrcoef(pairs["estimate_h"], pairs["estimate_l"])[0, 1])
    human["mean r"] = float(np.mean([human[s] for s, _, _ in STUDIES]))
    rows.append(human)
    return pd.DataFrame(rows)


def structure_ablation(donors: tuple[str, ...]) -> pd.DataFrame:
    """Control-arm quality per structural donor, against the external anchors."""
    design = T1.pfander_instrument()
    needed = tuple(dict.fromkeys((*R.BEST_RANKERS, *donors, R.PARTY_DONOR)))
    runs = R.load_runs(needed)
    anchored = tuple(R.HUMAN_DISPERSION)
    rows = []
    for donor in donors:
        recipe = R.Recipe(
            name=f"donor={donor}",
            effects_from=R.BEST_RANKERS,
            level_from=donor,
            offsets_from=donor,
            residuals_from=donor,
            within_shrink=R.WITHIN_SHRINK,
            shrink=R.shrink_for_runs(R.BEST_RANKERS),
            flatten_noise=True,
            party_offsets_from=R.PARTY_DONOR,
            party_gap_anchors=R.PARTY_GAP_ANCHORS,
            party_gap_weight=R.PARTY_GAP_WEIGHT,
            residual_scale=R.RESIDUAL_SCALE,
        )
        frame, _ = R.apply(recipe, runs=runs, instrument=design)
        control = frame[frame["condition"] == design.control]
        level_err, disp = [], []
        for outcome in anchored:
            values = pd.to_numeric(control[outcome], errors="coerce")
            disp.append(values.std() / R.HUMAN_DISPERSION[outcome])
        for outcome, target in (
            ("trust_multidimensional", 67.58),
            ("trust_post", 67.01),
            ("policy_role_mean", 64.99),
        ):
            level_err.append(
                abs(pd.to_numeric(control[outcome], errors="coerce").mean() - target)
            )
        gaps, targets = [], []
        for outcome, target in R.PARTY_GAP_ANCHORS.items():
            dem = pd.to_numeric(
                control[
                    control["party"].astype(str).str.contains("Democrat", na=False)
                ][outcome],
                errors="coerce",
            )
            rep = pd.to_numeric(
                control[
                    control["party"].astype(str).str.contains("Republican", na=False)
                ][outcome],
                errors="coerce",
            )
            gaps.append((dem.mean() - rep.mean()) / design.scales[outcome] * 100)
            targets.append(target)
        rows.append(
            {
                "structural donor": donor,
                "level err (pp)": float(np.mean(level_err)),
                "sd ratio": float(np.mean(disp)),
                "|sd ratio - 1|": float(np.mean([abs(d - 1) for d in disp])),
                "party RMSE": float(
                    np.sqrt(np.mean((np.array(targets) - np.array(gaps)) ** 2))
                ),
            }
        )
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--half", default="both", choices=["effects", "structure", "both"])
    args = ap.parse_args()
    pd.set_option("display.width", 200)
    if args.half in ("effects", "both"):
        print("=== effect side: pearson r against human effects, out-of-fold ===")
        print("(the global factor cannot move r; it is listed to show that)\n")
        print(
            effect_ablation().to_string(index=False, float_format=lambda v: f"{v:6.3f}")
        )
        print()
    if args.half in ("structure", "both"):
        print("=== structure side: control-arm quality against external anchors ===")
        print("level error is measured with anchoring OFF, so the donors differ;")
        print("the shipped entry anchors 3 of 13 outcomes exactly.\n")
        donors = ("v4_flash", "v4_flash_demo", "qwen25_72b_demo", "qwen25_7b_demo")
        print(
            structure_ablation(donors).to_string(
                index=False, float_format=lambda v: f"{v:7.3f}"
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
