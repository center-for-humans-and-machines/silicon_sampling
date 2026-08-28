"""Grade the shipped *structural* calibration on CCC, held out.

The nested cross-validation grades the effect side: averaging, the within-outcome
factor, the global factor.  It says nothing about the other three quarters of the
benchmark, because ``response_distributions``, ``subgroup_distributions``,
``compare_demographic_baselines`` and ``demographic_parity_gap`` all read the
**control arm only** and none of them involves an arm contrast at all.

Until now that half was graded only on Pfander itself, against TISP anchors — one
study, three outcomes, and no way to check whether the correction transfers.  CCC
is the first study that can grade it out of sample, because it publishes 12,757
real respondents on nine outcomes that Pfander's crosswalk reaches.

**The leakage rule.**  Six of the nine shipped party-gap anchors and five of the
eight dispersion anchors were measured *on CCC*.  Grading a CCC prediction that
used them against CCC humans would be scoring a number against itself.  So this
script switches them off via :func:`silicon_sampling.anchors.ccc.for_study`, which
returns no anchors when CCC is the held-out study.  What is left is what genuinely
transfers:

* ``RESIDUAL_SCALE = 1.12``, fitted on TISP's three trust outcomes
* the rule "rescale the model's within-arm spread to human spread", as a rule
* the raw, uncorrected model party gaps — because with CCC's own anchors gone
  there is no party anchor that reaches a climate outcome, and the honest
  question becomes how good the models are unaided

That last point is the one worth reading carefully.  It measures what the party
correction is *worth*: if raw gaps are already close, the anchoring is decoration;
if they are far off, the six CCC anchors are load-bearing for Pfander and their
absence here is the price of an honest fold.

Run: ``python scripts/score_ccc_holdout.py``
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from silicon_sampling import models as MODELS
from silicon_sampling.anchors import ccc as CCC_ANCHORS
from silicon_sampling.benchmark import distributions as DIST
from silicon_sampling.calibration import recipes as R
from silicon_sampling.ccc import outcomes as OC
from silicon_sampling.ccc import paths as PATHS
from silicon_sampling.ccc import score as SCORE

warnings.filterwarnings("ignore")

RUNS = ("qwen25_7b", "qwen25_72b", "muse_glimmer_30b")

#: Half-split seed, matching ``benchmark.reference.half_split`` so the human
#: ceiling reported here is the same split the cross-validation uses.
SEED = 20240517


def controls(frame: pd.DataFrame) -> pd.DataFrame:
    """The pooled control arm, whichever vocabulary the frame arrived with."""
    return SCORE.pool_controls(frame).query("condition == @SCORE.CONTROL")


def party_gap(frame: pd.DataFrame, outcome: str, span: float) -> float:
    """Democrat minus Republican, in pp of the outcome's range."""
    party = frame["party"].astype(str)
    dem = pd.to_numeric(
        frame[party.str.contains("Democrat", na=False)][outcome], errors="coerce"
    )
    rep = pd.to_numeric(
        frame[party.str.contains("Republican", na=False)][outcome], errors="coerce"
    )
    if len(dem.dropna()) < 30 or len(rep.dropna()) < 30:
        return float("nan")
    return float((dem.mean() - rep.mean()) / span * 100)


def grade(human: pd.DataFrame, synth: pd.DataFrame, scale: float) -> pd.DataFrame:
    """Per-outcome level, shape and party-gap error for one control arm."""
    rows = []
    for outcome, span in OC.SCORED.items():
        if outcome not in human.columns or outcome not in synth.columns:
            continue
        left = pd.to_numeric(human[outcome], errors="coerce").dropna()
        right = pd.to_numeric(synth[outcome], errors="coerce").dropna()
        if len(left) < 30 or len(right) < 30 or left.std() == 0:
            continue
        scaled = (right.mean() + scale * (right - right.mean())).clip(0.0, span)
        shape = DIST.compare_distributions(left, scaled, lo=0.0, hi=span)
        adjusted = synth.assign(**{outcome: scaled.reindex(synth.index)})
        rows.append(
            {
                "outcome": outcome,
                "human mean": float(left.mean()),
                "ours": float(scaled.mean()),
                "level err (pp)": abs(float(scaled.mean()) - float(left.mean()))
                / span
                * 100,
                "human sd": float(left.std()),
                "sd ratio": float(scaled.std() / left.std()),
                "ovl": shape["ovl"],
                "ks": shape["ks"],
                "w1": shape["w1"],
                "human gap": party_gap(human, outcome, span),
                "our gap": party_gap(adjusted, outcome, span),
            }
        )
    frame = pd.DataFrame(rows)
    frame["gap err"] = (frame["our gap"] - frame["human gap"]).abs()
    return frame


def summarise(label: str, frame: pd.DataFrame) -> dict:
    return {
        "variant": label,
        "level err (pp)": frame["level err (pp)"].mean(),
        "sd ratio": frame["sd ratio"].mean(),
        "|sd ratio-1|": (frame["sd ratio"] - 1).abs().mean(),
        "ovl": frame["ovl"].mean(),
        "ks": frame["ks"].mean(),
        "w1": frame["w1"].mean(),
        "party gap RMSE": float(np.sqrt((frame["gap err"] ** 2).mean())),
    }


def main() -> int:
    pd.set_option("display.width", 250)

    withheld = CCC_ANCHORS.for_study("CCC")
    print("=== leakage guard ===\n")
    print(f"  anchors available with CCC held out: {withheld or '(none)'}")
    shipped_from_ccc = [
        k
        for k in R.PARTY_GAP_ANCHORS
        if k in {a.pfander_outcome for a in CCC_ANCHORS.ANCHORS}
    ]
    print(
        f"  shipped party-gap anchors that came from CCC, now switched off: {len(shipped_from_ccc)}"
    )
    print(
        f"  shipped dispersion anchors from CCC, now switched off: "
        f"{len([k for k in R.HUMAN_DISPERSION if k in shipped_from_ccc])}"
    )
    print(
        f"  residual scale under test: {R.RESIDUAL_SCALE} (fitted on TISP, not on CCC)\n"
    )

    humans = SCORE.load_humans()
    human_control = controls(humans)
    print(f"  CCC human control arm: {len(human_control)} respondents\n")

    rows, detail = [], {}
    for run in RUNS:
        key = MODELS.resolve_run(PATHS.samples_dir, run)
        if key is None:
            print(f"  {run}: no sample, skipped")
            continue
        sample = pd.read_csv(PATHS.samples_dir(key) / "samples.csv", low_memory=False)
        synth = controls(sample)
        for label, scale in (("raw", 1.0), (f"x{R.RESIDUAL_SCALE}", R.RESIDUAL_SCALE)):
            frame = grade(human_control, synth, scale)
            rows.append(summarise(f"{run}, {label}", frame))
            detail[f"{run}, {label}"] = frame

    # The rule rather than the constant: rescale to whatever this study's own
    # models say, which is what a Pfander run can actually do (it has no human
    # dispersion to read).  Reported to separate "1.12 is right" from "rescaling
    # is right".
    print("=== held-out control-arm quality on CCC ===\n")
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda v: f"{v:8.3f}"))

    # Human ceiling: split the control arm in half and grade one half against the
    # other, which is the best any method could score given sampling noise alone.
    rng = np.random.default_rng(SEED)
    order = rng.permutation(len(human_control))
    half = len(order) // 2
    left = human_control.iloc[order[:half]]
    right = human_control.iloc[order[half:]]
    ceiling = grade(left, right, 1.0)
    print("\nhuman half-against-half on the same outcomes (the ceiling):\n")
    print(
        pd.DataFrame([summarise("human replication", ceiling)]).to_string(
            index=False, float_format=lambda v: f"{v:8.3f}"
        )
    )

    best = min(rows, key=lambda r: r["level err (pp)"])
    print(f"\n\n=== per-outcome detail for the best level fit: {best['variant']} ===\n")
    print(
        detail[best["variant"]].to_string(
            index=False, float_format=lambda v: f"{v:8.2f}"
        )
    )

    print("\n\n=== what the party anchors are worth ===\n")
    print("  shipped anchor values against what CCC humans actually show:\n")
    measured = CCC_ANCHORS.measure()
    print(measured.to_string(index=False, float_format=lambda v: f"{v:8.2f}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
