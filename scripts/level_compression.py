"""Do the models compress control-arm levels toward the scale midpoint?

The CCC hold-out turned up a pattern the Pfander-only grading could not see. Our
control-arm means span 43-57 where the humans span 33-71: every high outcome is
understated and every low one overstated. If that is a *systematic* compression
rather than nine unrelated errors, it is correctable — regress the human mean on
ours across outcomes, and the slope is the expansion factor.

The correction is only worth having if it transfers, so this script asks three
questions in increasing order of difficulty:

1. **Is the compression there at all**, per study and per model? Slope > 1 means
   compressed toward the middle; slope near 1 means the levels are already
   spread correctly and only shifted.
2. **Does it transfer across outcomes within a study?** Leave one outcome out,
   fit on the rest, predict the held-out one. This is the situation Pfander is
   in for its unanchored outcomes.
3. **Does it transfer across studies?** Fit on three studies, apply to the
   fourth. This is the harder and more relevant question, because Pfander is a
   different instrument on a different topic.

A slope fitted on one study and applied to another is only legitimate if the
scales are commensurate, so everything here works in percent of each outcome's
own range.

Run: ``python scripts/level_compression.py``
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from silicon_sampling import models as MODELS

warnings.filterwarnings("ignore")

RUNS = ("qwen25_7b_v3", "qwen25_72b_v3", "muse_glimmer_30b")


def studies() -> list[dict]:
    from silicon_sampling.ccc import outcomes as co
    from silicon_sampling.goldwert import outcomes as go
    from silicon_sampling.voelkel import outcomes as vo

    return [
        {
            "name": "Voelkel",
            "score": "silicon_sampling.voelkel.score",
            "paths": "silicon_sampling.voelkel.paths",
            "scales": {k: 100.0 for k in vo.OUTCOMES},
        },
        {
            "name": "ICPC",
            "score": "silicon_sampling.icpc.score",
            "paths": "silicon_sampling.icpc.paths",
            "scales": {"belief": 100.0, "policy": 100.0, "sharing": 1.0, "wept": 8.0},
        },
        {
            "name": "Goldwert",
            "score": "silicon_sampling.goldwert.score",
            "paths": "silicon_sampling.goldwert.paths",
            "scales": dict(go.SCORED),
        },
        {
            "name": "CCC",
            "score": "silicon_sampling.ccc.score",
            "paths": "silicon_sampling.ccc.paths",
            "scales": dict(co.SCORED),
        },
    ]


def control_means(cfg: dict) -> pd.DataFrame:
    """Human and per-run control-arm means, in percent of each outcome's range."""
    score = __import__(cfg["score"], fromlist=["x"])
    paths = __import__(cfg["paths"], fromlist=["x"])
    humans = score.load_humans()
    column = next(c for c in ("condName", "condition", "cond") if c in humans.columns)
    human = humans[
        humans[column].astype(str).str.contains("ontrol", case=False, na=False)
    ]
    rows = []
    for outcome, span in cfg["scales"].items():
        if outcome not in human.columns:
            continue
        left = pd.to_numeric(human[outcome], errors="coerce").dropna()
        if len(left) < 30:
            continue
        row = {
            "study": cfg["name"],
            "outcome": outcome,
            "human": left.mean() / span * 100,
        }
        for run in RUNS:
            key = MODELS.resolve_run(paths.samples_dir, run)
            if key is None:
                continue
            sample = pd.read_csv(
                paths.samples_dir(key) / "samples.csv", low_memory=False
            )
            if hasattr(score, "pool_controls"):
                sample = score.pool_controls(sample)
            synth = sample[
                sample["condition"]
                .astype(str)
                .str.contains("ontrol", case=False, na=False)
            ]
            if outcome not in synth.columns:
                continue
            right = pd.to_numeric(synth[outcome], errors="coerce").dropna()
            if len(right) < 30:
                continue
            row[run] = right.mean() / span * 100
        rows.append(row)
    return pd.DataFrame(rows)


def fit(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Least squares ``y = a + b x``; ``b`` is the expansion factor."""
    b, a = np.polyfit(x, y, 1)
    return float(a), float(b)


def main() -> int:
    pd.set_option("display.width", 250)
    frames = {cfg["name"]: control_means(cfg) for cfg in studies()}

    print("=== 1. is the compression there? slope of human-mean on our-mean ===\n")
    print("    slope > 1: we are compressed toward the middle and need expanding")
    print("    slope ~ 1: levels are correctly spread, only shifted\n")
    rows = []
    for name, frame in frames.items():
        row = {"study": name, "n outcomes": len(frame)}
        for run in RUNS:
            if run not in frame or frame[run].isna().all():
                continue
            sub = frame.dropna(subset=[run, "human"])
            if len(sub) < 3:
                continue
            a, b = fit(sub[run].to_numpy(), sub["human"].to_numpy())
            r = float(np.corrcoef(sub[run], sub["human"])[0, 1])
            row[f"{run.replace('_v3', '')} slope"] = b
            row[f"{run.replace('_v3', '')} r"] = r
        rows.append(row)
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda v: f"{v:7.3f}"))

    print("\n\n=== 2. transfer across outcomes, within a study (leave one out) ===\n")
    print("    mean |error| in pp of range, raw against expansion-corrected\n")
    rows = []
    for name, frame in frames.items():
        for run in RUNS:
            if run not in frame:
                continue
            sub = frame.dropna(subset=[run, "human"]).reset_index(drop=True)
            if len(sub) < 4:
                continue
            raw, fixed = [], []
            for i in range(len(sub)):
                train = sub.drop(index=i)
                a, b = fit(train[run].to_numpy(), train["human"].to_numpy())
                truth = sub.loc[i, "human"]
                ours = sub.loc[i, run]
                raw.append(abs(ours - truth))
                fixed.append(abs(np.clip(a + b * ours, 0, 100) - truth))
            rows.append(
                {
                    "study": name,
                    "run": run.replace("_v3", ""),
                    "n": len(sub),
                    "raw err": float(np.mean(raw)),
                    "corrected": float(np.mean(fixed)),
                    "change": float(np.mean(fixed) - np.mean(raw)),
                }
            )
    within = pd.DataFrame(rows)
    print(within.to_string(index=False, float_format=lambda v: f"{v:7.2f}"))
    print(
        f"\n    overall: raw {within['raw err'].mean():.2f} pp -> "
        f"corrected {within['corrected'].mean():.2f} pp "
        f"({within['change'].mean():+.2f})"
    )
    print(
        f"    helped in {(within['change'] < 0).sum()}/{len(within)} study-model cells"
    )

    print(
        "\n\n=== 3. transfer across studies (fit on three, apply to the fourth) ===\n"
    )
    rows = []
    for held in frames:
        train = pd.concat(
            [f for n, f in frames.items() if n != held], ignore_index=True
        )
        test = frames[held]
        for run in RUNS:
            if run not in train or run not in test:
                continue
            tr = train.dropna(subset=[run, "human"])
            te = test.dropna(subset=[run, "human"])
            if len(tr) < 4 or len(te) < 3:
                continue
            a, b = fit(tr[run].to_numpy(), tr["human"].to_numpy())
            raw = float((te[run] - te["human"]).abs().mean())
            fixed = float((np.clip(a + b * te[run], 0, 100) - te["human"]).abs().mean())
            rows.append(
                {
                    "held out": held,
                    "run": run.replace("_v3", ""),
                    "fitted slope": b,
                    "fitted intercept": a,
                    "raw err": raw,
                    "corrected": fixed,
                    "change": fixed - raw,
                }
            )
    across = pd.DataFrame(rows)
    print(across.to_string(index=False, float_format=lambda v: f"{v:7.2f}"))
    print(
        f"\n    overall: raw {across['raw err'].mean():.2f} pp -> "
        f"corrected {across['corrected'].mean():.2f} pp "
        f"({across['change'].mean():+.2f})"
    )
    print(f"    helped in {(across['change'] < 0).sum()}/{len(across)} folds")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
