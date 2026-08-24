"""Three things called "ensembling", measured against real participants.

Averaging models is the largest free improvement this project found, so it is
worth being exact about what gets averaged when the unit of submission is a
synthetic person.  Three operations all go by the name and behave differently:
averaging the fitted effects and rebuilding one model's respondents onto them;
pooling respondents so each row is a real output of some model; and averaging each
respondent's own answers across models.

All three are constructible because every model answered the same profile set --
``profile_id`` v00001 is the same person with the same seed in each run -- which is
a deliberate property of the sampler.

The result, and the reason this script exists: with balanced n all three produce
*identical* aggregate effects, because the ATE of a mixture is the mixture of the
ATEs.  They differ only in what they do to the individuals, which is what the
benchmark's response-distribution section scores -- and per-respondent averaging
divides each person's idiosyncratic variance by the number of models, manufacturing
the exact under-dispersion that section exists to catch.

Run: ``python scripts/compare_ensembling.py``
"""

import numpy as np
import pandas as pd
from silicon_sampling.voelkel import outcomes as voc
from silicon_sampling.voelkel import paths as P
from silicon_sampling.voelkel import score as S
from silicon_sampling.benchmark.reference import half_split, ate_pairs
from silicon_sampling.benchmark import distributions as D
from silicon_sampling.benchmark import metrics as M
from silicon_sampling.calibration import components as C
from silicon_sampling.calibration import tier1 as T1

OUT = list(voc.OUTCOMES)
CONTROL = S.CONTROL
MODS = ("party_gen", "gender", "race", "education", "age_band")
h1, h2 = half_split(S.load_humans())
ref = S.effects(h1)
runs = {
    m: pd.read_csv(P.samples_dir(m) / "samples.csv", low_memory=False)
    for m in ("qwen25_7b", "qwen25_72b", "v4_flash")
}
DESIGN = T1.Instrument(
    scales=dict(voc.OUTCOMES),
    control=CONTROL,
    moderators=MODS,
    binary=(),
    composites={},
)


def score(frame, label):
    p = ate_pairs(ref, S.effects(frame)).dropna(subset=["estimate_h", "estimate_l"])
    m = M.pooled_metrics(p) | M.run_calibration_pooled(p)
    ctrl_h = h1[h1.condition == CONTROL]
    ctrl_s = frame[frame.condition == CONTROL]
    sh = pd.DataFrame(
        [D.compare_distributions(ctrl_h[o].dropna(), ctrl_s[o].dropna()) for o in OUT]
    )
    return {
        "submission": label,
        "n_rows": len(frame),
        "r": m["pearson_r"],
        "rho": m["spearman_rho"],
        "dir": m["directional_pct"],
        "rmse": m["rmse"],
        "var_ratio": sh["variance_ratio"].median(),
        "ovl": sh["ovl"].median(),
        "ks": sh["ks"].median(),
        "w1": sh["w1"].median(),
    }


rows = [score(f, f"single: {k}") for k, f in runs.items()]

# (a) EFFECT-LEVEL averaging: average the 54 ATEs, rebuild one model's respondents
#     onto those targets. Individuals stay one model's; only the aggregate is mixed.
tables = []
for k, f in runs.items():
    for o in OUT:
        for cond, v in C.condition_effects(f, o, CONTROL).items():
            tables.append(
                {
                    "model": k,
                    "outcome": o,
                    "condition": cond,
                    "estimate": float(v),
                    "se": 1.0,
                }
            )
avg = (
    pd.DataFrame(tables)
    .groupby(["outcome", "condition"], as_index=False)
    .agg(estimate=("estimate", "mean"), se=("se", "mean"))
)
eff_avg, drift = T1.calibrate(
    runs["qwen25_7b"], targets=avg, instrument=DESIGN, targets_in_pp=False
)
rows.append(score(eff_avg, "(a) effect-level average") | {})

# (b) RESPONDENT POOLING: each row is a real synthetic respondent from one model.
#     Disjoint thirds so n stays 6,203 and every arm keeps its share.
rng = np.random.default_rng(11)
assign = pd.Series(
    rng.integers(0, 3, len(runs["qwen25_7b"])), index=runs["qwen25_7b"].index
)
pooled = pd.concat([runs[m].loc[assign == i] for i, m in enumerate(runs)]).sort_index()
rows.append(score(pooled, "(b) pooled respondents (n same)"))
rows.append(
    score(pd.concat(list(runs.values()), ignore_index=True), "(b') pooled, all 3x n")
)

# (c) PER-RESPONDENT averaging: same person, mean of their answers across models.
per = runs["qwen25_7b"].copy()
for o in OUT:
    stack = np.vstack(
        [
            runs[m].set_index("profile_id").loc[per.profile_id, o].to_numpy(float)
            for m in runs
        ]
    )
    per[o] = np.nanmean(stack, axis=0)
rows.append(score(per, "(c) per-respondent average"))

rows.append(score(h2, "human replication"))
t = pd.DataFrame(rows)
pd.set_option("display.width", 220)
print(t.to_string(index=False, float_format=lambda v: f"{v:8.3f}"))
print(f"\n(a) effect drift {drift['max_abs_effect_drift'].max():.1e}")
print("""
Targets: r/rho/dir high, rmse low; var_ratio 1, ovl 1, ks 0, w1 0.
The distribution columns are what separate the three schemes -- they all move the
aggregate effects similarly, and differ in what they do to the individuals.""")
