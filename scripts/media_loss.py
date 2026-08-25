"""Are the models worse on arms whose stimulus was not only text?

Every model in this project reads a questionnaire as prose.  Where an arm's
stimulus was a photograph, an infographic or a video, the text template can at
best *describe* it, and for six of Goldwert's eighteen arms the fidelity audit
concluded the intervention's core is not in the text at all.  That is a known,
unavoidable handicap of text-only silicon sampling, and it should show up as
worse effect recovery on exactly those arms.

This script tests that.  The severity ratings come from the audit's
``modality_audit.csv`` and were assigned by reading each arm's stimulus, before
any of the accuracy numbers below were computed -- so the grouping is not fitted
to the outcome it is being used to explain.

Accuracy is measured on **within-outcome demeaned** effects.  Raw effects are
dominated by which outcome is being moved rather than by which arm is moving it,
and that between-outcome component is shared by every arm, so it would wash out
the very contrast being tested.  Demeaning within outcome leaves the part of the
effect that is about the arm.

Run: ``python scripts/media_loss.py [--model qwen25_72b_v3]``
"""

from __future__ import annotations

import argparse
import warnings

import numpy as np
import pandas as pd

from silicon_sampling import models as MODELS
from silicon_sampling import paths as P
from silicon_sampling.benchmark.reference import ate_pairs, half_split

warnings.filterwarnings("ignore")


def arm_accuracy(study: str, score, paths, run: str) -> pd.DataFrame:
    """Per-arm recovery of within-outcome demeaned effects."""
    key = MODELS.resolve_run(paths.samples_dir, run)
    if key is None:
        return pd.DataFrame()
    sample = pd.read_csv(paths.samples_dir(key) / "samples.csv", low_memory=False)
    human1, _ = half_split(score.load_humans())
    pairs = ate_pairs(score.effects(human1), score.effects(sample)).dropna(
        subset=["estimate_h", "estimate_l"]
    )

    # Effects are on different scales per outcome, so standardise within outcome
    # before demeaning; otherwise a 0-100 slider dominates a 0-1 binary.
    def z(g: pd.DataFrame) -> pd.DataFrame:
        for col in ("estimate_h", "estimate_l"):
            sd = g[col].std(ddof=0)
            g[col + "_z"] = (g[col] - g[col].mean()) / (sd if sd > 0 else np.nan)
        return g

    pairs = pairs.groupby("outcome", group_keys=False).apply(z)
    out = []
    for arm, g in pairs.groupby("condition"):
        g = g.dropna(subset=["estimate_h_z", "estimate_l_z"])
        if len(g) < 3:
            continue
        h, ours = g["estimate_h_z"].to_numpy(), g["estimate_l_z"].to_numpy()
        out.append(
            {
                "study": study,
                "condition": str(arm),
                "n_outcomes": len(g),
                "dir_pct": float((np.sign(h) == np.sign(ours)).mean() * 100),
                "mae": float(np.abs(h - ours).mean()),
                "r_arm": float(np.corrcoef(h, ours)[0, 1]) if np.std(ours) > 0 else np.nan,
            }
        )
    return pd.DataFrame(out)


def goldwert_severity() -> pd.DataFrame:
    a = pd.read_csv(P.resolve("Goldwert", "text_templates", "modality_audit.csv"))
    return a[["condName", "media_loss", "media_loss_meaning", "n_assets"]].rename(
        columns={"condName": "condition"}
    )


def icpc_severity() -> pd.DataFrame:
    """ICPC's audit counts assets rather than rating loss, so bucket the counts.

    Its stimuli are uniformly static images inside otherwise self-contained
    argumentative text, which is why the audit rated none of them unusable.  The
    split here is therefore image-heavy against image-light, not a loss rating,
    and is reported as such.
    """
    a = pd.read_csv(P.resolve("ICPC", "text_templates", "modality_audit.csv"))
    a = a.assign(n_assets=a["image"] + a["video"] + a["iframe"])
    a["media_loss"] = pd.cut(a["n_assets"], [-1, 0, 4, 100], labels=[0, 1, 2]).astype(
        int
    )
    a["media_loss_meaning"] = a["media_loss"].map(
        {0: "no assets", 1: "1-4 images", 2: "5+ images"}
    )
    # Two arms are stored in the samples under their raw Qualtrics name rather
    # than the audit's alias ("Identity-Social-Norms-Intervention" for
    # WorkTogetherNorm, "Letter2Future" for LetterFutureGen).  Emitting a row for
    # each spelling keeps them in the comparison; merging on the alias alone
    # dropped both, which is how the image-heaviest arm in the study went missing.
    cols = ["media_loss", "media_loss_meaning", "n_assets"]
    by_alias = a[["alias"] + cols].rename(columns={"alias": "condition"})
    by_raw = a[["condition"] + cols]
    return pd.concat([by_alias, by_raw]).drop_duplicates(subset=["condition"])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="qwen25_72b_v3")
    args = ap.parse_args()

    from silicon_sampling.goldwert import paths as gp
    from silicon_sampling.goldwert import score as gs
    from silicon_sampling.icpc import paths as ip
    from silicon_sampling.icpc import score as isc

    print(f"model: {args.model}\n")
    for study, score, paths, sev in (
        ("Goldwert", gs, gp, goldwert_severity()),
        ("ICPC", isc, ip, icpc_severity()),
    ):
        acc = arm_accuracy(study, score, paths, args.model)
        if acc.empty:
            print(f"{study}: no sample\n")
            continue
        m = acc.merge(sev, on="condition", how="left")
        unmatched = m["media_loss"].isna().sum()
        m = m.dropna(subset=["media_loss"])
        print(f"=== {study} ===")
        print(
            m[
                [
                    "condition",
                    "media_loss",
                    "n_assets",
                    "n_outcomes",
                    "dir_pct",
                    "r_arm",
                    "mae",
                ]
            ]
            .sort_values("media_loss")
            .to_string(index=False, float_format=lambda v: f"{v:6.2f}")
        )
        g = m.groupby("media_loss").agg(
            arms=("condition", "count"),
            dir_pct=("dir_pct", "mean"),
            r_arm=("r_arm", "mean"),
            mae=("mae", "mean"),
        )
        print("\nby severity:")
        print(g.to_string(float_format=lambda v: f"{v:6.2f}"))
        if m["media_loss"].nunique() > 1:
            rho = m["media_loss"].corr(m["r_arm"], method="spearman")
            rho_m = m["media_loss"].corr(m["mae"], method="spearman")
            print(
                f"\nSpearman(severity, r_arm)  = {rho:+.3f}"
                f"   [negative = worse when more is lost]"
                f"\nSpearman(severity, mae)    = {rho_m:+.3f}"
                f"   [positive = worse when more is lost]"
            )
        if unmatched:
            print(f"\n{unmatched} arm(s) had no severity rating and were dropped")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
