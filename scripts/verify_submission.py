"""Run every scored analysis over a built entry, to catch structural gaps early.

The format checker in ``build_entries.py`` answers "is this a valid file". This
answers the different question "will every analysis the benchmark runs actually
produce a number from it" -- a submission can be perfectly well-formed and still
have a moderator level so thin that a subgroup model is undefined, or a condition
that drops out of a within-outcome correlation.

Pfander publishes no human data, so there is nothing to score against. A second
entry stands in as the reference purely to make the machinery run. **Every value
it prints is meaningless as a measure of accuracy** and the script says so rather
than tempting a reader to quote it; what is meaningful is the shape of the
tables and whether anything came back undefined.

Reading the output: NaN in ``pearson_within`` for an intervention is structural,
since a within-outcome correlation needs more than one outcome in the group. NaN
in ``pearson_adj`` or ``beta_adj`` is the *stand-in's* fault -- both corrections
divide by the reference's own sampling variance, and a synthetic reference does
not have human-shaped standard errors. Against real participants they come back
finite and healthy: pearson_adj is 0.469 on Voelkel and 0.656 on Goldwert.

Anything else undefined is worth investigating.

Run: ``python scripts/verify_submission.py [--entry primary] [--reference secondary-2]``
"""

from __future__ import annotations

import argparse
import dataclasses
import glob
import warnings

import numpy as np
import pandas as pd

from silicon_sampling.benchmark import scored as SC
from silicon_sampling.pfander.scoring import DESIGN

warnings.filterwarnings("ignore")

#: Moderator levels the study's own design allocates nobody to.
#:
#: Pfander's Preregistration Table 3 quotas gender as male and female only, and
#: its four age bands sum to exactly 18,000 across those two columns.  The
#: questionnaire still offers "Other" as a response option, but no respondent is
#: assigned to it, so an empty cell there reproduces the design rather than
#: missing part of it.
UNALLOCATED = {("gender", "Other")}

#: Columns whose emptiness carries no information about the submission.
BENIGN = {
    "pearson_within": "undefined for a group with one outcome",
    "pearson_adj": "needs the reference's standard errors; the stand-in has none",
    "beta_adj": "needs the reference's standard errors; the stand-in has none",
    "groups_skipped": "a reporting column; empty means nothing was skipped",
    "groups_skipped_detail": "a reporting column; empty means nothing was skipped",
}


def load(entry: str) -> pd.DataFrame:
    hits = sorted(glob.glob(f"data/pfander/submission/{entry}/predictions/*.csv"))
    if not hits:
        raise SystemExit(f"no predictions under data/pfander/submission/{entry}/")
    return pd.read_csv(hits[0], low_memory=False)


def scalars(payload: dict) -> list[tuple[str, float]]:
    def walk(node, path=""):
        found: list[tuple[str, float]] = []
        if isinstance(node, dict):
            for key, value in node.items():
                found += walk(value, f"{path}.{key}")
        elif isinstance(node, (int, float, np.floating)) and not isinstance(node, bool):
            found.append((path.lstrip("."), float(node)))
        return found

    return walk({k: v for k, v in payload.items() if not isinstance(v, pd.DataFrame)})


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--entry", default="primary")
    ap.add_argument("--reference", default="secondary-2")
    args = ap.parse_args()

    prediction, reference = load(args.entry), load(args.reference)
    print(f"entry '{args.entry}', with '{args.reference}' standing in as the reference")
    print("Accuracy numbers here are MEANINGLESS -- this checks structure only.\n")

    sides = SC.reference_sides(reference, DESIGN, subgroup=True)
    report = SC.score_submission(
        reference, prediction, DESIGN, label=args.entry, sides=sides, subgroups=True
    )
    payload = (
        dataclasses.asdict(report)
        if dataclasses.is_dataclass(report)
        else dict(report.__dict__)
    )
    frames = {k: v for k, v in payload.items() if isinstance(v, pd.DataFrame)}

    print("tables produced")
    problems = 0
    for name, table in sorted(frames.items()):
        scored = table
        if {"moderator", "level"} <= set(table.columns):
            keep = ~table.apply(
                lambda r: (r["moderator"], r["level"]) in UNALLOCATED, axis=1
            )
            dropped = int((~keep).sum())
            scored = table[keep]
            if dropped:
                print(
                    f"  {'':28s} ({dropped} row(s) for levels the quota allocates nobody to)"
                )
        empty = {
            c: int(scored[c].isna().sum()) for c in scored if scored[c].isna().any()
        }
        unexplained = {c: n for c, n in empty.items() if c not in BENIGN}
        problems += len(unexplained)
        note = f"  <-- unexplained: {unexplained}" if unexplained else ""
        print(f"  {name:28s} {len(table):5d} x {table.shape[1]:2d}{note}")

    values = scalars(payload)
    undefined = [p for p, v in values if not np.isfinite(v)]
    unexplained_scalars = [p for p in undefined if p.rsplit(".", 1)[-1] not in BENIGN]
    problems += len(unexplained_scalars)
    print(f"\nscalar metrics: {len(values)}, undefined: {len(undefined)}")
    for path in undefined:
        key = path.rsplit(".", 1)[-1]
        print(f"  {path:36s} {BENIGN.get(key, 'UNEXPLAINED')}")

    expected = (DESIGN.n_pairs_expected, len(frames))
    print(
        f"\npairs built: {len(frames.get('pairs', []))} (design expects {expected[0]})"
    )
    print("VERDICT:", "clean" if problems == 0 else f"{problems} unexplained gap(s)")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
