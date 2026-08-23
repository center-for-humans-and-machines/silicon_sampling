"""Command line for the Goldwert calibration study."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from . import outcomes as oc
from . import paths, profiles


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="silicon_sampling.goldwert")
    sub = parser.add_subparsers(dest="command", required=True)

    audit = sub.add_parser(
        "modality-audit", help="per-arm media counts and the keep-or-drop call"
    )
    audit.add_argument("--out", default=None)

    sub.add_parser("render-templates", help="write data/Goldwert/text_templates")

    verify = sub.add_parser(
        "verify-outcomes",
        help="recompute every derived column from raw items and check the published one",
    )
    verify.add_argument("--out", default=None)

    reference = sub.add_parser(
        "human-reference",
        help="control levels, reference ATEs and the human-replication score",
    )
    reference.add_argument("--out", default=None)
    reference.add_argument("--seed", type=int, default=42)

    anchors = sub.add_parser(
        "pfander-anchors",
        help="what this study can calibrate for the Pfänder target, and at what level",
    )
    anchors.add_argument("--out", default=None)

    sub.add_parser(
        "us-coverage", help="per-arm counts and the reach of every US-looking column"
    )

    build = sub.add_parser(
        "build-profiles", help="draw respondents from the published marginals"
    )
    build.add_argument("--seed", type=int, default=20260823)
    build.add_argument("--per-arm", type=int, default=None)

    args = parser.parse_args(argv)
    pd.set_option("display.width", 220)

    if args.command == "modality-audit":
        from .templates import write_modality_audit

        rows = write_modality_audit()
        table = pd.DataFrame(rows)
        shown = table[
            [
                "cond",
                "condName",
                "modality",
                "usable",
                "n_blocks",
                "words",
                "image",
                "piped_image",
                "graphic",
                "video",
                "iframe",
            ]
        ]
        print(shown.to_string(index=False))
        usable = int((table["usable"] == "yes").sum())
        print(f"\n{usable} of {len(table)} arms usable in a text transcript")
        print(f"wrote {paths.MODALITY_AUDIT}")
        return 0

    if args.command == "render-templates":
        from .templates import render_all

        manifest = render_all()
        written = [a for a in manifest["arms"].values() if a["usable"]]
        print(f"wrote {len(written)} templates to {paths.TEMPLATES}")
        for name, entry in manifest["arms"].items():
            if entry["usable"]:
                print(
                    f"  {entry['file']:46} {entry['n_slots']:3} slots  {entry['chars']:6} chars"
                )
            else:
                print(f"  -- dropped {name:32} ({entry['modality']})")
        return 0

    if args.command == "verify-outcomes":
        frame = pd.read_csv(paths.RESPONSES_CSV, low_memory=False)
        table = oc.verify_against_published(frame)
        print(table.to_string(index=False))
        print(f"\nall derived columns reproduce: {bool(table['matches'].all())}")
        print()
        print(oc.coverage(oc.compute(frame)).to_string(index=False))
        if args.out:
            table.to_csv(Path(args.out), index=False)
        return 0 if bool(table["matches"].all()) else 1

    if args.command == "human-reference":
        from . import score

        humans = score.load_humans()
        print(
            f"humans in usable arms: {len(humans)} across {humans['condition'].nunique()} arms"
        )
        print()
        print("Control-arm levels")
        print(score.control_levels(humans).to_string(index=False))
        print()
        board, reference = score.leaderboard(frame=humans, seed=args.seed)
        columns = [
            c
            for c in (
                "submission",
                "n_pairs",
                "directional_pct",
                "pearson_r",
                "rmse",
                "beta",
                "n_clusters",
            )
            if c in board.columns
        ]
        print("Human replication and baselines")
        print(board[columns].to_string(index=False))
        print()
        print("Display-position sensitivity of each outcome")
        print(score.position_effects(humans).to_string(index=False))
        if args.out:
            out = Path(args.out)
            out.mkdir(parents=True, exist_ok=True)
            board.to_csv(out / "leaderboard.csv", index=False)
            reference.to_csv(out / "reference_effects.csv", index=False)
            score.control_levels(humans).to_csv(out / "control_levels.csv", index=False)
            print(f"\nwrote {out}")
        return 0

    if args.command == "pfander-anchors":
        frame = pd.read_csv(paths.RESPONSES_CSV, low_memory=False)
        table = oc.pfander_anchor_table()
        print(
            table[
                [
                    "goldwert_column",
                    "goldwert_scale",
                    "pfander_outcome",
                    "pfander_scale",
                    "closeness",
                ]
            ].to_string(index=False)
        )
        print()
        for anchor in oc.PFANDER_ANCHORS:
            print(
                f"{anchor.goldwert_column} -> {anchor.pfander_outcome} ({anchor.closeness})"
            )
            print(f"   anchors : {anchor.what_it_anchors}")
            print(f"   gap     : {anchor.wording_gap}")
        print()
        print(
            f"Pfänder outcomes with no counterpart here: {', '.join(oc.PFANDER_UNCOVERED)}"
        )
        print()
        levels = oc.anchor_levels(frame)
        print(levels.to_string(index=False))
        if args.out:
            out = Path(args.out)
            out.mkdir(parents=True, exist_ok=True)
            table.to_csv(out / "pfander_anchors.csv", index=False)
            levels.to_csv(out / "anchor_levels.csv", index=False)
            print(f"\nwrote {out}")
        return 0

    if args.command == "us-coverage":
        from . import score

        print(score.US_FILTER_NOTE)
        print()
        table = score.us_coverage()
        print(table.to_string(index=False))
        totals = table.drop(columns=["condName", "cond"]).sum()
        print()
        print("totals: " + "  ".join(f"{k}={v}" for k, v in totals.items()))
        return 0

    if args.command == "build-profiles":
        built = profiles.build(seed=args.seed, per_arm=args.per_arm)
        profiles.write_csv(built, paths.PROFILES_CSV)
        print(f"wrote {len(built)} profiles to {paths.PROFILES_CSV}")
        print(json.dumps(profiles.arm_table()[:3], indent=2))
        return 0

    return 1  # pragma: no cover


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
