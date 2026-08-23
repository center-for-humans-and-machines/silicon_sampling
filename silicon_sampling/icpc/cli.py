"""Command line for the ICPC study package.

Everything here is cheap and runs on a laptop: rendering templates, auditing the
stimulus modality, verifying the outcome construction against the second
publication, drawing profiles, and computing the human reference the eventual
silicon sample will be scored against.  Sampling itself is not wired up yet — see
``docs/reports`` for what remains.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from . import paths, profiles


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="silicon_sampling.icpc")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("render-templates", help="write data/ICPC/text_templates")
    sub.add_parser("modality-audit", help="print the keep-or-drop call per arm")
    sub.add_parser(
        "verify-outcomes",
        help="recompute the four outcomes from Doell and check Vlasceanu",
    )

    build = sub.add_parser("build-profiles", help="draw synthetic respondents")
    build.add_argument("--seed", type=int, default=20260823)
    build.add_argument("--per-arm", type=int, default=None)
    build.add_argument("--out", default=None)

    dry = sub.add_parser("dry-run", help="drive one respondent per arm to the end")
    dry.add_argument("--out", default=None, help="write one filled transcript here")

    human = sub.add_parser("human-reference", help="the human side of the scoreboard")
    human.add_argument("--seed", type=int, default=42)
    human.add_argument("--out", default=None, help="directory for the CSVs")

    args = parser.parse_args(argv)
    pd.set_option("display.width", 220)

    if args.command == "render-templates":
        from .templates import render_all

        manifest = render_all()
        print(f"wrote {len(manifest['arms'])} templates to {paths.TEMPLATES}")
        for key, entry in manifest["arms"].items():
            print(f"  {entry['file']:34s} slots={entry['n_slots']:4d}  {key}")
        return 0

    if args.command == "modality-audit":
        from .templates import write_modality_audit

        rows = write_modality_audit()
        table = pd.DataFrame(rows)
        print(
            table[
                [
                    "cond",
                    "condition",
                    "decision",
                    "n_screens",
                    "chars",
                    "video",
                    "audio",
                    "image",
                    "iframe",
                    "script",
                ]
            ].to_string(index=False)
        )
        print(f"\nwrote {paths.MODALITY_AUDIT}")
        return 0

    if args.command == "verify-outcomes":
        from .score import verify_items, verify_outcomes

        table = verify_outcomes()
        print(table.to_string(index=False))
        items = verify_items()
        print("\nPer item, which is what a permuted battery would fail")
        print(
            items[
                [
                    "outcome",
                    "item",
                    "published_column",
                    "n_compared",
                    "mean_ours",
                    "mean_published",
                    "max_abs_diff",
                    "matches",
                ]
            ]
            .round(4)
            .to_string(index=False)
        )
        return 0 if bool(table["matches"].all() and items["matches"].all()) else 1

    if args.command == "build-profiles":
        built = profiles.build(seed=args.seed, per_arm=args.per_arm)
        out = Path(args.out) if args.out else paths.PROFILES_CSV
        profiles.write_csv(built, out)
        print(f"wrote {len(built)} profiles to {out}")
        summary = profiles.sanity(built)
        print(f"  per arm      : {sorted(set(summary['per_arm'].values()))}")
        print(f"  gender       : {summary['gender']}")
        print(f"  age bands    : {summary['age_band']}")
        print(f"  DV orders    : {summary['battery_orders']} of 6")
        return 0

    if args.command == "dry-run":
        from .validate import dry_run_all

        runs = dry_run_all()
        for run in runs:
            print(
                f"  {run.condition:36s} asked={run.n_asked:4d} "
                f"chars={len(run.transcript):6d}"
            )
        if args.out:
            path = Path(args.out)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(runs[0].transcript, encoding="utf-8")
            print(f"wrote {path}")
        return 0

    if args.command == "human-reference":
        from .score import human_reference

        result = human_reference(seed=args.seed)
        print("Respondents and outcome means per arm (US quota subsample)")
        print(result["counts"].to_string(index=False))
        print("\nTreatment effects, full US sample (points of each outcome's scale)")
        wide = result["effects_all"].pivot(
            index="condition", columns="outcome", values="estimate"
        )
        print(wide.round(2).to_string())
        print("\nScoreboard against Human 1")
        columns = [
            "submission",
            "n_pairs",
            "directional_pct",
            "pearson_r",
            "pearson_adj",
            "rmse",
            "beta",
        ]
        print(result["board"][columns].round(3).to_string(index=False))
        if args.out:
            out = Path(args.out)
            out.mkdir(parents=True, exist_ok=True)
            result["counts"].to_csv(out / "human_arm_counts.csv", index=False)
            result["effects_all"].to_csv(out / "human_effects_all.csv", index=False)
            result["effects"].to_csv(out / "human_effects_half1.csv", index=False)
            result["board"].to_csv(out / "human_reference_board.csv", index=False)
            print(f"\nwrote CSVs to {out}")
        return 0

    return 1  # pragma: no cover


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
