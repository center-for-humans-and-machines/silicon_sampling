"""Command line for the Goldwert calibration study.

Most subcommands are cheap desk work — the modality audit, the template render,
the outcome verification, the human reference, the Pfaender anchor table.  Two are
not: ``sample`` drives a model over the profiles and ``build-csv`` turns its answer
log into the analysis frame.

    python -m silicon_sampling.goldwert.cli build-profiles
    python -m silicon_sampling.goldwert.cli sample --run qwen25_7b
    python -m silicon_sampling.goldwert.cli build-csv --run qwen25_7b

``sample`` re-enters cleanly: ``answers.jsonl`` is the source of truth for what is
finished, so running it again asks only for the rest.  That is why the cluster
wrapper is a retry loop and nothing more.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from ..models import DEFAULT_RUN, MODELS, engine_defaults, model_id
from ..sampling.driver import SamplerConfig
from ..sampling.engine import EngineConfig, cache_root
from . import outcomes as oc
from . import paths, profiles

#: Default context cap for a sampling run.  The longest arm runs to about 7.9k
#: tokens, which makes 8192 the one value that looks safe and is not: it would
#: fail on exactly that arm.  ``tests/test_goldwert.py`` measures the worst case
#: against this number rather than trusting it.
MAX_MODEL_LEN = 12288


def _profiles(args, run_dir: Path | None = None) -> list[profiles.Profile]:
    """The respondents to sample, and a copy of them filed with the run.

    Every model samples the *same* profiles with the same seeds, which is what
    makes two runs comparable respondent by respondent; so a second run reads the
    profiles the first one built rather than drawing its own.
    """
    path = Path(args.profiles) if args.profiles else paths.PROFILES_CSV
    if not path.exists():
        raise SystemExit(f"no profiles at {path}; run build-profiles first")
    everyone = profiles.read_csv(path)
    if run_dir is not None and not (run_dir / "profiles.csv").exists():
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "profiles.csv").write_bytes(path.read_bytes())
        print(f"[run] copied {len(everyone)} profiles from {path}")
    if args.limit:
        # An arm-stratified slice, so a pilot still touches as many of the eleven
        # usable arms as it has respondents for.
        by_arm: dict[str, list] = {}
        for profile in everyone:
            by_arm.setdefault(profile.condition, []).append(profile)
        per_arm = max(1, args.limit // len(by_arm))
        picked = [profile for group in by_arm.values() for profile in group[:per_arm]]
        return sorted(picked, key=lambda p: p.profile_id)[: args.limit]
    return everyone


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="silicon_sampling.goldwert")
    sub = parser.add_subparsers(dest="command", required=True)

    audit = sub.add_parser(
        "modality-audit", help="per-arm media counts and the keep-or-drop call"
    )
    audit.add_argument("--out", default=None)

    sub.add_parser("render-templates", help="write data/Goldwert/text_templates")

    dry = sub.add_parser(
        "validate", help="drive one respondent per usable arm to the end"
    )
    dry.add_argument("--out", default=None, help="write one filled transcript here")

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

    sample = sub.add_parser("sample", help="run the model over the profiles")
    sample.add_argument("--profiles", default=None)
    sample.add_argument(
        "--out", default=None, help="output directory (default: the run key's dir)"
    )
    sample.add_argument(
        "--limit",
        type=int,
        default=0,
        help="pilot on this many respondents, stratified by arm",
    )
    sample.add_argument(
        "--group-size",
        type=int,
        default=0,
        help="0 = size it automatically, per arm, from the KV cache the engine gets",
    )
    sample.add_argument("--draws-per-call", type=int, default=4)
    sample.add_argument(
        "--run",
        default=DEFAULT_RUN,
        help=f"which model's run to write ({', '.join(sorted(MODELS))})",
    )
    sample.add_argument(
        "--model", default=None, help="override the run key's Hugging Face id"
    )
    sample.add_argument(
        "--kv-cache-dtype",
        default=None,
        choices=["auto", "fp8", "fp8_ds_mla"],
        help="default: whatever the run key requires (see models.ENGINE_DEFAULTS)",
    )
    sample.add_argument("--max-model-len", type=int, default=MAX_MODEL_LEN)
    sample.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    sample.add_argument("--max-num-seqs", type=int, default=256)
    sample.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="GPUs to shard the weights over; >1 disables the in-process engine",
    )
    sample.add_argument(
        "--expert-parallel",
        action="store_true",
        default=False,
        help="route MoE experts across ranks instead of slicing every expert",
    )
    sample.add_argument(
        "--no-token-ids",
        dest="token_id_prompts",
        action="store_false",
        default=True,
        help="submit prompts as text, making vLLM re-tokenise every transcript",
    )
    sample.add_argument(
        "--eager",
        dest="enforce_eager",
        action="store_true",
        default=False,
        help="disable torch.compile + CUDA graphs (faster startup, slower decode)",
    )
    sample.add_argument("--no-resume", action="store_true")

    csv_cmd = sub.add_parser(
        "build-csv", help="answers.jsonl -> samples.csv with every scored outcome"
    )
    csv_cmd.add_argument("--out", default=None)
    csv_cmd.add_argument("--run", default=DEFAULT_RUN)

    args = parser.parse_args(argv)
    pd.set_option("display.width", 220)

    if args.command == "validate":
        from .validate import dry_run_all

        runs = dry_run_all()
        for run in runs:
            print(
                f"  {run.condition:26s} asked={run.n_asked:4d} "
                f"chars={len(run.transcript):6d}"
            )
        if args.out:
            path = Path(args.out)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(runs[0].transcript, encoding="utf-8")
            print(f"wrote {path}")
        return 0

    if args.command == "sample":
        from ..sampling.engine import configure_runtime

        tp = args.tensor_parallel_size
        # Tensor parallelism *is* vLLM's worker processes, so the in-process
        # engine and TP > 1 are mutually exclusive.
        configure_runtime(cache_root(paths.DATA.parent / ".cache"), in_process=tp == 1)
        from .run import Runner

        out = Path(args.out) if args.out else paths.samples_dir(args.run)
        runner = Runner(
            out,
            EngineConfig(
                model=args.model or model_id(args.run),
                max_model_len=args.max_model_len,
                gpu_memory_utilization=args.gpu_memory_utilization,
                kv_cache_dtype=(
                    args.kv_cache_dtype
                    or engine_defaults(args.run).get("kv_cache_dtype", "auto")
                ),
                enforce_eager=args.enforce_eager,
                max_num_seqs=args.max_num_seqs,
                tensor_parallel_size=tp,
                enable_expert_parallel=args.expert_parallel,
            ),
            SamplerConfig(
                group_size=args.group_size,
                draws_per_call=args.draws_per_call,
                token_id_prompts=args.token_id_prompts,
            ),
        )
        runner.run(_profiles(args, out), resume=not args.no_resume)
        return 0

    if args.command == "build-csv":
        from .export import build_csvs

        out = Path(args.out) if args.out else paths.samples_dir(args.run)
        print(json.dumps(build_csvs(out), indent=2, default=str))
        return 0

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
