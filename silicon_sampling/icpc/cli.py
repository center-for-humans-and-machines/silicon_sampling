"""Command line for the ICPC study package.

Most of it is cheap and runs on a laptop: rendering templates, auditing the
stimulus modality, verifying the outcome construction against the second
publication, drawing profiles, and computing the human reference a silicon sample
is scored against.  Two subcommands are not: ``sample`` drives a model over the
profiles and ``build-csv`` turns its answer log into the analysis frame.

    python -m silicon_sampling.icpc.cli build-profiles
    python -m silicon_sampling.icpc.cli sample --run qwen25_7b
    python -m silicon_sampling.icpc.cli build-csv --run qwen25_7b

``sample`` re-enters cleanly: ``answers.jsonl`` is the source of truth for what is
finished, so running it again asks only for the rest.  That is why the cluster
wrapper is a retry loop and nothing more.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ..lazy import lazy_module

from ..models import DEFAULT_RUN, MODELS, engine_defaults, model_id
from ..sampling.driver import SamplerConfig
from ..sampling.engine import EngineConfig, cache_root
from . import paths, profiles

#: Imported on first use.  The sampling entry point is this module, and it has
#: to run in the Muse-Glimmer container, which ships no pandas; only the
#: reporting subcommands here touch it.
pd = lazy_module("pandas")

#: Default context cap for a sampling run.  The longest transcript this instrument
#: can produce is about 16k tokens -- twice the Pfaender figure, because every
#: respondent answers a thirteen-item battery, an eight-page effort task and a
#: demographic block on top of a stimulus.  A cap below that does not throttle
#: throughput, it makes the longest arm fail outright, so ``tests/test_icpc.py``
#: measures the worst case against this number rather than trusting it.
MAX_MODEL_LEN = 20480


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
        # An arm-stratified slice, so a pilot still touches as many of the twelve
        # arms as it has respondents for.
        by_arm: dict[str, list] = {}
        for profile in everyone:
            by_arm.setdefault(profile.condition, []).append(profile)
        per_arm = max(1, args.limit // len(by_arm))
        picked = [profile for group in by_arm.values() for profile in group[:per_arm]]
        return sorted(picked, key=lambda p: p.profile_id)[: args.limit]
    return everyone


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
        "build-csv", help="answers.jsonl -> samples.csv with the four outcomes"
    )
    csv_cmd.add_argument("--out", default=None)
    csv_cmd.add_argument("--run", default=DEFAULT_RUN)

    args = parser.parse_args(argv)
    pd.set_option("display.width", 220)

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
