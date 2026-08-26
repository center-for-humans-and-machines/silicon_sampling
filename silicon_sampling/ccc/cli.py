"""Command line for the Climate Change Challenge silicon-sampling run."""

from __future__ import annotations

import argparse
from pathlib import Path

from ..models import DEFAULT_RUN, MODELS, engine_defaults, model_id
from ..sampling.driver import SamplerConfig
from ..sampling.engine import EngineConfig, cache_root
from . import paths, profiles


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="silicon_sampling.ccc")
    sub = parser.add_subparsers(dest="command", required=True)

    build = sub.add_parser(
        "build-profiles",
        help="resample respondents from the study's own participants",
    )
    build.add_argument("--seed", type=int, default=20260826)
    build.add_argument(
        "--n",
        type=int,
        default=None,
        help="total profiles (default: mirror the retained human sample exactly)",
    )
    build.add_argument("--out", default=None)
    build.add_argument("--overwrite", action="store_true")

    sub.add_parser("render-templates", help="write the arm templates and manifest")
    sub.add_parser(
        "validate", help="range and composite checks; must pass before sampling"
    )
    sub.add_parser("modality-audit", help="per-arm media loss table")

    sample = sub.add_parser("sample", help="run the model over the profiles")
    sample.add_argument("--out", default=None)
    sample.add_argument("--limit", type=int, default=0)
    sample.add_argument("--group-size", type=int, default=0)
    sample.add_argument("--draws-per-call", type=int, default=4)
    sample.add_argument("--profiles", default=None, help="profiles.csv to sample")
    sample.add_argument(
        "--run",
        default=DEFAULT_RUN,
        help=f"which model's run to write ({', '.join(sorted(MODELS))})",
    )
    sample.add_argument("--model", default=None)
    sample.add_argument("--max-model-len", type=int, default=16384)
    sample.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    sample.add_argument("--max-num-seqs", type=int, default=128)
    sample.add_argument("--max-num-batched-tokens", type=int, default=4096)
    sample.add_argument(
        "--kv-cache-dtype", default=None, choices=["auto", "fp8", "fp8_ds_mla"]
    )
    sample.add_argument("--tensor-parallel-size", type=int, default=1)
    sample.add_argument("--expert-parallel", action="store_true")
    sample.add_argument("--enforce-eager", dest="enforce_eager", action="store_true")
    sample.add_argument(
        "--no-token-ids", dest="token_id_prompts", action="store_false", default=True
    )
    sample.add_argument("--no-resume", action="store_true")

    csv = sub.add_parser("build-csv", help="answers.jsonl -> samples.csv")
    csv.add_argument("--run", default=DEFAULT_RUN)

    args = parser.parse_args(argv)

    if args.command == "build-profiles":
        target = Path(args.out) if args.out else paths.PROFILES_CSV
        if target.exists() and not args.overwrite:
            raise SystemExit(
                f"{target} already exists and is a finished run's provenance; "
                "pass --out to write elsewhere, or --overwrite to replace it"
            )
        built = profiles.build(total=args.n, seed=args.seed)
        profiles.write_csv(built, target)
        print(f"wrote {len(built)} profiles to {target}")
        return 0

    if args.command == "render-templates":
        from .templates import render_all

        manifest = render_all()
        for condition, info in manifest["conditions"].items():
            print(
                f"  {info['file']:34s} {info['n_slots']:4d} slots "
                f"{info['chars']:7d} chars  media_loss={info['media_loss']}"
            )
        for arm, why in manifest["dropped"].items():
            print(f"  -- dropped {arm}: {why}")
        return 0

    if args.command == "validate":
        from .validate import main as validate_main

        return validate_main()

    if args.command == "modality-audit":
        from .audit import main as audit_main

        return audit_main()

    if args.command == "build-csv":
        from .export import build_csv

        report = build_csv(args.run)
        print(f"rows {report['rows']}  columns {report['columns']}")
        print(f"wrote {report['csv']}")
        for name, n in report["complete_per_scored_outcome"].items():
            print(f"  {name:18s} complete for {n} respondents")
        return 0

    if args.command == "sample":
        from ..sampling.engine import configure_runtime

        tp = args.tensor_parallel_size
        configure_runtime(cache_root(), in_process=tp == 1)
        from .run import make_runner

        source = Path(args.profiles) if args.profiles else paths.PROFILES_CSV
        everyone = profiles.read_csv(source)
        if args.limit:
            by_condition: dict[str, list] = {}
            for profile in everyone:
                by_condition.setdefault(profile.condition, []).append(profile)
            per_arm = max(1, args.limit // len(by_condition))
            picked = [p for group in by_condition.values() for p in group[:per_arm]]
            everyone = sorted(picked, key=lambda p: p.profile_id)[: args.limit]
        out = Path(args.out) if args.out else paths.samples_dir(args.run)
        if not (out / "profiles.csv").exists():
            out.mkdir(parents=True, exist_ok=True)
            (out / "profiles.csv").write_bytes(source.read_bytes())
            print(f"[run] copied {len(everyone)} profiles from {source}")
        runner = make_runner(
            out,
            EngineConfig(
                model=args.model or model_id(args.run),
                max_model_len=args.max_model_len,
                gpu_memory_utilization=args.gpu_memory_utilization,
                enforce_eager=args.enforce_eager,
                max_num_seqs=args.max_num_seqs,
                max_num_batched_tokens=args.max_num_batched_tokens,
                kv_cache_dtype=(
                    args.kv_cache_dtype
                    or engine_defaults(args.run).get("kv_cache_dtype", "auto")
                ),
                tensor_parallel_size=tp,
                enable_expert_parallel=args.expert_parallel,
            ),
            SamplerConfig(
                group_size=args.group_size,
                draws_per_call=args.draws_per_call,
                token_id_prompts=args.token_id_prompts,
            ),
        )
        runner.run(everyone, resume=not args.no_resume)
        return 0

    parser.error(f"unknown command {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
