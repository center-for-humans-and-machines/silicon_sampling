"""Command line for the Pfänder silicon-sampling run.

python -m silicon_sampling.pfander.cli render-templates
python -m silicon_sampling.pfander.cli validate
python -m silicon_sampling.pfander.cli build-profiles
python -m silicon_sampling.pfander.cli sample            # group size auto-fits the KV cache
python -m silicon_sampling.pfander.cli build-csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ..models import DEFAULT_RUN, MODELS, engine_defaults, model_id
from ..sampling.driver import SamplerConfig
from ..sampling.engine import EngineConfig
from . import paths, profiles, templates, validate


def _profiles(args, run_dir: Path | None = None) -> list[profiles.Profile]:
    """The respondents to sample, and a copy of them filed with the run.

    Every model samples the *same* profiles with the same seeds, which is what
    makes two runs comparable respondent by respondent; so a new run reads the
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
        # Take a condition-stratified slice, so a pilot still covers all 17 arms.
        by_condition: dict[str, list] = {}
        for profile in everyone:
            by_condition.setdefault(profile.condition, []).append(profile)
        per_arm = max(1, args.limit // len(by_condition))
        picked = [
            profile for group in by_condition.values() for profile in group[:per_arm]
        ]
        return sorted(picked, key=lambda p: p.profile_id)[: args.limit]
    return everyone


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="silicon_sampling.pfander")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("render-templates", help="write data/pfander/text_templates")
    sub.add_parser("validate", help="check the instrument before spending GPU time")

    build = sub.add_parser(
        "build-profiles", help="apportion respondents to the preregistered quotas"
    )
    build.add_argument("--n", type=int, default=profiles.TOTAL_N)
    build.add_argument("--seed", type=int, default=20260814)
    build.add_argument(
        "--out",
        default=None,
        help="where to write (default: the qwen25_7b run's profiles.csv)",
    )
    build.add_argument(
        "--overwrite",
        action="store_true",
        help="replace an existing profiles.csv; refused by default",
    )

    sample = sub.add_parser("sample", help="run the model over the profiles")
    sample.add_argument("--profiles", default=None)
    sample.add_argument(
        "--out", default=None, help="output directory (default: the qwen25_7b run dir)"
    )
    sample.add_argument(
        "--limit",
        type=int,
        default=0,
        help="pilot on this many respondents, stratified by condition",
    )
    sample.add_argument(
        "--group-size",
        type=int,
        default=0,
        help="0 = size it automatically from the KV cache the engine gets",
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
    sample.add_argument("--max-model-len", type=int, default=8192)
    sample.add_argument("--gpu-memory-utilization", type=float, default=0.92)
    sample.add_argument("--max-num-seqs", type=int, default=512)
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
        help="disable torch.compile + CUDA graphs (faster startup, much slower decode)",
    )
    sample.add_argument("--no-resume", action="store_true")

    csv_cmd = sub.add_parser(
        "build-csv", help="answers.jsonl -> samples.csv + tier1_submission.csv"
    )
    csv_cmd.add_argument("--out", default=None)
    csv_cmd.add_argument("--run", default=DEFAULT_RUN)

    cmp_cmd = sub.add_parser(
        "compare", help="one model's run against another (docs/reports/...)"
    )
    cmp_cmd.add_argument(
        "--runs",
        nargs="+",
        default=["qwen25_7b", "v4_flash"],
        help="run keys, baseline first",
    )
    cmp_cmd.add_argument("--out", default=None)

    stats_cmd = sub.add_parser(
        "stats", help="live throughput and illegal-generation rate of a run"
    )
    stats_cmd.add_argument("--out", default=None)
    stats_cmd.add_argument(
        "--window",
        type=float,
        default=120.0,
        help="seconds to watch the answer log for an instantaneous rate",
    )
    stats_cmd.add_argument("--run", default=DEFAULT_RUN)
    stats_cmd.add_argument("--model", default=None)
    stats_cmd.add_argument("--json", action="store_true")

    args = parser.parse_args(argv)

    if args.command == "render-templates":
        manifest = templates.render_all()
        print(f"wrote {len(manifest['conditions'])} templates to {paths.TEMPLATES}")
        return 0

    if args.command == "validate":
        return validate.main()

    if args.command == "build-profiles":
        # The default target sits inside a *finished* run's directory, and that
        # file is the only record of which respondent a sampled transcript
        # belongs to.  Silently replacing it would make a completed run
        # unreproducible, and the schema now differs between a prefilled build
        # and the ones on disk, so an accidental rerun would not even be a
        # no-op.  Overwriting therefore has to be asked for.
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

    if args.command == "sample":
        from ..sampling.engine import configure_runtime

        tp = args.tensor_parallel_size
        # Tensor parallelism *is* vLLM's worker processes, so the in-process
        # engine and TP > 1 are mutually exclusive.
        configure_runtime(paths.CACHE, in_process=tp == 1)
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
        summary = build_csvs(out)
        print(summary)
        return 0

    if args.command == "compare":
        from .compare import generate

        if len(args.runs) < 2:
            raise SystemExit("compare needs at least two run keys")
        result = generate(
            args.runs,
            out=Path(args.out) if args.out else paths.REPORT,
            baseline=args.runs[0],
            contender=args.runs[1],
        )
        print(result["agreement"].to_string(index=False))
        print()
        print(result["demographics"].to_string(index=False))
        return 0

    if args.command == "stats":
        import json as _json

        from ..sampling.engine import configure_runtime

        configure_runtime(paths.CACHE)

        from .stats import format_report, report

        out = Path(args.out) if args.out else paths.samples_dir(args.run)
        result = report(out, model=args.model or model_id(args.run), window=args.window)
        print(_json.dumps(result, indent=2) if args.json else format_report(result))
        return 0

    return 1  # pragma: no cover


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
