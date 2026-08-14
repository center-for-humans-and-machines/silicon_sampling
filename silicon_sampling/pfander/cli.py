"""Command line for the Pfänder silicon-sampling run.

python -m silicon_sampling.pfander.cli render-templates
python -m silicon_sampling.pfander.cli validate
python -m silicon_sampling.pfander.cli build-profiles
python -m silicon_sampling.pfander.cli sample --limit 64 --group-size 24
python -m silicon_sampling.pfander.cli build-csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ..sampling.driver import SamplerConfig
from ..sampling.engine import EngineConfig
from . import paths, profiles, templates, validate


def _profiles(args) -> list[profiles.Profile]:
    path = Path(args.profiles) if args.profiles else paths.PROFILES_CSV
    if not path.exists():
        raise SystemExit(f"no profiles at {path}; run build-profiles first")
    everyone = profiles.read_csv(path)
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
    sample.add_argument("--group-size", type=int, default=24)
    sample.add_argument("--draws-per-call", type=int, default=4)
    sample.add_argument("--model", default=EngineConfig.model)
    sample.add_argument("--kv-cache-dtype", default="auto", choices=["auto", "fp8"])
    sample.add_argument("--max-model-len", type=int, default=8192)
    sample.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    sample.add_argument(
        "--compile",
        dest="enforce_eager",
        action="store_false",
        default=True,
        help="use torch.compile + CUDA graphs (slower startup)",
    )
    sample.add_argument("--no-resume", action="store_true")

    csv_cmd = sub.add_parser(
        "build-csv", help="answers.jsonl -> samples.csv + tier1_submission.csv"
    )
    csv_cmd.add_argument("--out", default=None)

    args = parser.parse_args(argv)

    if args.command == "render-templates":
        manifest = templates.render_all()
        print(f"wrote {len(manifest['conditions'])} templates to {paths.TEMPLATES}")
        return 0

    if args.command == "validate":
        return validate.main()

    if args.command == "build-profiles":
        built = profiles.build(total=args.n, seed=args.seed)
        profiles.write_csv(built, paths.PROFILES_CSV)
        print(f"wrote {len(built)} profiles to {paths.PROFILES_CSV}")
        return 0

    if args.command == "sample":
        from ..sampling.engine import configure_runtime

        configure_runtime(paths.CACHE)
        from .run import Runner

        out = Path(args.out) if args.out else paths.SAMPLES
        runner = Runner(
            out,
            EngineConfig(
                model=args.model,
                max_model_len=args.max_model_len,
                gpu_memory_utilization=args.gpu_memory_utilization,
                kv_cache_dtype=args.kv_cache_dtype,
                enforce_eager=args.enforce_eager,
            ),
            SamplerConfig(
                group_size=args.group_size, draws_per_call=args.draws_per_call
            ),
        )
        runner.run(_profiles(args), resume=not args.no_resume)
        return 0

    if args.command == "build-csv":
        from .export import build_csvs

        out = Path(args.out) if args.out else paths.SAMPLES
        summary = build_csvs(out)
        print(summary)
        return 0

    return 1  # pragma: no cover


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
