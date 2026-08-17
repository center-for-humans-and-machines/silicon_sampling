"""Command line for the Voelkel silicon-sampling run."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ..models import DEFAULT_RUN, MODELS, model_id
from ..sampling.driver import SamplerConfig
from ..sampling.engine import EngineConfig, cache_root
from . import paths, profiles


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="silicon_sampling.voelkel")
    sub = parser.add_subparsers(dest="command", required=True)

    build = sub.add_parser(
        "build-profiles", help="draw respondents from the published marginals"
    )
    build.add_argument("--seed", type=int, default=20260815)
    build.add_argument("--per-intervention", type=int, default=None)

    sub.add_parser("render-templates", help="write data/Voelkel/text_templates")

    sample = sub.add_parser("sample", help="run the model over the profiles")
    sample.add_argument("--out", default=None)
    sample.add_argument("--limit", type=int, default=0)
    sample.add_argument("--group-size", type=int, default=0)
    sample.add_argument("--draws-per-call", type=int, default=4)
    sample.add_argument(
        "--run",
        default=DEFAULT_RUN,
        help=f"which model's run to write ({', '.join(sorted(MODELS))})",
    )
    sample.add_argument(
        "--model", default=None, help="override the run key's Hugging Face id"
    )
    sample.add_argument("--max-model-len", type=int, default=12288)
    sample.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    sample.add_argument("--max-num-seqs", type=int, default=128)
    sample.add_argument("--max-num-batched-tokens", type=int, default=4096)
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
        "--eager", dest="enforce_eager", action="store_true", default=False
    )
    sample.add_argument("--no-resume", action="store_true")

    csv_cmd = sub.add_parser(
        "build-csv", help="answers.jsonl -> samples.csv with the nine outcomes"
    )
    csv_cmd.add_argument("--out", default=None)
    csv_cmd.add_argument("--run", default=DEFAULT_RUN)

    score_cmd = sub.add_parser(
        "score", help="score the sample against the real responses"
    )
    score_cmd.add_argument("--out", default=None)
    score_cmd.add_argument(
        "--runs",
        nargs="+",
        default=[DEFAULT_RUN],
        help="run keys to score; more than one puts both models on the board",
    )

    args = parser.parse_args(argv)

    if args.command == "build-profiles":
        built = profiles.build(seed=args.seed, per_intervention=args.per_intervention)
        profiles.write_csv(built, paths.PROFILES_CSV)
        print(f"wrote {len(built)} profiles to {paths.PROFILES_CSV}")
        return 0

    if args.command == "render-templates":
        from .templates import render_all

        manifest = render_all()
        print(f"wrote {len(manifest['conditions'])} templates to {paths.TEMPLATES}")
        return 0

    if args.command == "sample":
        from ..sampling.engine import configure_runtime

        tp = args.tensor_parallel_size
        # Tensor parallelism *is* vLLM's worker processes, so the in-process
        # engine and TP > 1 are mutually exclusive.
        configure_runtime(cache_root(), in_process=tp == 1)
        from .run import make_runner

        everyone = profiles.read_csv(paths.PROFILES_CSV)
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
            (out / "profiles.csv").write_bytes(paths.PROFILES_CSV.read_bytes())
            print(f"[run] copied {len(everyone)} profiles from {paths.PROFILES_CSV}")
        runner = make_runner(
            out,
            EngineConfig(
                model=args.model or model_id(args.run),
                max_model_len=args.max_model_len,
                gpu_memory_utilization=args.gpu_memory_utilization,
                enforce_eager=args.enforce_eager,
                max_num_seqs=args.max_num_seqs,
                max_num_batched_tokens=args.max_num_batched_tokens,
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

    if args.command == "build-csv":
        from .export import build_csv

        summary = build_csv(Path(args.out) if args.out else paths.samples_dir(args.run))
        print(summary)
        return 0

    if args.command == "score":
        from .report import generate

        result = generate(out=Path(args.out) if args.out else paths.REPORT)
        print(result["board"].to_string(index=False))
        return 0

    return 1  # pragma: no cover


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
