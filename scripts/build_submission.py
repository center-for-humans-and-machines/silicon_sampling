"""Package a finished run into a submission directory, and validate it.

Two subcommands, in the order you need them:

    python scripts/build_submission.py build \\
        --predictions data/pfander/silicon_sampling/qwen25_7b/tier1_submission.csv \\
        --out data/pfander/submission/qwen25_7b \\
        --raw-export data/pfander/silicon_sampling/qwen25_7b/samples.csv \\
        --run-meta data/pfander/silicon_sampling/qwen25_7b/run_meta.json

    python scripts/build_submission.py check data/pfander/submission/qwen25_7b

``build`` stages the template's own files (``registration.md``, ``codebook.csv``,
``survey/``) alongside the prediction file, so ``check`` sees a complete repo.
Nothing is overwritten unless ``--overwrite`` says so: the data tree is
git-ignored, so a clobbered submission cannot be recovered.

``check`` is the Python port of ``make check`` and exits non-zero on a FAIL,
mirroring ``scripts/check.R``.  Warnings do not change the exit status; read
them anyway — every one of them is a real property of the file.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from silicon_sampling.submission import build, check, spec


def _spec_preamble() -> None:
    """Report any drift between our hardcoded schema and the shipped materials."""
    problems = spec.verify_against_codebook()
    if problems:
        print("spec vs shipped materials:")
        for problem in problems:
            print(f"  ! {problem}")
        print()


def do_build(args) -> int:
    meta = build.SubmissionMeta(
        team_id=args.team_id,
        entry=args.entry,
        version=args.version,
        models=build.models_from_run_meta(args.run_meta) if args.run_meta else (),
    )
    result = build.build_submission(
        args.predictions,
        args.out,
        meta,
        raw_export=args.raw_export,
        template_root=args.template or spec.default_template_root(),
        overwrite=args.overwrite,
    )
    print(result.summary())
    if not args.check:
        return 0
    print()
    return do_check(argparse.Namespace(root=args.out, report=args.report))


def do_check(args) -> int:
    _spec_preamble()
    result = check.check_repo(Path(args.root))
    check.print_report(result)
    if args.report:
        path = check.write_report(result, Path(args.report))
        print(f"\nreport written to {path}")
    return 0 if result.passed else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="build_submission")
    sub = parser.add_subparsers(dest="command", required=True)

    builder = sub.add_parser("build", help="assemble a submission directory")
    builder.add_argument(
        "--predictions", required=True, help="respondent-level Tier-1 CSV"
    )
    builder.add_argument("--out", required=True, help="submission directory to create")
    builder.add_argument("--raw-export", default=None, help="raw run output to deposit")
    builder.add_argument(
        "--run-meta", default=None, help="run_meta.json, for the models list"
    )
    builder.add_argument(
        "--template", default=None, help="submission-template checkout to stage from"
    )
    builder.add_argument("--team-id", default="mpib")
    builder.add_argument("--entry", default="primary")
    builder.add_argument("--version", type=int, default=1)
    builder.add_argument("--overwrite", action="store_true")
    builder.add_argument(
        "--check", action="store_true", help="run the checker afterwards"
    )
    builder.add_argument(
        "--report", default=None, help="write the check report to this path"
    )
    builder.set_defaults(func=do_build)

    checker = sub.add_parser("check", help="validate a submission directory")
    checker.add_argument("root")
    checker.add_argument(
        "--report", default=None, help="write the check report to this path"
    )
    checker.set_defaults(func=do_check)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
